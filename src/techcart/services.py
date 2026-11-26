"""Domain services shared by the REST API, the agents' native tools and the MCP servers."""
from __future__ import annotations

import json
import threading
import uuid
from datetime import datetime, timedelta
from functools import lru_cache

from .db import connect, rows

_conn = None
_lock = threading.Lock()


def conn():
    global _conn
    if _conn is None:
        _conn = connect()
    return _conn


def q(sql: str, params=()) -> list[dict]:
    with _lock:
        return rows(conn(), sql, params)


def execute(sql: str, params=()) -> None:
    with _lock:
        conn().execute(sql, params)
        conn().commit()


# ------------------------------------------------------------------------------- catalog
@lru_cache(maxsize=1)
def _products_cached() -> tuple:
    return tuple(q("SELECT * FROM products"))


def all_products() -> list[dict]:
    return list(_products_cached())


def product(pid: str) -> dict | None:
    return next((p for p in _products_cached() if p["product_id"] == pid), None)


SORTS = {
    "popular": lambda p: (-(p["rating_count"] or 0), p["price"]),
    "price_asc": lambda p: p["price"],
    "price_desc": lambda p: -p["price"],
    "rating": lambda p: (-(p["rating"] or 0), -(p["rating_count"] or 0)),
}


def catalog_counts() -> list[dict]:
    """Categories with product counts and sub-category counts (for the storefront navigation)."""
    from collections import Counter

    from .data.catalog import CATEGORIES

    cats = Counter(p["category"] for p in _products_cached())
    subs: dict[str, Counter] = {}
    for p in _products_cached():
        subs.setdefault(p["category"], Counter())[p["subcategory"]] += 1
    order = {c: i for i, c in enumerate(CATEGORIES)}
    return [{"category": c, "count": n, "subcategories": [{"name": s, "count": k} for s, k in subs[c].most_common()]}
            for c, n in sorted(cats.items(), key=lambda x: order.get(x[0], 99))]


def list_products(category: str | None = None, subcategory: str | None = None, sort: str = "popular",
                  offset: int = 0, limit: int = 12) -> dict:
    items = [p for p in _products_cached() if (not category or p["category"] == category)
             and (not subcategory or p["subcategory"] == subcategory)]
    items.sort(key=SORTS.get(sort, SORTS["popular"]))
    stock = stock_map()
    page = items[offset:offset + limit]
    return {"category": category, "subcategory": subcategory, "sort": sort, "total": len(items), "offset": offset,
            "items": [product_card(p, stock) for p in page], "has_more": offset + limit < len(items)}


def stock_map() -> dict[str, int]:
    return {r["product_id"]: r["stock_on_hand"] - r["reserved"] for r in q("SELECT * FROM inventory")}


def product_card(p: dict, stock: dict | None = None) -> dict:
    stock = stock if stock is not None else stock_map()
    s = stock.get(p["product_id"], 0)
    return {"product_id": p["product_id"], "title": p["title"], "brand": p["brand"], "category": p["category"],
            "subcategory": p["subcategory"], "price": p["price"], "rating": p["rating"],
            "rating_count": p["rating_count"], "in_stock": s > 0, "stock": s,
            "stock_label": "Out of stock" if s <= 0 else f"Only {s} left" if s < 5 else "In stock",
            "specs": json.loads(p["specs_json"] or "{}"), "image_url": p.get("image_url")}


# ------------------------------------------------------------------------------- customers & orders
def customer(cid: str) -> dict | None:
    r = q("SELECT customer_id, name, city, segment, is_business, marketing_consent FROM customers WHERE customer_id=?",
          (cid,))
    return r[0] if r else None


def customer_history(cid: str) -> list[tuple[str, datetime]]:
    r = q("""SELECT i.product_id, o.created_at FROM order_items i JOIN orders o USING(order_id)
             WHERE o.customer_id=? AND o.status!='cancelled' ORDER BY o.created_at""", (cid,))
    return [(x["product_id"], datetime.fromisoformat(x["created_at"])) for x in r]


def order_status(order_id: str, customer_id: str | None = None) -> dict | None:
    r = q("SELECT order_id, customer_id, created_at, status, total, ship_city, payment_method FROM orders WHERE order_id=?",
          (order_id,))
    if not r:
        return None
    o = r[0]
    if customer_id and o["customer_id"] != customer_id:  # authorisation: customers only see their own orders
        return {"error": "Order not found for this account"}
    o["items"] = q("""SELECT p.title, i.qty, i.line_total FROM order_items i JOIN products p USING(product_id)
                      WHERE order_id=?""", (order_id,))
    created = datetime.fromisoformat(o["created_at"])
    o["estimated_delivery"] = (created + timedelta(days=1 if o["ship_city"] == "Riyadh" else 3)).date().isoformat()
    return o


def recent_orders(cid: str, n: int = 5) -> list[dict]:
    return q("SELECT order_id, created_at, status, total FROM orders WHERE customer_id=? ORDER BY created_at DESC LIMIT ?",
             (cid, n))


# ------------------------------------------------------------------------------- cart (server-side, per session)
CARTS: dict[str, dict[str, int]] = {}


def cart_add(session_id: str, pid: str, qty: int = 1) -> dict:
    if not product(pid):
        raise ValueError(f"Unknown product {pid}")
    if qty < 1 or qty > 50:
        raise ValueError("Quantity must be between 1 and 50 (use a B2B quotation for larger volumes)")
    c = CARTS.setdefault(session_id, {})
    c[pid] = c.get(pid, 0) + qty
    return cart_view(session_id)


def cart_remove(session_id: str, pid: str) -> dict:
    CARTS.get(session_id, {}).pop(pid, None)
    return cart_view(session_id)


def cart_view(session_id: str) -> dict:
    c = CARTS.get(session_id, {})
    lines = []
    for pid, qty in c.items():
        p = product(pid)
        lines.append({"product_id": pid, "title": p["title"], "qty": qty, "unit_price": p["price"],
                      "line_total": round(p["price"] * qty, 2)})
    subtotal = round(sum(l["line_total"] for l in lines), 2)
    shipping = 0.0 if subtotal >= 200 or not lines else 25.0
    total = round(subtotal + shipping, 2)
    return {"session_id": session_id, "lines": lines, "subtotal": subtotal, "shipping": shipping, "total": total,
            "vat_included": round(total * 15 / 115, 2)}


def next_order_id() -> str:
    with _lock:
        r = conn().execute("SELECT MAX(CAST(substr(order_id,2) AS INTEGER)) FROM orders "
                           "WHERE order_id GLOB 'O[0-9][0-9][0-9][0-9][0-9][0-9][0-9]'").fetchone()[0]
        return f"O{(r or 0) + 1:07d}"


def create_order(session_id: str, customer_id: str, payment_method: str, ship_city: str, payment_ref: str,
                 channel: str = "assistant") -> dict:
    cart = cart_view(session_id)
    if not cart["lines"]:
        raise ValueError("Cart is empty")
    oid = next_order_id()
    now = datetime.now().isoformat(timespec="seconds")
    execute("""INSERT INTO orders (order_id, customer_id, created_at, status, channel, payment_method, subtotal,
               discount, vat, shipping, total, ship_city, is_anomaly) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,0)""",
            (oid, customer_id, now, "processing", channel, payment_method, cart["subtotal"], 0.0,
             cart["vat_included"], cart["shipping"], cart["total"], ship_city))
    for l in cart["lines"]:
        execute("INSERT INTO order_items VALUES (?,?,?,?,?)", (oid, l["product_id"], l["qty"], l["unit_price"],
                                                                l["line_total"]))
    execute("INSERT INTO receipts VALUES (?,?,?,?,?,?)", ("R" + oid[1:], oid, now, cart["total"], payment_method,
                                                          payment_ref))
    CARTS.pop(session_id, None)
    return {"order_id": oid, "total": cart["total"], "status": "processing", "created_at": now}


def audit(session_id: str, actor: str, action: str, payload: dict) -> None:
    from .guardrails import redact

    execute("INSERT INTO agent_audit VALUES (?,?,?,?,?)",
            (datetime.now().isoformat(timespec="seconds"), session_id, actor, action, redact(json.dumps(payload))))


def log_search(session_id: str, query: str, intent: str, conf: float, used_llm: bool) -> None:
    from .guardrails import redact

    execute("INSERT INTO search_log VALUES (?,?,?,?,?,?)", (datetime.now().isoformat(timespec="seconds"), session_id,
                                                         redact(query), intent, conf, int(used_llm)))


def new_session() -> str:
    return uuid.uuid4().hex[:12]
