"""Native (in-process) tools used by both the LLM agents and the rule-based fallback agent.
Each returns JSON-serialisable data; LangChain wrappers are created in graph.py."""
from __future__ import annotations

from datetime import datetime
from functools import lru_cache

from .. import services
from ..ml.tracking import Registry
from ..rag.retriever import answer
from ..search import search_products


@lru_cache(maxsize=1)
def _rec():
    return Registry().load("recommender")


def search_catalog(query: str, limit: int = 6) -> dict:
    r = search_products(query, k=limit)
    return {"matched_on": r["matched_on"], "relaxed": r["relaxed"], "intent": r["understanding"]["intent"],
            "products": [{k: x[k] for k in ("product_id", "title", "price", "rating", "stock_label", "category")}
                         for x in r["results"]]}


def product_details(product_id: str) -> dict:
    p = services.product(product_id)
    return services.product_card(p) if p else {"error": "unknown product"}


def recommend_for_customer(customer_id: str | None, session_id: str | None = None, limit: int = 6) -> list[dict]:
    history = services.customer_history(customer_id) if customer_id else []
    cart = services.cart_view(session_id)["lines"] if session_id else []
    history += [(l["product_id"], datetime.now()) for l in cart]
    consent = services.customer(customer_id)["marketing_consent"] if customer_id and services.customer(customer_id) else 0
    if customer_id and not consent and not cart:
        history = []  # PDPL: no personalisation without consent -> trending items only
    stock = services.stock_map()
    recs = _rec().recommend(history, k=limit, in_stock={k for k, v in stock.items() if v > 0},
                            now=datetime.now(), explain=True)
    out = []
    for r in recs:
        card = services.product_card(services.product(r["product_id"]), stock)
        card["reason"] = r.get("reason", "Trending this month")
        out.append(card)
    return out


def policy_qa(question: str) -> dict:
    return answer(question, role="customer")


def order_lookup(order_id: str, customer_id: str | None) -> dict:
    return services.order_status(order_id, customer_id) or {"error": "order not found"}


def add_to_cart(session_id: str, product_id: str, qty: int = 1) -> dict:
    return services.cart_add(session_id, product_id, qty)


def view_cart(session_id: str) -> dict:
    return services.cart_view(session_id)


def remove_from_cart(session_id: str, product_id: str) -> dict:
    return services.cart_remove(session_id, product_id)


def request_b2b_quote(customer_id: str, product_id: str, qty: int, note: str = "") -> dict:
    """Creates a customer quotation with the volume-discount policy (10-24: 5%, 25-49: 8%, 50+: 12%)."""
    from datetime import date, timedelta

    p = services.product(product_id)
    if not p:
        return {"error": "unknown product"}
    if qty < 10:
        return {"error": "quotations start at 10 units; smaller orders can be placed directly"}
    disc = 0.12 if qty >= 50 else 0.08 if qty >= 25 else 0.05
    unit = round(p["price"] * (1 - disc), 2)
    n = services.q("SELECT COUNT(*) n FROM quotations")[0]["n"] + 1
    qid = f"QT-C-{n:04d}A"
    total = round(unit * qty, 2)
    services.execute("INSERT INTO quotations VALUES (?,?,?,?,?,?,?,?)",
                     (qid, "customer_quote", customer_id, date.today().isoformat(),
                      (date.today() + timedelta(days=14)).isoformat(), "draft", total,
                      f"Volume discount {int(disc * 100)}%. {note}".strip()))
    services.execute("INSERT INTO quote_lines VALUES (?,?,?,?)", (qid, product_id, qty, unit))
    return {"quote_id": qid, "product": p["title"], "qty": qty, "unit_price": unit, "discount_pct": disc * 100,
            "total": total, "valid_days": 14, "status": "draft - a sales rep will review and send it"}
