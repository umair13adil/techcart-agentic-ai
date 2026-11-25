"""Live order simulator: writes a realistic new order every few seconds and publishes it to subscribers
(the admin dashboard's SSE stream). Each order is risk-scored by the production fraud model on arrival -
the same online-scoring path a real checkout would use."""
from __future__ import annotations

import asyncio
import json
import random
from datetime import datetime

import pandas as pd

from .. import services
from ..data.catalog import COMPLEMENTS
from ..ml import fraud
from ..ml.tracking import Registry


class Broadcaster:
    def __init__(self):
        self.queues: set[asyncio.Queue] = set()

    def subscribe(self) -> asyncio.Queue:
        q: asyncio.Queue = asyncio.Queue(maxsize=100)
        self.queues.add(q)
        return q

    def unsubscribe(self, q: asyncio.Queue) -> None:
        self.queues.discard(q)

    def publish(self, event: dict) -> None:
        for q in list(self.queues):
            if not q.full():
                q.put_nowait(event)


BUS = Broadcaster()


def _score(order: dict, items: list[dict]) -> tuple[float, list[str]]:
    try:
        model = Registry().load("fraud")
    except FileNotFoundError:
        return 0.0, []
    cust = pd.DataFrame(services.q("SELECT customer_id, created_at, city FROM customers WHERE customer_id=?",
                                   (order["customer_id"],)))
    prods = pd.DataFrame(services.q("SELECT product_id, category FROM products"))
    feats = fraud.build_features(pd.DataFrame([order]), pd.DataFrame([{**i, "order_id": order["order_id"]} for i in items]),
                                 cust, prods)
    feats["prior_orders"] = services.q("SELECT COUNT(*) n FROM orders WHERE customer_id=?", (order["customer_id"],))[0]["n"]
    s = float(fraud.score(model, feats)[0])
    return round(s, 4), fraud.reasons(feats.iloc[0])


def make_order(rng: random.Random) -> dict:
    custs = services.q("SELECT customer_id, city, segment, is_business, created_at FROM customers "
                       "ORDER BY RANDOM() LIMIT 1")
    c = custs[0]
    products = services.all_products()
    fraudy = rng.random() < 0.015
    cat = rng.choice(["Laptops", "Laptops", "Monitors", "Mice", "Headsets", "Storage", "Keyboards", "Docks & Hubs",
                      "Graphics Cards", "Chargers & Power"])
    pool = [p for p in products if p["category"] == cat]
    if fraudy:
        pool = sorted([p for p in products if p["category"] in ("Laptops", "Graphics Cards")], key=lambda p: -p["price"])[:6]
    primary = rng.choice(pool)
    qty = rng.randint(3, 6) if fraudy else (rng.choice([5, 10]) if c["is_business"] and cat == "Laptops" else 1)
    items = [{"product_id": primary["product_id"], "qty": qty, "unit_price": primary["price"],
              "line_total": round(primary["price"] * qty, 2)}]
    if not fraudy and rng.random() < 0.45 and COMPLEMENTS.get(cat):
        comp = rng.choice(COMPLEMENTS[cat])
        cands = [p for p in products if p["category"] == comp]
        if cands:
            p = rng.choice(cands)
            items.append({"product_id": p["product_id"], "qty": 1, "unit_price": p["price"], "line_total": p["price"]})
    subtotal = round(sum(i["line_total"] for i in items), 2)
    ship = 0.0 if subtotal >= 200 else 25.0
    now = datetime.now()
    if fraudy:
        now = now.replace(hour=3)
    oid = services.next_order_id()
    order = {"order_id": oid, "customer_id": c["customer_id"], "created_at": now.isoformat(timespec="seconds"),
             "status": "processing", "channel": rng.choice(["web", "app", "app", "assistant"]),
             "payment_method": "credit_card" if fraudy else rng.choice(["mada", "mada", "apple_pay", "credit_card", "tabby"]),
             "subtotal": subtotal, "discount": 0.0, "vat": round((subtotal + ship) * 15 / 115, 2), "shipping": ship,
             "total": round(subtotal + ship, 2),
             "ship_city": rng.choice(["Riyadh", "Jeddah", "Dammam", "Abha"]) if fraudy else c["city"],
             "is_anomaly": int(fraudy)}
    risk, why = _score(order, items)
    order["risk_score"] = risk
    cols = list(order)
    services.execute(f"INSERT INTO orders ({','.join(cols)}) VALUES ({','.join('?' * len(cols))})",
                     tuple(order[k] for k in cols))
    for i in items:
        services.execute("INSERT INTO order_items VALUES (?,?,?,?,?)", (oid, i["product_id"], i["qty"], i["unit_price"],
                                                                         i["line_total"]))
    return {"type": "order", "order_id": oid, "ts": order["created_at"], "total": order["total"],
            "city": order["ship_city"], "channel": order["channel"], "risk": risk, "risk_reasons": why,
            "items": [{"title": services.product(i["product_id"])["title"][:60], "qty": i["qty"],
                       "category": services.product(i["product_id"])["category"]} for i in items]}


async def run_simulator(interval: tuple[float, float] = (5.0, 12.0), seed: int | None = None) -> None:
    rng = random.Random(seed)
    while True:
        await asyncio.sleep(rng.uniform(*interval))
        try:
            ev = await asyncio.to_thread(make_order, rng)
            BUS.publish(ev)
        except Exception as exc:  # never kill the loop
            BUS.publish({"type": "error", "message": str(exc)})


def sse(event: dict) -> str:
    return f"data: {json.dumps(event, default=str)}\n\n"
