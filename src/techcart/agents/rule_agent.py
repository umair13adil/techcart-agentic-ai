"""Deterministic fallback agent (no LLM). Uses the intent classifier + entity extractor to call the same tools and
the same MCP checkout workflow as the LLM agent.

Why it exists: (1) graceful degradation when the LLM endpoint is down / rate-limited / over budget,
(2) a cheap, fully reproducible path for CI and demos, (3) a baseline to justify the LLM's cost.
"""
from __future__ import annotations

import re

from .. import services
from ..guardrails import looks_like_injection
from ..search import understander
from . import checkout, tools
from .mcp_bridge import MCPHub

SESSION: dict[str, dict] = {}
ORDINALS = {"first": 0, "1st": 0, "second": 1, "2nd": 1, "third": 2, "3rd": 2, "fourth": 3, "last": -1}
PAY = [("apple pay", "apple_pay"), ("tabby", "tabby"), ("installment", "tabby"), ("cash", "cod"), ("cod", "cod"),
       ("credit", "credit_card"), ("card", "credit_card"), ("mada", "mada")]
CHECKOUT_WORDS = re.compile(r"check ?out|place (the |my )?order|proceed|pay (with|by|now)|complete (my )?purchase|confirm the order")


def _fmt_price(x: float) -> str:
    return f"{x:,.0f} SAR"


def _resolve_product(msg: str, st: dict) -> str | None:
    t = msg.lower()
    last = st.get("last_products", [])
    if m := re.search(r"#\s?(\d)|number (\d)|option (\d)", t):
        i = int(next(g for g in m.groups() if g)) - 1
        return last[i]["product_id"] if 0 <= i < len(last) else None
    for w, i in ORDINALS.items():
        if re.search(rf"\b{w}\b", t) and last:
            return last[i]["product_id"]
    if re.search(r"\b(it|that|this|that one|this one)\b", t) and last:
        return last[0]["product_id"]
    for p in last:  # name mention
        if any(tok in t for tok in p["title"].lower().split(" - ")[0].split()[1:3] if len(tok) > 3):
            return p["product_id"]
    found = tools.search_catalog(msg, limit=1)["products"]
    return found[0]["product_id"] if found else None


async def handle(hub: MCPHub, session_id: str, customer_id: str | None, message: str) -> dict:
    st = SESSION.setdefault(session_id, {})
    out = {"agent": "rules", "products": [], "pending": None, "trace": []}
    if looks_like_injection(message):
        services.audit(session_id, "guardrail", "prompt_injection_blocked", {"message": message[:200]})
        return {**out, "intent": "blocked", "reply": "I can't help with that, but I'm happy to help you find "
                                                      "products, track an order or answer policy questions."}
    u = understander().understand(message)
    services.log_search(session_id, message, u.intent, u.confidence, u.tier == "llm")
    out["intent"], ent = u.intent, u.entities
    out["trace"].append({"step": "understand", "intent": u.intent, "confidence": u.confidence, "tier": u.tier})
    cust = services.customer(customer_id) if customer_id else None

    if u.intent in ("product_search", "comparison"):
        r = tools.search_catalog(message, limit=6)
        st["last_products"] = r["products"]
        out["products"] = [services.product_card(services.product(p["product_id"])) for p in r["products"]]
        note = f" (I relaxed: {', '.join(r['relaxed'])})" if r["relaxed"] else ""
        if not r["products"]:
            out["reply"] = "I couldn't find a match. Could you tell me your budget and what you'll use it for?"
        elif u.intent == "comparison" and len(r["products"]) >= 2:
            sides = re.split(r"\s+(?:vs\.?|versus|or|compared to)\s+", re.sub(r"^(compare|which is better)\s+", "",
                                                                             message.lower().strip(" ?")), maxsplit=1)
            picks = [tools.search_catalog(sd, limit=1)["products"] for sd in sides] if len(sides) == 2 else []
            pair = [x[0] for x in picks if x] if len(picks) == 2 and all(picks) else r["products"][:2]
            if len(pair) == 2 and pair[0]["product_id"] == pair[1]["product_id"]:
                pair = r["products"][:2]
            st["last_products"] = pair
            out["products"] = [services.product_card(services.product(p["product_id"])) for p in pair]
            a, b = out["products"]
            keys = [k for k in ("cpu", "ram_gb", "storage_gb", "gpu", "screen_in", "refresh_hz", "weight_kg",
                                "battery_h", "resolution", "panel", "dpi", "wireless") if k in a["specs"] or k in b["specs"]]
            rows = "\n".join(f"- {k.replace('_', ' ')}: {a['specs'].get(k, '-')} vs {b['specs'].get(k, '-')}" for k in keys)
            out["reply"] = (f"**{a['title']}** ({_fmt_price(a['price'])}) vs **{b['title']}** "
                            f"({_fmt_price(b['price'])}):\n{rows}")
        else:
            top = r["products"][0]
            out["reply"] = (f"I found {len(r['products'])} options{note}. Top pick: **{top['title']}** at "
                            f"{_fmt_price(top['price'])} ({top['stock_label']}). Say 'add the first one' to buy.")
    elif u.intent == "recommendation":
        recs = tools.recommend_for_customer(customer_id, session_id, limit=6)
        st["last_products"] = recs
        out["products"] = recs
        out["reply"] = "Based on your purchases and cart, these go well together:\n" + "\n".join(
            f"- {r['title']} - {_fmt_price(r['price'])} ({r['reason']})" for r in recs[:4])
    elif u.intent in ("policy_question", "return_refund") and not ent.get("order_id"):
        a = tools.policy_qa(message)
        out["reply"], out["citations"] = a["answer"], a["citations"]
    elif u.intent in ("order_status", "return_refund"):
        oid = ent.get("order_id") or (services.recent_orders(customer_id, 1) or [{}])[0].get("order_id")
        o = tools.order_lookup(oid, customer_id) if oid else {"error": "no orders found"}
        if "error" in o:
            out["reply"] = "I couldn't find that order on your account. Please check the order number."
        elif u.intent == "order_status":
            out["reply"] = (f"Order **{o['order_id']}** ({len(o['items'])} item(s), {_fmt_price(o['total'])}) is "
                            f"**{o['status']}**. Estimated delivery to {o['ship_city']}: {o['estimated_delivery']}.")
        else:
            out["reply"] = (f"I can help return items from order **{o['order_id']}** (status {o['status']}). Returns "
                            f"are accepted within 7 days of delivery if unused; faulty items are replaced (DOA "
                            f"within 14 days). I've logged your request for our team.")
            services.audit(session_id, "agent", "return_requested", {"order_id": o["order_id"]})
    elif u.intent == "purchase":
        want_checkout = bool(CHECKOUT_WORDS.search(message.lower()))
        cart = services.cart_view(session_id)
        pid = None if (want_checkout and cart["lines"] and not re.search(r"\b(buy|add)\b", message.lower())) \
            else _resolve_product(message, st)
        if pid:
            qty = int(m.group(1)) if (m := re.search(r"\b(\d{1,2})\s*(?:x|units|of|pcs)?\b", message)) and \
                int(m.group(1)) <= 10 else 1
            cart = tools.add_to_cart(session_id, pid, qty)
            out["reply"] = f"Added **{services.product(pid)['title']}** to your cart. Cart total {_fmt_price(cart['total'])}."
        if want_checkout or (not pid and cart["lines"]):
            method = next((v for k, v in PAY if k in message.lower()), "mada")
            city = next((c for c in ["Riyadh", "Jeddah", "Dammam", "Khobar", "Mecca", "Medina", "Abha", "Tabuk"]
                         if c.lower() in message.lower()), (cust or {}).get("city", "Riyadh"))
            prep = await checkout.prepare(hub, session_id, customer_id or "GUEST", method, city)
            out["trace"].append({"step": "checkout.prepare", "mcp_calls": hub.calls[-4:]})
            if prep["status"] == "awaiting_confirmation":
                out["pending"] = prep
                out["reply"] = (out.get("reply", "") + f"\nReady to place your order: {_fmt_price(prep['total'])} via "
                                f"{method.replace('_', ' ')}, delivery to {city} ({prep['shipping']['carrier']}, "
                                f"~{prep['shipping']['eta_days']} day(s)). Please confirm.").strip()
            else:
                out["reply"] = f"I couldn't start checkout: {prep.get('problems') or prep.get('detail') or prep['status']}"
        elif not pid:
            out["reply"] = "Which product would you like? You can say 'add the first one'."
    elif u.intent == "quote_request":
        qty = ent.get("quantity") or 10
        found = tools.search_catalog(message, limit=1)["products"]
        if not customer_id or not found:
            out["reply"] = "Please sign in with your business account and tell me the model and quantity."
        else:
            qt = tools.request_b2b_quote(customer_id, found[0]["product_id"], qty)
            out["reply"] = (f"Draft quotation **{qt['quote_id']}**: {qty} x {qt['product']} at "
                            f"{_fmt_price(qt['unit_price'])} each ({qt['discount_pct']:.0f}% volume discount), total "
                            f"{_fmt_price(qt['total'])}, valid 14 days. A sales rep will confirm.") if "error" not in qt \
                else qt["error"]
    else:
        name = f" {cust['name'].split()[0]}" if cust and not cust["is_business"] else ""
        out["reply"] = (f"Hi{name}! I'm the TechCart assistant. I can find laptops & accessories, recommend what goes "
                        "with your setup, answer warranty/shipping questions, track orders and check you out.")
    out["cart"] = services.cart_view(session_id)
    return out


async def confirm(hub: MCPHub, session_id: str, approved: bool, otp: str | None = None) -> dict:
    r = await checkout.confirm(hub, session_id, approved, otp)
    msg = {"placed": lambda: f"Order **{r['order']['order_id']}** placed! Paid {_fmt_price(r['order']['total'])} "
                             f"(ref {r['payment']['txn_ref'] or 'COD'}). Tracking: {r['shipment']['tracking_number']}, "
                             f"ETA {r['shipment']['eta']}.",
           "cancelled": lambda: "No problem - I've cancelled checkout and released the reserved stock.",
           "otp_required": lambda: "Please enter the one-time password sent by your bank.",
           "payment_failed": lambda: f"The payment was declined ({r.get('reason')}). Your items are still in your "
                                     "cart - try another payment method?",
           "nothing_pending": lambda: "There's no checkout waiting for confirmation."}[r["status"]]()
    return {"agent": "rules", "reply": msg, "result": r, "cart": services.cart_view(session_id),
            "pending": checkout.PENDING.get(session_id)}
