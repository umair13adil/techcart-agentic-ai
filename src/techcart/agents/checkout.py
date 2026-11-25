"""Deterministic, money-safe checkout workflow orchestrated over MCP.

Design rule: the LLM decides *whether* the customer wants to buy; code decides *how* money and stock move.
Every side-effecting step is idempotent and compensating actions run on failure (saga pattern):

  prepare:  validate guardrails -> reserve stock (ERP) -> shipping quote (Logistics) -> payment intent (PayGate)
            => returns a summary that MUST be confirmed by the human (LangGraph interrupt / UI button)
  confirm:  confirm payment (3DS OTP) -> commit reservation -> write order + receipt -> create shipment
  failure:  release reservation (compensation); payment intent left unconfirmed = no charge
"""
from __future__ import annotations

import hashlib
import json

from .. import services
from ..guardrails import validate_checkout
from .mcp_bridge import MCPHub

PENDING: dict[str, dict] = {}  # session_id -> prepared checkout awaiting confirmation


def _idem(session_id: str, cart: dict, method: str) -> str:
    raw = json.dumps([session_id, method, [(l["product_id"], l["qty"]) for l in cart["lines"]]], sort_keys=True)
    return "chk_" + hashlib.sha256(raw.encode()).hexdigest()[:20]


async def prepare(hub: MCPHub, session_id: str, customer_id: str, payment_method: str, city: str,
                  express: bool = False) -> dict:
    cart = services.cart_view(session_id)
    problems = validate_checkout(cart, payment_method)
    if problems:
        return {"status": "blocked", "problems": problems, "cart": cart}
    key = _idem(session_id, cart, payment_method)
    res = await hub.call("reserve_stock", order_ref=key,
                         lines=[{"product_id": l["product_id"], "qty": l["qty"]} for l in cart["lines"]])
    if res.get("status") != "reserved":
        return {"status": "out_of_stock", "detail": res.get("short"), "cart": cart}
    options = await hub.call("quote_shipping", city=city, order_value=cart["subtotal"])
    ship = next((o for o in options if o["service"] == ("same_day" if express else "standard")), options[0])
    total = round(cart["subtotal"] + ship["fee"], 2)
    intent = await hub.call("create_payment_intent", amount=total, currency="SAR", method=payment_method,
                            idempotency_key=key, customer_ref=customer_id)
    if "error" in intent:
        await hub.call("release_reservation", order_ref=key)
        return {"status": "payment_error", "detail": intent["error"]}
    summary = {"status": "awaiting_confirmation", "order_ref": key, "intent_id": intent["intent_id"],
               "needs_otp": intent.get("next_action") == "3ds_otp", "payment_method": payment_method,
               "city": city, "shipping": ship, "lines": cart["lines"], "subtotal": cart["subtotal"],
               "total": total, "vat_included": round(total * 15 / 115, 2), "customer_id": customer_id}
    PENDING[session_id] = summary
    services.audit(session_id, "agent", "checkout_prepared", {"order_ref": key, "total": total})
    return summary


async def confirm(hub: MCPHub, session_id: str, approved: bool, otp: str | None = None) -> dict:
    pending = PENDING.pop(session_id, None)
    if not pending:
        return {"status": "nothing_pending"}
    if not approved:
        await hub.call("release_reservation", order_ref=pending["order_ref"])
        services.audit(session_id, "customer", "checkout_cancelled", {"order_ref": pending["order_ref"]})
        return {"status": "cancelled"}
    pay = await hub.call("confirm_payment", intent_id=pending["intent_id"], otp=otp)
    if pay.get("error") == "otp_required":
        PENDING[session_id] = pending  # keep it; ask again
        return {"status": "otp_required"}
    if pay.get("status") not in ("succeeded", "pending_cod"):
        await hub.call("release_reservation", order_ref=pending["order_ref"])  # compensation
        services.audit(session_id, "system", "payment_failed", {"reason": pay.get("failure_reason")})
        return {"status": "payment_failed", "reason": pay.get("failure_reason")}
    await hub.call("commit_reservation", order_ref=pending["order_ref"])
    order = services.create_order(session_id, pending["customer_id"], pending["payment_method"], pending["city"],
                                  pay.get("txn_ref", "COD"))
    shipment = await hub.call("create_shipment", order_id=order["order_id"], city=pending["city"],
                              service=pending["shipping"]["service"])
    services.audit(session_id, "system", "order_placed", {"order_id": order["order_id"], "total": order["total"],
                                                          "tracking": shipment["tracking_number"]})
    return {"status": "placed", "order": order, "payment": {"txn_ref": pay.get("txn_ref"), "status": pay["status"]},
            "shipment": shipment}
