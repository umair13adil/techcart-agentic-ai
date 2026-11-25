"""MCP server: PayGate (mock payment gateway in the style of Stripe / HyperPay / Moyasar).

Demonstrates the integration patterns that matter more than the vendor:
  * idempotency keys (retries by an LLM agent must never double-charge)
  * two-step intent -> confirm (3-D Secure / OTP step happens with the customer, not the agent)
  * the agent never sees card numbers: it passes a payment *method token* from the client
  * deterministic test behaviour: amounts ending in .13 are declined, 'tok_insufficient' -> insufficient funds

    python -m techcart.mcp_servers.payments --http --port 8102
"""
from __future__ import annotations

import time
import uuid

from mcp.server.mcpserver import MCPServer

server = MCPServer("paygate", instructions="Payment intents, confirmation and refunds. Always pass an idempotency_key.")
INTENTS: dict[str, dict] = {}
BY_KEY: dict[str, str] = {}
SUPPORTED = {"mada", "credit_card", "apple_pay", "tabby", "cod"}


@server.tool(description="Create a payment intent. Same idempotency_key => same intent (safe to retry).")
def create_payment_intent(amount: float, currency: str, method: str, idempotency_key: str,
                          customer_ref: str, method_token: str = "tok_test_ok") -> dict:
    if idempotency_key in BY_KEY:
        return {**INTENTS[BY_KEY[idempotency_key]], "idempotent_replay": True}
    if method not in SUPPORTED:
        return {"error": f"unsupported method {method}"}
    if amount <= 0 or amount > 50_000:
        return {"error": "amount out of range"}
    pid = "pi_" + uuid.uuid4().hex[:16]
    needs_action = method in ("mada", "credit_card")  # 3DS OTP with the issuer
    intent = {"intent_id": pid, "amount": round(amount, 2), "currency": currency, "method": method,
              "customer_ref": customer_ref, "token": method_token,
              "status": "pending_cod" if method == "cod" else "requires_confirmation",
              "next_action": "3ds_otp" if needs_action else None, "created": int(time.time())}
    INTENTS[pid], BY_KEY[idempotency_key] = intent, pid
    return intent


@server.tool(description="Confirm a payment intent after the customer completed 3-D Secure / wallet approval.")
def confirm_payment(intent_id: str, otp: str | None = None) -> dict:
    it = INTENTS.get(intent_id)
    if not it:
        return {"error": "unknown intent"}
    if it["status"] in ("succeeded", "pending_cod"):
        return it
    if it["next_action"] == "3ds_otp" and not otp:
        return {**it, "error": "otp_required"}
    if f"{it['amount']:.2f}".endswith(".13") or it["token"] == "tok_declined":
        it.update(status="failed", failure_reason="card_declined")
    elif it["token"] == "tok_insufficient":
        it.update(status="failed", failure_reason="insufficient_funds")
    else:
        it.update(status="succeeded", txn_ref="TXN" + uuid.uuid4().hex[:10].upper())
    return it


@server.tool(description="Refund all or part of a succeeded payment.")
def refund(intent_id: str, amount: float | None = None, reason: str = "requested_by_customer") -> dict:
    it = INTENTS.get(intent_id)
    if not it or it["status"] != "succeeded":
        return {"error": "refundable payment not found"}
    amt = round(amount or it["amount"], 2)
    if amt > it["amount"] - it.get("refunded", 0):
        return {"error": "refund exceeds captured amount"}
    it["refunded"] = round(it.get("refunded", 0) + amt, 2)
    return {"refund_id": "re_" + uuid.uuid4().hex[:12], "intent_id": intent_id, "amount": amt, "reason": reason,
            "status": "pending", "eta_business_days": 5 if it["method"] in ("mada", "credit_card") else 7}


@server.tool(description="Get a payment intent.")
def get_payment(intent_id: str) -> dict:
    return INTENTS.get(intent_id, {"error": "unknown intent"})


def main() -> None:
    from ._run import serve

    serve(server, 8102)


if __name__ == "__main__":
    main()
