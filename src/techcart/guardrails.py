"""Guardrails: PII redaction (PDPL data minimisation), prompt-injection heuristics, and hard business limits
the agent cannot talk its way around (they are checked in code, not in the prompt)."""
from __future__ import annotations

import re

PII_PATTERNS = [
    (re.compile(r"\b(?:\d[ -]?){13,19}\b"), "[CARD]"),
    (re.compile(r"\b(?:\+?966|0)5\d{8}\b"), "[PHONE]"),
    (re.compile(r"\+9665\d{8}"), "[PHONE]"),
    (re.compile(r"[\w.+-]+@[\w-]+\.[\w.]+"), "[EMAIL]"),
    (re.compile(r"\b[12]\d{9}\b"), "[NATIONAL_ID]"),           # Saudi national ID / Iqama
    (re.compile(r"\bSA\d{2}[0-9A-Z]{20}\b"), "[IBAN]"),
]
INJECTION = re.compile(r"ignore (all|any|previous|the above) (instructions|rules)|system prompt|you are now|"
                       r"developer mode|reveal your (prompt|instructions)|set (the )?price to|apply .*100% discount",
                       re.I)

MAX_ORDER_TOTAL = 25_000.0      # above this the assistant must route to a human / B2B quote
MAX_QTY_PER_LINE = 10
ALLOWED_PAYMENTS = {"mada", "credit_card", "apple_pay", "tabby", "cod"}


def redact(text: str) -> str:
    for pat, repl in PII_PATTERNS:
        text = pat.sub(repl, text)
    return text


def looks_like_injection(text: str) -> bool:
    return bool(INJECTION.search(text or ""))


def validate_checkout(cart: dict, payment_method: str) -> list[str]:
    problems = []
    if not cart["lines"]:
        problems.append("cart is empty")
    if cart["total"] > MAX_ORDER_TOTAL:
        problems.append(f"order total {cart['total']:,.0f} SAR exceeds the assistant limit of "
                        f"{MAX_ORDER_TOTAL:,.0f} SAR - please request a B2B quotation")
    if any(l["qty"] > MAX_QTY_PER_LINE for l in cart["lines"]):
        problems.append(f"max {MAX_QTY_PER_LINE} units per item via the assistant")
    if payment_method not in ALLOWED_PAYMENTS:
        problems.append(f"unsupported payment method {payment_method}")
    if payment_method == "cod" and cart["total"] > 2500:
        problems.append("cash on delivery is limited to 2,500 SAR")
    if payment_method == "tabby" and not (100 <= cart["total"] <= 5000):
        problems.append("Tabby is available for orders between 100 and 5,000 SAR")
    return problems
