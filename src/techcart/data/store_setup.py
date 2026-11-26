"""Store setup: everything the shop needs before its first real customer arrives.

Only three things are seeded, and none of them is activity:
  * products          - from Hugging Face (Amazon-Reviews-2023), see catalog.py
  * starting stock    - derived from each product's real demand signal (Amazon review count) and price
  * supplier list     - the distributors the store buys from (master data)

Customers, orders, receipts, purchase orders, quotations, searches and chats are NOT generated. They are created
by real use of the app (Entra ID sign-in, search, the assistant, checkout) and the AI analysis of them is
persisted alongside (intent/entities per query, risk score + reasons per order, a log per assistant turn).
"""
from __future__ import annotations

import json
import math
import random
from pathlib import Path

from .policies import POLICIES

SUPPLIERS = [
    ("SUP01", "Gulf IT Distribution LLC", "UAE", ["Dell", "HP", "Lenovo", "Microsoft"], "Net 45", 7),
    ("SUP02", "Arabian Tech Distributors", "KSA", ["ASUS", "Acer", "MSI", "Gigabyte", "Sapphire"], "Net 30", 5),
    ("SUP03", "Peninsula Apple Partners", "KSA", ["Apple"], "Prepaid", 4),
    ("SUP04", "Desert Peripherals Trading", "KSA", ["Logitech", "Razer", "SteelSeries", "HyperX", "Keychron",
                                                    "Jabra", "Sony", "Elgato"], "Net 30", 6),
    ("SUP05", "Al-Wadi Storage & Memory", "UAE", ["Samsung", "WD", "Western Digital", "Crucial", "SanDisk",
                                                  "Seagate", "Kingston", "Corsair", "G.Skill"], "Net 30", 8),
    ("SUP06", "Kingdom Accessories Co. (general distributor)", "KSA",
     ["Anker", "UGREEN", "Targus", "Samsonite", "TP-Link", "APC", "Cooler Master", "CalDigit", "BenQ", "LG",
      "Intel"], "Net 15", 5),
]
GENERAL_SUPPLIER = "SUP06"


def supplier_rows() -> list[dict]:
    return [{"supplier_id": s[0], "name": s[1], "country": s[2], "brands": ",".join(s[3]), "payment_terms": s[4],
             "lead_time_days": s[5]} for s in SUPPLIERS]


def supplier_for(brand: str) -> tuple[str, int]:
    b = (brand or "").lower()
    for sid, _, _, brands, _, lead in SUPPLIERS:
        if any(x.lower() == b or b.startswith(x.lower() + " ") for x in brands):
            return sid, lead
    return GENERAL_SUPPLIER, 5


def build_inventory(products: list[dict], seed: int = 7) -> list[dict]:
    """Opening stock per SKU. Demand proxy = log(Amazon review count): popular items get a deeper target level,
    expensive items a shallower one. The reorder point follows demand (not current stock), and each SKU starts
    somewhere in its replenishment cycle - so, as in any live store, some lines open low or out of stock."""
    rng = random.Random(seed)
    out = []
    for p in products:
        demand = math.log1p(p.get("rating_count") or 0)
        price_factor = 1.4 if p["price"] < 500 else 1.0 if p["price"] < 3000 else 0.6
        target = max(3, int(round((2 + 1.6 * demand) * price_factor)))
        reorder_point = max(2, int(round(target * 0.3)))
        fill = rng.random()
        stock = 0 if fill < 0.04 else int(round(target * (0.15 + 1.05 * fill)))
        sid, lead = supplier_for(p["brand"])
        out.append({"product_id": p["product_id"], "stock_on_hand": stock, "reserved": 0,
                    "reorder_point": reorder_point, "lead_time_days": lead,
                    "warehouse": "RUH-1" if rng.random() < 0.7 else "JED-1"})
    return out


def write_documents(out_dir: Path, products: list[dict]) -> list[dict]:
    """RAG corpus = product sheets (from the HF catalog) + store policies. Business documents (receipts,
    quotations, purchase orders) join the corpus only once they exist in the database (see rag/retriever.py)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest: list[dict] = []

    def emit(doc_id, doc_type, audience, title, body, **meta):
        path = out_dir / doc_type / f"{doc_id}.md"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"# {title}\n\n{body.strip()}\n")
        manifest.append({"doc_id": doc_id, "doc_type": doc_type, "audience": audience, "title": title,
                         "path": str(path.relative_to(out_dir)), **meta})

    for p in products:
        specs = json.loads(p["specs_json"])
        spec_md = "\n".join(f"- **{k.replace('_', ' ')}**: {v}" for k, v in specs.items())
        feat_md = "\n".join(f"- {f}" for f in p["features"].split("\n") if f)
        emit(p["product_id"], "product", "public", p["title"], f"""
**Brand:** {p['brand']}  |  **Category:** {p['category']} / {p['subcategory']}  |  **SKU:** {p['sku']}
**Price:** {p['price']:,.2f} SAR (VAT inclusive)  |  **Rating:** {p['rating']} / 5 from {p['rating_count']} Amazon reviews

## Overview
{p['description']}

## Key features
{feat_md}

## Specifications
{spec_md}
""", product_id=p["product_id"], category=p["category"], brand=p["brand"], price=p["price"])

    for pid, (title, body) in POLICIES.items():
        emit(pid, "policy", "public", title, body)

    with (out_dir / "manifest.jsonl").open("w") as f:
        for m in manifest:
            f.write(json.dumps(m) + "\n")
    return manifest


def build_rag_eval(products: list[dict], n_products: int = 30, seed: int = 3) -> list[dict]:
    """Questions with known gold documents, written against the real catalog and the store policies."""
    rng = random.Random(seed)
    qs = [
        {"q": "How many days do I have to return a laptop?", "gold": ["policy-returns"], "answer": "7 days of delivery", "audience": "public"},
        {"q": "Is shipping free?", "gold": ["policy-shipping"], "answer": "Free on orders above 200 SAR, otherwise 25 SAR", "audience": "public"},
        {"q": "Can I pay in installments?", "gold": ["policy-payments"], "answer": "Tabby, split into 4 interest-free payments for 100-5,000 SAR", "audience": "public"},
        {"q": "What is the maximum order value for cash on delivery?", "gold": ["policy-payments"], "answer": "2,500 SAR", "audience": "public"},
        {"q": "How much does a 2 year TechCare+ plan cost for a 4500 SAR laptop?", "gold": ["policy-warranty"], "answer": "499 SAR", "audience": "public"},
        {"q": "What discount do companies get for 30 laptops?", "gold": ["policy-b2b"], "answer": "8%", "audience": "public"},
        {"q": "How long is a business quotation valid?", "gold": ["policy-b2b"], "answer": "14 days", "audience": "public"},
        {"q": "Can I upgrade the RAM in a MacBook Air?", "gold": ["faq-compatibility"], "answer": "No, memory is soldered", "audience": "public"},
        {"q": "Which GPU do I need for 1440p gaming?", "gold": ["faq-compatibility"], "answer": "RTX 4070", "audience": "public"},
        {"q": "Do you deliver same day in Riyadh?", "gold": ["policy-shipping"], "answer": "Yes, express same-day 49 SAR before 12 PM", "audience": "public"},
        {"q": "How long do you keep my chat history?", "gold": ["policy-privacy"], "answer": "12 months", "audience": "public"},
        {"q": "What happens to my data on a traded-in laptop?", "gold": ["policy-tradein"], "answer": "Wiped to NIST 800-88 with certificate", "audience": "public"},
        {"q": "If I find a lower price elsewhere will you match it?", "gold": ["policy-pricematch"], "answer": "Yes within 7 days at authorised Saudi retailers", "audience": "public"},
    ]
    for p in rng.sample(products, min(n_products, len(products))):
        specs = json.loads(p["specs_json"])
        t = p["title"].split(" - ")[0].split(",")[0][:80]
        qs.append({"q": f"How much is the {t}?", "gold": [p["product_id"]], "answer": f"{p['price']:,.0f} SAR",
                   "audience": "public"})
        if "ram_gb" in specs and rng.random() < 0.5:
            qs.append({"q": f"How much RAM does the {t} have?", "gold": [p["product_id"]],
                       "answer": f"{specs['ram_gb']}GB", "audience": "public"})
    return qs
