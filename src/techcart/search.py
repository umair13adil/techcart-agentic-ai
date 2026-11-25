"""Smart product search: understand -> filter -> hybrid retrieve -> rank -> explain.

Hard constraints (category, brand, budget, min specs) become metadata/SQL filters. Use-case is a soft boost.
If nothing matches, constraints are relaxed in a fixed order and the response says what was relaxed - a better
experience than an empty page, and transparent about it.
"""
from __future__ import annotations

import json
from functools import lru_cache

from . import services
from .ml.intent import QueryUnderstander, Understanding
from .ml.tracking import Registry
from .rag.retriever import get_retriever

USE_CASE_SUBCATS = {
    "gaming": {"gaming", "gaming desktop", "mechanical", "nvidia", "amd"},
    "creative": {"creator", "external ssd", "streaming"},
    "office": {"business", "office", "dock", "business desktop"},
    "student": {"student", "ultrabook", "2-in-1", "backpack", "sleeve"},
    "travel": {"ultrabook", "2-in-1", "sleeve", "power bank", "hub", "charger"},
    "programming": {"ultrabook", "business", "creator"},
    "ai_ml": {"gaming", "creator", "nvidia"},
}


@lru_cache(maxsize=1)
def understander() -> QueryUnderstander:
    return QueryUnderstander(Registry().load("intent"))


def _passes(p: dict, e: dict, relax: set) -> bool:
    specs = json.loads(p["specs_json"] or "{}")
    if e.get("category") and "category" not in relax and p["category"] != e["category"]:
        return False
    if e.get("brand") and "brand" not in relax and p["brand"].lower() != e["brand"]:
        return False
    widen = 1.2 if "price" in relax else 1.0
    if e.get("max_price") and p["price"] > e["max_price"] * widen:
        return False
    if e.get("min_price") and p["price"] < e["min_price"] / widen:
        return False
    if "specs" not in relax:
        if e.get("ram_gb") and int(specs.get("ram_gb", 0) or 0) < e["ram_gb"]:
            return False
        if e.get("storage_gb") and int(specs.get("storage_gb", specs.get("capacity_gb", 0)) or 0) < e["storage_gb"] * 0.95:
            return False
        if e.get("gpu") and e["gpu"].replace(" ", "").lower() not in str(specs.get("gpu", p["title"])).replace(" ", "").lower():
            return False
    return True


def search_products(query: str, k: int = 12, understanding: Understanding | None = None) -> dict:
    u = understanding or understander().understand(query)
    e = u.entities
    products = services.all_products()
    stock = services.stock_map()
    relaxed: list[str] = []
    allowed: list[dict] = []
    for step in [set(), {"specs"}, {"specs", "brand"}, {"specs", "brand", "price"}, {"specs", "brand", "price", "category"}]:
        allowed = [p for p in products if _passes(p, e, step)]
        if allowed:
            relaxed = sorted(step)
            break
    ids = {p["product_id"] for p in allowed}
    hits = get_retriever().retrieve(u.rewritten_query, role="customer", k=min(60, len(ids)) or 1,
                                    doc_type="product", doc_ids=ids)
    rank = {h.doc_id: (i, h.score) for i, h in enumerate(hits)}
    subs = USE_CASE_SUBCATS.get(e.get("use_case") or "", set())

    def score(p: dict) -> float:
        r = rank.get(p["product_id"])
        s = 1.0 / (10 + r[0]) if r else 0.0
        s += 0.03 if p["subcategory"] in subs else 0.0
        s += 0.004 * (p["rating"] or 0)
        title = p["title"].lower()
        s += 0.012 * sum(1 for kw in e.get("keywords", []) if len(kw) > 1 and kw in title)  # exact-token boost
        s -= 0.05 if stock.get(p["product_id"], 0) <= 0 else 0.0
        return s

    ranked = sorted(allowed, key=score, reverse=True)[:k]
    explain = [f"{key.replace('_', ' ')}: {val}" for key, val in e.items()
               if val not in (None, [], "") and key not in ("keywords",)]
    return {"understanding": u.dict(), "relaxed": relaxed, "matched_on": explain,
            "results": [services.product_card(p, stock) for p in ranked]}
