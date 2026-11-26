import json
from datetime import datetime

import pandas as pd

from techcart import services
from techcart.config import settings
from techcart.ml import fraud
from techcart.ml.intent import extract_entities
from techcart.ml.tracking import Registry


def summary():
    return json.loads((settings.artifacts_dir / "training_summary.json").read_text())


def test_no_generated_activity(dataset):
    """Store setup seeds products, stock and suppliers only - customers, orders and logs start empty."""
    c = dataset["seed_counts"]
    assert c["products"] > 100 and c["suppliers"] > 0
    assert all(c[t] == 0 for t in ("customers", "orders", "receipts", "quotations", "purchase_orders",
                                   "search_log", "chat_log")), c


def test_cold_start_modes():
    s = summary()
    assert s["intent"]["test_macro_f1"] >= 0.75
    assert s["recommender"]["mode"] == "cold_start"
    assert s["fraud"]["mode"] == "rules" and s["forecast"]["mode"] == "waiting_for_data"


def test_entity_extraction():
    e = extract_entities("gaming laptop under 6,000 SAR with 16GB RAM and RTX 4060")
    assert (e.category, e.use_case, e.max_price, e.ram_gb, e.gpu) == ("Laptops", "gaming", 6000, 16, "RTX 4060")
    e = extract_entities("4k monitor for my macbook")
    assert e.category == "Monitors" and e.brand is None          # head-noun rule
    e = extract_entities("laptop bag for 15 inch")
    assert e.category == "Bags & Sleeves" and e.screen_in == 15
    e = extract_entities("between 3k and 4.5k sar")
    assert (e.min_price, e.max_price) == (3000, 4500)
    assert extract_entities("where is O0012345").order_id == "O0012345"
    assert extract_entities("quote for 25 dell latitude laptops").quantity == 25


def test_cold_start_recommendations_follow_complements():
    rec = Registry().load("recommender")
    laptop = next(p for p in rec.pid if rec.cat[p] == "Laptops")
    out = rec.recommend([(laptop, datetime.now())], k=8, now=datetime.now())
    cats = [rec.cat[r["product_id"]] for r in out]
    assert "Laptops" not in cats                       # just bought one -> don't push another
    assert not {"Desktops", "Graphics Cards"} & set(cats)  # title similarity must not beat the complement prior
    assert all(c in ("Mice", "Bags & Sleeves", "Docks & Hubs", "Monitors", "Headsets", "Storage",
                     "Chargers & Power", "Cooling Pads") for c in cats), cats
    assert max(cats.count(c) for c in set(cats)) <= 2  # diversity cap
    reasons = {r.get("reason") for r in rec.recommend([(laptop, datetime.now())], k=4, explain=True)}
    assert reasons == {"Goes with your Laptops"}           # no "customers who bought..." before there are customers
    anon = rec.recommend([], k=5)                      # guests: ranked by real Amazon review volume
    counts = [services.product(r["product_id"])["rating_count"] for r in anon]
    assert counts[0] >= sorted(p["rating_count"] for p in services.all_products())[len(services.all_products()) // 2]


def test_rule_based_fraud_scoring_explains_itself():
    row = pd.DataFrame([{"account_age_days": 0, "night": 1, "city_mismatch": 1, "max_qty": 4, "high_value_share": 1.0,
                         "log_total": 11.0}])
    risk, reasons, scorer = fraud.score_order(row, None)
    assert risk >= fraud.RULE_ALERT and scorer == "rules" and "new account (<2 days)" in reasons
    calm = pd.DataFrame([{"account_age_days": 200, "night": 0, "city_mismatch": 0, "max_qty": 1,
                          "high_value_share": 0.2, "log_total": 6.0}])
    assert fraud.score_order(calm, None)[0] == 0


def test_drift_report_handles_empty_history():
    from techcart.db import connect
    from techcart.ml.drift import drift_report, psi_categorical

    assert psi_categorical(pd.Series(list("aabb")), pd.Series(list("aabb"))) < 1e-6
    assert drift_report(connect())["overall"] in {"stable", "moderate", "significant"}
