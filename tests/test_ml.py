import json
from datetime import datetime, timedelta

from techcart.config import settings
from techcart.data.catalog import DURABLE
from techcart.ml.intent import extract_entities
from techcart.ml.tracking import Registry


def test_models_registered_and_promoted():
    info = Registry().info()
    for name in ("intent", "recommender", "fraud", "forecast"):
        assert name in info


def test_intent_quality_gate():
    m = json.loads((settings.artifacts_dir / "training_summary.json").read_text())["intent"]
    assert m["test_macro_f1"] >= 0.75
    assert m["accuracy_when_confident"] >= 0.85


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


def test_recommender_rules():
    rec = Registry().load("recommender")
    laptop = next(p for p in rec.pid if rec.cat[p] == "Laptops")
    out = rec.recommend([(laptop, rec.ref_time - timedelta(days=3))], k=10, now=rec.ref_time)
    cats = [rec.cat[r["product_id"]] for r in out]
    assert laptop not in {r["product_id"] for r in out}
    assert "Laptops" not in cats                      # just bought one -> don't push another
    assert max(cats.count(c) for c in set(cats)) <= 2  # diversity cap
    assert all(c not in DURABLE or c != "Laptops" for c in cats)


def test_recommender_vs_popularity():
    """Same rule as the promotion gate: not significantly worse than popularity (paired bootstrap CI),
    while exposing far more of the catalog than a best-seller list."""
    m = json.loads((settings.artifacts_dir / "training_summary.json").read_text())["recommender"]
    assert m["ndcg_diff_ci_high"] >= 0
    assert m["hybrid_catalog_coverage"] > 5 * m["popular_catalog_coverage"]


def test_fraud_model_ranks_anomalies():
    m = json.loads((settings.artifacts_dir / "training_summary.json").read_text())["fraud"]
    assert m["roc_auc"] > 0.85
    assert m["pr_auc"] > 5 * m["base_rate"]  # far better than random


def test_drift_report_runs():
    from techcart.db import connect
    from techcart.ml.drift import drift_report, psi_categorical
    import pandas as pd

    assert psi_categorical(pd.Series(list("aabb")), pd.Series(list("aabb"))) < 1e-6
    rep = drift_report(connect())
    assert rep["overall"] in {"stable", "moderate", "significant"}
