"""Train, evaluate, register and (gated) promote every classical model.

    python scripts/train_all.py

This is the script a CI/CD job (GitHub Actions / Azure ML pipeline / Vertex Pipeline) would run on a schedule or
when drift is detected. Promotion gates stop a worse model from reaching production.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd  # noqa: E402

from techcart.config import settings  # noqa: E402
from techcart.db import connect  # noqa: E402
from techcart.ml import forecast, fraud, recommender  # noqa: E402
from techcart.ml.intent import train_and_evaluate  # noqa: E402
from techcart.ml.tracking import Registry  # noqa: E402

GATES = {
    "intent": {"test_macro_f1": 0.75, "accuracy_when_confident": 0.85},
    # relative + statistical: promote unless significantly WORSE than the popularity baseline
    # (absolute NDCG falls as the catalog grows, and small holdouts make point estimates noisy)
    "recommender": {"ndcg_diff_ci_high": 0.0},
    "fraud": {"roc_auc": 0.85},
    "forecast": {"wape_gain_vs_naive": 0.0},  # must beat seasonal-naive
}


def main() -> None:
    reg = Registry()
    conn = connect()
    P = pd.read_sql("select * from products", conn)
    O = pd.read_sql("select * from orders", conn)
    I = pd.read_sql("select * from order_items", conn)
    C = pd.read_sql("select * from customers", conn)
    summary = {}

    # 1) intent classifier ------------------------------------------------------------
    pipe, m, report = train_and_evaluate()
    v = reg.log_model("intent", pipe, m, params={"model": "tfidf(word+char)+calibrated-logreg"},
                      card={"report": report, "labels": list(pipe.classes_)})
    summary["intent"] = {"version": v, "promoted": reg.promote("intent", v, GATES["intent"]), **m}

    live = O[O["status"] != "cancelled"]
    n_orders = len(live)

    # 2) recommender ------------------------------------------------------------------------
    if n_orders >= recommender.MIN_ORDERS_TO_EVALUATE:
        best_w, m, grid = recommender.tune(P, O, I)  # temporal split on real orders, gated vs popularity
        rec = recommender.HybridRecommender(**best_w).fit(P, live, I)
        v = reg.log_model("recommender", rec, {**m, "mode": "trained", "orders": n_orders},
                          params={**rec.w, "grid": grid})
        promoted = reg.promote("recommender", v, GATES["recommender"])
    else:  # cold start: product similarity + category complements + Amazon review volume
        rec = recommender.HybridRecommender().fit(P, live, I)
        m = {"mode": "cold_start", "orders": n_orders, "needs_orders": recommender.MIN_ORDERS_TO_EVALUATE}
        v = reg.log_model("recommender", rec, m, params=rec.w)
        promoted = reg.promote("recommender", v)
    summary["recommender"] = {"version": v, "promoted": promoted, **m}

    # 3) fraud / order risk -------------------------------------------------------------------
    if n_orders >= fraud.MIN_ORDERS_TO_TRAIN:
        F = fraud.build_features(O, I, C, P)
        model, m = fraud.train(F)
        v = reg.log_model("fraud", model, {**m, "mode": "trained"},
                          params={"model": "IsolationForest", "contamination": 0.01, "features": fraud.FEATURES})
        summary["fraud"] = {"version": v, "promoted": reg.promote("fraud", v, GATES["fraud"]), **m}
    else:
        summary["fraud"] = {"mode": "rules", "orders": n_orders, "needs_orders": fraud.MIN_ORDERS_TO_TRAIN,
                            "note": "orders are scored at checkout with transparent rules until enough history"}

    # 4) revenue forecast ----------------------------------------------------------------------
    series = forecast.daily_series(O) if n_orders else None
    days = 0 if series is None else len(series)
    fc_path = settings.artifacts_dir / "forecast.json"
    if days >= forecast.MIN_DAYS:
        r = forecast.fit_forecast(series)
        v = reg.log_model("forecast", {"model": r["model"], "t0": r["t0"]}, r["metrics"])
        summary["forecast"] = {"version": v, "promoted": reg.promote("forecast", v, GATES["forecast"]),
                               **r["metrics"]}
        fc_path.write_text(json.dumps(r["forecast"], indent=2))
    else:
        summary["forecast"] = {"mode": "waiting_for_data", "days_of_sales": days, "needs_days": forecast.MIN_DAYS}
        fc_path.unlink(missing_ok=True)

    (settings.artifacts_dir / "training_summary.json").write_text(json.dumps(summary, indent=2, default=str))
    print(json.dumps(summary, indent=2, default=str))


if __name__ == "__main__":
    main()
