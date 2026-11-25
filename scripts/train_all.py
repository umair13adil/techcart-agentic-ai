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
    "recommender": {"hybrid_ndcg@10": 0.12},
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

    # 2) recommender: evaluate on temporal split, then refit on all data -----------------
    best_w, m, grid = recommender.tune(P, O, I)
    rec = recommender.HybridRecommender(**best_w).fit(P, O[O["status"] != "cancelled"], I)
    v = reg.log_model("recommender", rec, m, params={**rec.w, "grid": grid})
    summary["recommender"] = {"version": v, "promoted": reg.promote("recommender", v, GATES["recommender"]), **m}

    # 3) fraud / order risk ----------------------------------------------------------------
    F = fraud.build_features(O, I, C, P)
    model, m = fraud.train(F)
    v = reg.log_model("fraud", model, m, params={"model": "IsolationForest", "contamination": 0.01,
                                                 "features": fraud.FEATURES})
    summary["fraud"] = {"version": v, "promoted": reg.promote("fraud", v, GATES["fraud"]), **m}
    scores = fraud.score(model, F)
    conn.executemany("UPDATE orders SET risk_score=? WHERE order_id=?",
                     [(round(float(s), 4), oid) for s, oid in zip(scores, F.index)])
    conn.commit()

    # 4) revenue forecast -------------------------------------------------------------------
    r = forecast.fit_forecast(forecast.daily_series(O))
    v = reg.log_model("forecast", {"model": r["model"], "t0": r["t0"]}, r["metrics"])
    summary["forecast"] = {"version": v, "promoted": reg.promote("forecast", v, GATES["forecast"]), **r["metrics"]}
    (settings.artifacts_dir / "forecast.json").write_text(json.dumps(r["forecast"], indent=2))

    (settings.artifacts_dir / "training_summary.json").write_text(json.dumps(summary, indent=2, default=str))
    print(json.dumps(summary, indent=2, default=str))


if __name__ == "__main__":
    main()
