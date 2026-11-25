"""Order risk scoring (fraud / abuse) - unsupervised IsolationForest + transparent rule signals.

Same shape as an insurance claims-fraud model: few labels, heavy class imbalance, analysts need *reasons*.
Labels from the generator (orders.is_anomaly) are used ONLY for evaluation (ROC-AUC, PR-AUC, precision@k),
mirroring how you'd validate against a small set of confirmed chargebacks / SIU-confirmed claims.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

FEATURES = ["log_total", "n_lines", "max_qty", "account_age_days", "night", "city_mismatch", "is_card",
            "prior_orders", "high_value_share", "unit_price_max_log"]
HIGH_VALUE_CATS = {"Laptops", "Graphics Cards", "Desktops"}


def build_features(orders: pd.DataFrame, items: pd.DataFrame, customers: pd.DataFrame,
                   products: pd.DataFrame) -> pd.DataFrame:
    o = orders.merge(customers[["customer_id", "created_at", "city"]].rename(
        columns={"created_at": "cust_created", "city": "cust_city"}), on="customer_id", how="left")
    ts = pd.to_datetime(o["created_at"])
    o["account_age_days"] = (ts - pd.to_datetime(o["cust_created"])).dt.days.clip(lower=0).fillna(0)
    o["night"] = ts.dt.hour.between(1, 5).astype(int)
    o["city_mismatch"] = (o["ship_city"] != o["cust_city"]).astype(int)
    o["is_card"] = (o["payment_method"] == "credit_card").astype(int)
    o["log_total"] = np.log1p(o["total"])
    o = o.sort_values("created_at")
    o["prior_orders"] = o.groupby("customer_id").cumcount()
    it = items.merge(products[["product_id", "category"]], on="product_id", how="left")
    it["hv"] = it["category"].isin(HIGH_VALUE_CATS) * it["line_total"]
    agg = it.groupby("order_id").agg(n_lines=("product_id", "count"), max_qty=("qty", "max"),
                                     hv=("hv", "sum"), tot=("line_total", "sum"), upmax=("unit_price", "max"))
    agg["high_value_share"] = agg["hv"] / agg["tot"].clip(lower=1)
    agg["unit_price_max_log"] = np.log1p(agg["upmax"])
    o = o.merge(agg[["n_lines", "max_qty", "high_value_share", "unit_price_max_log"]], left_on="order_id",
                right_index=True, how="left").fillna(0)
    return o.set_index("order_id")


def reasons(row: pd.Series) -> list[str]:
    r = []
    if row["account_age_days"] < 2:
        r.append("new account")
    if row["night"]:
        r.append("placed 1-5 AM")
    if row["city_mismatch"]:
        r.append("ship city differs from profile")
    if row["max_qty"] >= 3 and row["high_value_share"] > 0.8:
        r.append(f"{int(row['max_qty'])}x high-value units")
    if row["log_total"] > np.log1p(20000):
        r.append("very high basket")
    return r


ALERT_PERCENTILE = 0.99  # top 1% of risk goes to the manual-review queue


class FraudModel:
    """IsolationForest + the reference score distribution, so scores are reported as a percentile
    (0.99 = riskier than 99% of historical orders) - stable, interpretable and easy to threshold."""

    def __init__(self, pipeline: Pipeline, ref_scores: np.ndarray):
        self.pipeline = pipeline
        self.ref = np.sort(ref_scores)

    def raw(self, feats: pd.DataFrame) -> np.ndarray:
        return -self.pipeline.decision_function(feats[FEATURES].to_numpy(dtype=float))

    def percentile(self, feats: pd.DataFrame) -> np.ndarray:
        return np.searchsorted(self.ref, self.raw(feats), side="right") / len(self.ref)


def train(feats: pd.DataFrame, contamination: float = 0.01, seed: int = 0) -> tuple[FraudModel, dict]:
    X = feats[FEATURES].to_numpy(dtype=float)
    pipe = Pipeline([("scale", StandardScaler()),
                     ("iso", IsolationForest(n_estimators=300, contamination=contamination, random_state=seed))])
    pipe.fit(X)
    score = -pipe.decision_function(X)  # higher = more anomalous
    model = FraudModel(pipe, score)
    metrics = {"n_orders": int(len(X)), "alert_percentile": ALERT_PERCENTILE}
    if "is_anomaly" in feats and feats["is_anomaly"].sum() > 0:
        y = feats["is_anomaly"].to_numpy()
        k = int(y.sum())
        top = np.argsort(-score)[:k]
        alerts = score >= np.quantile(score, ALERT_PERCENTILE)
        metrics.update({"alert_precision": round(float(y[alerts].mean()), 4),
                        "alert_recall": round(float(y[alerts].sum() / y.sum()), 4)})
        metrics.update({"roc_auc": round(roc_auc_score(y, score), 4),
                        "pr_auc": round(average_precision_score(y, score), 4),
                        f"precision@{k}": round(float(y[top].mean()), 4),
                        "base_rate": round(float(y.mean()), 4)})
    return model, metrics


def score(model: FraudModel, feats: pd.DataFrame) -> np.ndarray:
    return model.percentile(feats)
