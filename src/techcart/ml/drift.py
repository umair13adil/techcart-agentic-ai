"""Drift monitoring with the Population Stability Index (PSI).

PSI = sum_i (a_i - e_i) * ln(a_i / e_i)   over bins / categories (e = reference, a = current)
Rule of thumb: < 0.1 stable, 0.1-0.25 moderate shift (investigate), > 0.25 significant (retrain / review).

Monitored signals
  * query-intent mix (concept/prior drift in what shoppers ask -> classifier & routing assumptions)
  * classifier confidence + LLM-fallback rate (model uncertainty rising = data drift proxy)
  * order features used by the fraud model (total, lines, night share) - covariate drift
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def psi_categorical(ref: pd.Series, cur: pd.Series, eps: float = 1e-4) -> float:
    cats = sorted(set(ref.dropna()) | set(cur.dropna()))
    e = ref.value_counts(normalize=True).reindex(cats, fill_value=0).to_numpy() + eps
    a = cur.value_counts(normalize=True).reindex(cats, fill_value=0).to_numpy() + eps
    return float(np.sum((a - e) * np.log(a / e)))


def psi_numeric(ref: np.ndarray, cur: np.ndarray, bins: int = 10, eps: float = 1e-4) -> float:
    edges = np.unique(np.quantile(ref, np.linspace(0, 1, bins + 1)))
    if len(edges) < 3:
        return 0.0
    e = np.histogram(np.clip(ref, edges[0], edges[-1]), edges)[0] / len(ref) + eps
    a = np.histogram(np.clip(cur, edges[0], edges[-1]), edges)[0] / max(len(cur), 1) + eps
    return float(np.sum((a - e) * np.log(a / e)))


def status(psi: float) -> str:
    return "stable" if psi < 0.1 else "moderate" if psi < 0.25 else "significant"


def drift_report(conn, ref_days: int = 60, cur_days: int = 7) -> dict:
    logs = pd.read_sql("SELECT * FROM search_log", conn, parse_dates=["ts"])
    orders = pd.read_sql("SELECT order_id, created_at, total FROM orders WHERE status!='cancelled'", conn,
                         parse_dates=["created_at"])
    out: dict = {"signals": {}}
    if len(logs) > 50:
        end = logs["ts"].max()
        cur = logs[logs["ts"] > end - pd.Timedelta(days=cur_days)]
        ref = logs[(logs["ts"] <= end - pd.Timedelta(days=cur_days)) &
                   (logs["ts"] > end - pd.Timedelta(days=cur_days + ref_days))]
        if len(ref) > 20 and len(cur) > 20:
            p = psi_categorical(ref["intent"], cur["intent"])
            out["signals"]["intent_mix"] = {"psi": round(p, 4), "status": status(p)}
            p = psi_numeric(ref["confidence"].to_numpy(), cur["confidence"].to_numpy())
            out["signals"]["classifier_confidence"] = {"psi": round(p, 4), "status": status(p),
                                                       "mean_ref": round(ref["confidence"].mean(), 3),
                                                       "mean_cur": round(cur["confidence"].mean(), 3)}
            out["signals"]["llm_fallback_rate"] = {"ref": round(ref["used_llm"].mean(), 3),
                                                   "cur": round(cur["used_llm"].mean(), 3),
                                                   "low_conf_share_cur": round((cur["confidence"] < 0.55).mean(), 3)}
    if len(orders) > 100:
        end = orders["created_at"].max()
        cur = orders[orders["created_at"] > end - pd.Timedelta(days=cur_days)]
        ref = orders[(orders["created_at"] <= end - pd.Timedelta(days=cur_days)) &
                     (orders["created_at"] > end - pd.Timedelta(days=cur_days + ref_days))]
        if len(cur) > 20:
            p = psi_numeric(np.log1p(ref["total"].to_numpy()), np.log1p(cur["total"].to_numpy()))
            out["signals"]["order_value"] = {"psi": round(p, 4), "status": status(p)}
    worst = max((v.get("psi", 0) for v in out["signals"].values()), default=0)
    out["overall"] = status(worst)
    out["retrain_recommended"] = worst >= 0.25
    return out
