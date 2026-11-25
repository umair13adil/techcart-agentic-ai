"""Hybrid "next purchase" recommender.

score(j | user) =  sum_i  recency(i) * [ a * cobasket_sim(i, j) + b * transition_sim(i, j) ]   (behavioural)
                 + c * complement_prior(cat(i) -> cat(j))                                       (domain knowledge)
                 + d * content_sim(i, j)                                                        (cold start)
then business rules: drop owned items, suppress just-bought durable categories, in-stock only,
max 2 per category (diversity). Anonymous users / no history -> trending popularity.

Evaluated with a temporal split (train < cutoff, predict each user's first post-cutoff basket):
HitRate@K, Recall@K, NDCG@K against popularity and co-basket-only baselines.
"""
from __future__ import annotations

import math
from collections import defaultdict
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from ..data.catalog import COMPLEMENTS, DURABLE


def _norm_cooc(M: np.ndarray) -> np.ndarray:
    d = np.sqrt(np.maximum(np.diag(M).copy(), 1))
    S = M / d[:, None] / d[None, :]
    np.fill_diagonal(S, 0)
    return S


class HybridRecommender:
    def __init__(self, a=1.0, b=1.2, c=0.25, d=0.08, half_life_days=90):
        self.w = dict(a=a, b=b, c=c, d=d)
        self.half_life = half_life_days

    # ------------------------------------------------------------------ fit
    def fit(self, products: pd.DataFrame, orders: pd.DataFrame, items: pd.DataFrame) -> "HybridRecommender":
        self.products = products.reset_index(drop=True)
        self.pid = list(self.products["product_id"])
        self.idx = {p: i for i, p in enumerate(self.pid)}
        self.cat = dict(zip(self.products["product_id"], self.products["category"]))
        n = len(self.pid)
        df = items.merge(orders[["order_id", "customer_id", "created_at"]], on="order_id")
        df = df[df["product_id"].isin(self.idx)]
        # co-basket counts
        C = np.zeros((n, n))
        for _, g in df.groupby("order_id"):
            ix = [self.idx[p] for p in set(g["product_id"])]
            for i in ix:
                for j in ix:
                    C[i, j] += 1
        # sequential transitions: anything in an earlier order -> anything in a later order (same customer)
        T = np.zeros((n, n))
        for _, g in df.sort_values("created_at").groupby("customer_id"):
            baskets = [list({self.idx[p] for p in b["product_id"]}) for _, b in g.groupby("order_id", sort=False)]
            for k in range(1, len(baskets)):
                prev = {i for b in baskets[max(0, k - 3):k] for i in b}
                for i in prev:
                    for j in baskets[k]:
                        if i != j:
                            T[i, j] += 1
        self.S_co = _norm_cooc(C)
        tn = T / np.maximum(T.sum(1, keepdims=True), 1)
        self.S_tr = tn
        # category complement prior
        cats = self.products["category"].to_numpy()
        P = np.zeros((n, n))
        for i, ci in enumerate(cats):
            comps = COMPLEMENTS.get(ci, [])
            for r, cj in enumerate(comps):
                P[i, cats == cj] = 1.0 / (1 + r * 0.5)
        self.S_prior = P
        text = (self.products["title"] + " " + self.products["category"] + " " + self.products["subcategory"].fillna("")
                + " " + self.products["brand"]).str.lower()
        self.S_content = cosine_similarity(TfidfVectorizer(ngram_range=(1, 2)).fit_transform(text))
        np.fill_diagonal(self.S_content, 0)
        # popularity (last 30 days of the training window)
        end = pd.to_datetime(df["created_at"]).max()
        recent = df[pd.to_datetime(df["created_at"]) >= end - pd.Timedelta(days=30)]
        pop = recent.groupby("product_id")["qty"].sum()
        self.popularity = np.array([pop.get(p, 0) for p in self.pid], dtype=float)
        self.ref_time = end.to_pydatetime()
        return self

    # ------------------------------------------------------------------ inference
    def score(self, history: list[tuple[str, datetime]], now: datetime | None = None) -> np.ndarray:
        now = now or self.ref_time
        s = np.zeros(len(self.pid))
        for pid, ts in history:
            i = self.idx.get(pid)
            if i is None:
                continue
            age = max(0.0, (now - ts).days)
            r = 0.5 ** (age / self.half_life)
            s += r * (self.w["a"] * self.S_co[i] + self.w["b"] * self.S_tr[i] + self.w["c"] * self.S_prior[i]
                      + self.w["d"] * self.S_content[i])
        return s

    def recommend(self, history: list[tuple[str, datetime]], k: int = 10, in_stock: set | None = None,
                  now: datetime | None = None, explain: bool = False, mode: str = "hybrid") -> list[dict]:
        now = now or self.ref_time
        if mode == "popular" or not history:
            s = self.popularity.copy()
        elif mode == "cobasket":
            s = sum(self.S_co[self.idx[p]] for p, _ in history if p in self.idx)
        else:
            s = self.score(history, now) + 1e-3 * self.popularity / max(self.popularity.max(), 1)
        owned = {p for p, _ in history}
        recent_durables = {self.cat[p] for p, ts in history if self.cat.get(p) in DURABLE
                           and (now - ts) < timedelta(days=180)}
        out, per_cat = [], defaultdict(int)
        for j in np.argsort(-s):
            p = self.pid[j]
            c = self.cat[p]
            if p in owned or c in recent_durables or (in_stock is not None and p not in in_stock):
                continue
            if per_cat[c] >= 2 or s[j] <= 0:
                continue
            per_cat[c] += 1
            rec = {"product_id": p, "score": round(float(s[j]), 4)}
            if explain and history:
                anchors = [(hp, self.S_co[self.idx[hp], j] + self.S_tr[self.idx[hp], j]) for hp, _ in history
                           if hp in self.idx]
                best = max(anchors, key=lambda x: x[1])[0] if anchors else None
                rec["because_of"] = best
                rec["reason"] = f"Customers who bought {self.cat.get(best)} often add {c}" if best else "Trending"
            out.append(rec)
            if len(out) >= k:
                break
        return out


# ---------------------------------------------------------------------- offline evaluation
def evaluate(products: pd.DataFrame, orders: pd.DataFrame, items: pd.DataFrame, holdout_days: int = 60,
             k: int = 10, weights: dict | None = None) -> dict:
    orders = orders[orders["status"] != "cancelled"].copy()
    orders["ts"] = pd.to_datetime(orders["created_at"])
    cutoff = orders["ts"].max() - pd.Timedelta(days=holdout_days)
    train_o, test_o = orders[orders["ts"] < cutoff], orders[orders["ts"] >= cutoff]
    model = HybridRecommender(**(weights or {})).fit(products, train_o, items[items["order_id"].isin(train_o["order_id"])])
    hist = items.merge(train_o[["order_id", "customer_id", "ts"]], on="order_id")
    hist_by = {c: list(zip(g["product_id"], g["ts"].dt.to_pydatetime())) for c, g in hist.groupby("customer_id")}
    first_test = test_o.sort_values("ts").groupby("customer_id").head(1)
    truth = items[items["order_id"].isin(first_test["order_id"])].merge(first_test[["order_id", "customer_id", "ts"]])
    results = {m: defaultdict(list) for m in ("hybrid", "cobasket", "popular")}
    for cust, g in truth.groupby("customer_id"):
        h = hist_by.get(cust)
        if not h:
            continue
        owned = {p for p, _ in h}
        target = set(g["product_id"]) - owned
        if not target:
            continue
        now = g["ts"].iloc[0].to_pydatetime()
        for m in results:
            recs = [r["product_id"] for r in model.recommend(h, k=k, now=now, mode=m)]
            hits = [1 if r in target else 0 for r in recs]
            dcg = sum(hh / math.log2(i + 2) for i, hh in enumerate(hits))
            idcg = sum(1 / math.log2(i + 2) for i in range(min(len(target), k)))
            results[m]["hit"].append(1.0 if any(hits) else 0.0)
            results[m]["recall"].append(sum(hits) / len(target))
            results[m]["ndcg"].append(dcg / idcg if idcg else 0)
            results[m]["items"].extend(recs)
    out = {"users_evaluated": len(results["hybrid"]["hit"]), "k": k}
    for m, r in results.items():
        out[f"{m}_hitrate@{k}"] = round(float(np.mean(r["hit"])), 4)
        out[f"{m}_recall@{k}"] = round(float(np.mean(r["recall"])), 4)
        out[f"{m}_ndcg@{k}"] = round(float(np.mean(r["ndcg"])), 4)
        out[f"{m}_catalog_coverage"] = round(len(set(r["items"])) / len(products), 4)
    return out


GRID = [dict(a=1.0, b=1.2, c=0.25, d=0.08), dict(a=1.0, b=0.6, c=0.1, d=0.05), dict(a=1.0, b=0.3, c=0.05, d=0.0),
        dict(a=1.0, b=1.2, c=0.0, d=0.0), dict(a=0.6, b=1.0, c=0.1, d=0.02)]


def tune(products, orders, items, metric: str = "hybrid_ndcg@10") -> tuple[dict, dict, list]:
    """Small grid search on the temporal validation split. (With more time: a separate validation window and a
    final untouched test window, so the reported number isn't the one we tuned on.)"""
    results = [(w, evaluate(products, orders, items, weights=w)) for w in GRID]
    best_w, best_m = max(results, key=lambda r: r[1][metric])
    return best_w, best_m, [{"weights": w, metric: m[metric]} for w, m in results]
