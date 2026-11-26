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
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer

from ..data.catalog import COMPLEMENTS, DURABLE


def _norm_cooc(C: sparse.csr_matrix) -> sparse.csr_matrix:
    """Cosine-normalised co-occurrence: C_ij / sqrt(C_ii * C_jj), diagonal removed."""
    d = np.sqrt(np.maximum(C.diagonal(), 1))
    S = sparse.diags(1 / d) @ C @ sparse.diags(1 / d)
    S = S.tolil()
    S.setdiag(0)
    return S.tocsr()


class HybridRecommender:
    """Sparse implementation: memory grows with observed co-purchases, not with products^2, so it handles
    catalogs of tens of thousands of SKUs. Category-prior and content rows are computed on demand."""

    def __init__(self, a=1.0, b=1.2, c=0.25, d=0.08, e=0.0, half_life_days=90):
        self.w = dict(a=a, b=b, c=c, d=d, e=e)  # e = weight of global trending popularity
        self.half_life = half_life_days

    # ------------------------------------------------------------------ fit
    def fit(self, products: pd.DataFrame, orders: pd.DataFrame, items: pd.DataFrame) -> "HybridRecommender":
        self.products = products[["product_id", "title", "category", "subcategory", "brand"]].reset_index(drop=True)
        self.pid = list(self.products["product_id"])
        self.idx = {p: i for i, p in enumerate(self.pid)}
        self.cat = dict(zip(self.products["product_id"], self.products["category"]))
        n = len(self.pid)
        df = items.merge(orders[["order_id", "customer_id", "created_at"]], on="order_id")
        df = df[df["product_id"].isin(self.idx)]
        # co-basket: B (orders x products, binary) -> C = B^T B
        oid = {o: k for k, o in enumerate(df["order_id"].unique())}
        B = sparse.csr_matrix((np.ones(len(df)), (df["order_id"].map(oid), df["product_id"].map(self.idx))),
                              shape=(len(oid), n))
        B.data[:] = 1.0
        self.S_co = _norm_cooc((B.T @ B).tocsr())
        # sequential transitions: items in the previous 3 orders -> items in the next order (same customer)
        rows, cols = [], []
        for _, g in df.sort_values("created_at").groupby("customer_id"):
            baskets = [list({self.idx[p] for p in b["product_id"]}) for _, b in g.groupby("order_id", sort=False)]
            for k in range(1, len(baskets)):
                prev = {i for b in baskets[max(0, k - 3):k] for i in b}
                for i in prev:
                    for j in baskets[k]:
                        if i != j:
                            rows.append(i)
                            cols.append(j)
        T = sparse.csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(n, n))
        rs = np.asarray(T.sum(1)).ravel()
        self.S_tr = (sparse.diags(1 / np.maximum(rs, 1)) @ T).tocsr()
        # category back-off (sparse catalogs): category->category affinity LEARNED from baskets and next orders,
        # blended with the domain prior, times each item's popularity share inside its category.
        self.cats = sorted(set(self.products["category"]))
        cix = {c: k for k, c in enumerate(self.cats)}
        self.cat_of = self.products["category"].map(cix).to_numpy()
        nc = len(self.cats)
        prior = np.zeros((nc, nc), dtype=np.float32)
        for ci in self.cats:
            for r, cj in enumerate(COMPLEMENTS.get(ci, [])):
                if cj in cix:
                    prior[cix[ci], cix[cj]] = 1.0 / (1 + r * 0.5)
        G = sparse.csr_matrix((np.ones(n), (np.arange(n), self.cat_of)), shape=(n, nc))  # item -> category
        learned = (G.T @ ((B.T @ B) + T) @ G).toarray().astype(np.float32)
        np.fill_diagonal(learned, learned.diagonal() * 0.3)  # same-category repeats are weaker signals
        learned /= np.maximum(learned.sum(1, keepdims=True), 1e-9)
        prior /= np.maximum(prior.sum(1, keepdims=True), 1e-9)
        self.P_cat = 0.7 * learned + 0.3 * prior
        units = df.groupby("product_id")["qty"].sum()
        u = np.log1p(np.array([units.get(p, 0) for p in self.pid], dtype=np.float32)) + 0.1
        cat_tot = np.bincount(self.cat_of, weights=u, minlength=nc)
        self.pop_in_cat = (u / cat_tot[self.cat_of]).astype(np.float32)
        # content: L2-normalised TF-IDF rows; similarity row = X @ x_i^T on demand
        text = (self.products["title"] + " " + self.products["category"] + " " + self.products["subcategory"].fillna("")
                + " " + self.products["brand"]).str.lower()
        self.X = TfidfVectorizer(ngram_range=(1, 2), min_df=1, sublinear_tf=True).fit_transform(text).tocsr()
        # popularity (last 30 days of the training window)
        end = pd.to_datetime(df["created_at"]).max()
        recent = df[pd.to_datetime(df["created_at"]) >= end - pd.Timedelta(days=30)]
        pop = recent.groupby("product_id")["qty"].sum()
        self.popularity = np.array([pop.get(p, 0) for p in self.pid], dtype=float)
        self.ref_time = end.to_pydatetime()
        return self

    def _row(self, i: int) -> np.ndarray:
        co = self.S_co.getrow(i).toarray().ravel()
        tr = self.S_tr.getrow(i).toarray().ravel()
        prior = self.P_cat[self.cat_of[i]][self.cat_of] * self.pop_in_cat
        content = (self.X @ self.X.getrow(i).T).toarray().ravel()
        content[i] = 0.0
        return self.w["a"] * co + self.w["b"] * tr + self.w["c"] * prior + self.w["d"] * content

    # ------------------------------------------------------------------ inference
    def score(self, history: list[tuple[str, datetime]], now: datetime | None = None) -> np.ndarray:
        now = now or self.ref_time
        s = np.zeros(len(self.pid))
        for pid, ts in history:
            i = self.idx.get(pid)
            if i is None:
                continue
            age = max(0.0, (now - ts).days)
            s += 0.5 ** (age / self.half_life) * self._row(i)
        return s

    def recommend(self, history: list[tuple[str, datetime]], k: int = 10, in_stock: set | None = None,
                  now: datetime | None = None, explain: bool = False, mode: str = "hybrid") -> list[dict]:
        now = now or self.ref_time
        if mode == "popular" or not history:
            s = self.popularity.copy()
        elif mode == "cobasket":
            s = np.zeros(len(self.pid))
            for p, _ in history:
                if p in self.idx:
                    s += self.S_co.getrow(self.idx[p]).toarray().ravel()
        else:
            s = self.score(history, now)
            s = s / max(s.max(), 1e-9) + (self.w.get("e", 0) + 1e-3) * self.popularity / max(self.popularity.max(), 1)
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
            tcats = {model.cat[t] for t in target}
            results[m]["cat_hit"].append(1.0 if any(model.cat[r] in tcats for r in recs) else 0.0)
    out = {"users_evaluated": len(results["hybrid"]["hit"]), "k": k}
    for m, r in results.items():
        out[f"{m}_hitrate@{k}"] = round(float(np.mean(r["hit"])), 4)
        out[f"{m}_recall@{k}"] = round(float(np.mean(r["recall"])), 4)
        out[f"{m}_ndcg@{k}"] = round(float(np.mean(r["ndcg"])), 4)
        out[f"{m}_catalog_coverage"] = round(len(set(r["items"])) / len(products), 4)
        out[f"{m}_category_hitrate@{k}"] = round(float(np.mean(r["cat_hit"])), 4)
    out["ndcg_lift_vs_popular"] = round(out[f"hybrid_ndcg@{k}"] / max(out[f"popular_ndcg@{k}"], 1e-9), 3)
    # paired bootstrap 95% CI of (hybrid - popular) NDCG: small holdouts can't separate models by point estimates
    h, p = np.array(results["hybrid"]["ndcg"]), np.array(results["popular"]["ndcg"])
    if len(h):
        rng = np.random.default_rng(0)
        boots = [float((h[ix] - p[ix]).mean()) for ix in (rng.integers(0, len(h), len(h)) for _ in range(500))]
        out["ndcg_diff_ci_low"], out["ndcg_diff_ci_high"] = (round(float(np.percentile(boots, q)), 5) for q in (2.5, 97.5))
    return out


GRID = [dict(a=1.0, b=1.2, c=0.25, d=0.08), dict(a=1.0, b=1.2, c=1.0, d=0.02),
        dict(a=1.0, b=1.2, c=1.0, d=0.02, e=0.5), dict(a=0.5, b=0.5, c=2.0, d=0.0, e=1.0),
        dict(a=0.3, b=0.3, c=1.0, d=0.0, e=2.0), dict(a=1.0, b=1.2, c=0.0, d=0.0)]


def tune(products, orders, items, metric: str = "hybrid_ndcg@10") -> tuple[dict, dict, list]:
    """Small grid search on the temporal validation split. (With more time: a separate validation window and a
    final untouched test window, so the reported number isn't the one we tuned on.)"""
    results = [(w, evaluate(products, orders, items, weights=w)) for w in GRID]
    best_w, best_m = max(results, key=lambda r: r[1][metric])
    return best_w, best_m, [{"weights": w, metric: m[metric]} for w, m in results]
