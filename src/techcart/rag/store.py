"""Vector + keyword index with hybrid retrieval and metadata filtering.

LocalHybridStore   numpy dense index + BM25 sparse index, fused with Reciprocal Rank Fusion (RRF).
AzureSearchStore   same interface on Azure AI Search (hybrid query + OData filter, optional semantic ranker).

Why hybrid: dense vectors catch paraphrase ("lappy for uni" ~ "student laptop"), BM25 nails exact tokens
(SKUs, "QT-C-0041", "RTX 4070", "PO-2026-00412") that embeddings blur. RRF needs no score calibration.
Why filters: security trimming (audience = public | internal | owner) happens INSIDE retrieval, never in the
prompt, so a customer can never be shown a purchase order or someone else's receipt.
"""
from __future__ import annotations

import json
import math
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .chunking import Chunk

TOKEN = re.compile(r"[a-z0-9][a-z0-9\-\.]*")


def tokenize(t: str) -> list[str]:
    return TOKEN.findall(t.lower())


class BM25:
    def __init__(self, docs: list[str], k1: float = 1.4, b: float = 0.75):
        self.k1, self.b = k1, b
        self.toks = [tokenize(d) for d in docs]
        self.len = np.array([len(t) for t in self.toks], dtype=float)
        self.avg = float(self.len.mean()) if len(self.len) else 0.0
        self.tf = [Counter(t) for t in self.toks]
        df = Counter(w for t in self.toks for w in set(t))
        n = len(docs)
        self.idf = {w: math.log(1 + (n - f + 0.5) / (f + 0.5)) for w, f in df.items()}

    def scores(self, query: str, candidates: np.ndarray | None = None) -> np.ndarray:
        q = tokenize(query)
        idx = range(len(self.toks)) if candidates is None else candidates
        out = np.zeros(len(self.toks))
        for i in idx:
            tf, s = self.tf[i], 0.0
            for w in q:
                if w in tf:
                    s += self.idf[w] * tf[w] * (self.k1 + 1) / (
                        tf[w] + self.k1 * (1 - self.b + self.b * self.len[i] / self.avg))
            out[i] = s
        return out


@dataclass
class Hit:
    chunk_id: str
    doc_id: str
    text: str
    meta: dict
    score: float
    sources: dict


def _match(meta: dict, flt: dict) -> bool:
    for k, v in flt.items():
        if k == "audience_in":
            if meta.get("audience") not in v:
                return False
        elif k == "price_max":
            if meta.get("price") is not None and meta["price"] > v:
                return False
        elif k == "price_min":
            if meta.get("price") is not None and meta["price"] < v:
                return False
        elif k == "doc_ids":
            if meta.get("doc_id") not in v:
                return False
        elif isinstance(v, (list, set, tuple)):
            if meta.get(k) not in v:
                return False
        elif meta.get(k) != v:
            return False
    return True


class LocalHybridStore:
    def __init__(self, chunks: list[Chunk], vectors: np.ndarray, embedder_name: str):
        self.chunks, self.vectors, self.embedder_name = chunks, vectors, embedder_name
        self.bm25 = BM25([c.text for c in chunks])
        self.metas = [{**c.meta, "doc_id": c.doc_id} for c in chunks]

    # -------------------------------------------------------------- persistence
    def save(self, path: Path) -> None:
        path.mkdir(parents=True, exist_ok=True)
        np.save(path / "vectors.npy", self.vectors)
        with (path / "chunks.jsonl").open("w") as f:
            for c in self.chunks:
                f.write(json.dumps({"chunk_id": c.chunk_id, "doc_id": c.doc_id, "text": c.text, "meta": c.meta}) + "\n")
        (path / "info.json").write_text(json.dumps({"embedder": self.embedder_name, "n": len(self.chunks)}))

    @classmethod
    def load(cls, path: Path) -> "LocalHybridStore":
        chunks = [Chunk(**json.loads(l)) for l in (path / "chunks.jsonl").read_text().splitlines()]
        info = json.loads((path / "info.json").read_text())
        return cls(chunks, np.load(path / "vectors.npy"), info["embedder"])

    # -------------------------------------------------------------- search
    def search(self, query: str, qvec: np.ndarray | None, k: int = 6, filters: dict | None = None,
               mode: str = "hybrid", rrf_k: int = 60, pool: int = 40) -> list[Hit]:
        cand = np.array([i for i, m in enumerate(self.metas) if _match(m, filters or {})], dtype=int)
        if cand.size == 0:
            return []
        ranks: dict[str, dict[int, int]] = {}
        if mode in ("hybrid", "vector") and qvec is not None:
            sims = self.vectors[cand] @ qvec
            order = cand[np.argsort(-sims)[:pool]]
            ranks["vector"] = {int(i): r for r, i in enumerate(order)}
        if mode in ("hybrid", "bm25"):
            s = self.bm25.scores(query, cand)
            order = [i for i in cand[np.argsort(-s[cand])[:pool]] if s[i] > 0]
            ranks["bm25"] = {int(i): r for r, i in enumerate(order)}
        fused: dict[int, float] = {}
        for name, rk in ranks.items():
            for i, r in rk.items():
                fused[i] = fused.get(i, 0.0) + 1.0 / (rrf_k + r + 1)
        best = sorted(fused, key=lambda i: -fused[i])
        hits, seen_docs = [], Counter()
        for i in best:
            c = self.chunks[i]
            if seen_docs[c.doc_id] >= 2:  # diversity: at most 2 chunks per document
                continue
            seen_docs[c.doc_id] += 1
            hits.append(Hit(c.chunk_id, c.doc_id, c.text, self.metas[i], round(fused[i], 5),
                            {n: rk.get(i) for n, rk in ranks.items()}))
            if len(hits) >= k:
                break
        return hits


class AzureSearchStore:  # pragma: no cover - requires an Azure AI Search service
    """Same `search` contract backed by Azure AI Search. Index fields: id, doc_id, content, content_vector,
    audience, doc_type, category, brand, price, title. Create with scripts/build_index.py --backend azure_search."""

    def __init__(self, endpoint: str, key: str, index: str):
        from azure.core.credentials import AzureKeyCredential
        from azure.search.documents import SearchClient

        self.client = SearchClient(endpoint, index, AzureKeyCredential(key))

    def search(self, query, qvec, k=6, filters=None, mode="hybrid", **_):
        from azure.search.documents.models import VectorizedQuery

        odata = []
        for key, v in (filters or {}).items():
            if key == "audience_in":
                odata.append("search.in(audience, '" + ",".join(v) + "', ',')")
            elif key == "price_max":
                odata.append(f"price le {v}")
            elif key == "price_min":
                odata.append(f"price ge {v}")
            elif isinstance(v, str):
                odata.append(f"{key} eq '{v}'")
        vq = [VectorizedQuery(vector=qvec.tolist(), k_nearest_neighbors=40, fields="content_vector")] \
            if qvec is not None and mode != "bm25" else None
        res = self.client.search(search_text=query if mode != "vector" else None, vector_queries=vq,
                                 filter=" and ".join(odata) or None, top=k)
        return [Hit(r["id"], r["doc_id"], r["content"], {k2: r.get(k2) for k2 in ("title", "doc_type", "audience")},
                    r["@search.score"], {}) for r in res]
