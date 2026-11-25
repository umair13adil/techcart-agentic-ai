"""Embedding backends with an on-disk cache keyed by content hash (re-indexing only pays for changed chunks).

* AzureOpenAIEmbedder - text-embedding-3-small deployment (1536-d), batched, retried
* HashingEmbedder     - local, deterministic, no network: char n-gram hashing + L2 norm. Weaker semantics, but
                        lets CI and offline demos run the full pipeline.
"""
from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import HashingVectorizer

from ..config import settings


class HashingEmbedder:
    name = "hashing-char-1024"
    dim = 1024

    def __init__(self):
        self.v = HashingVectorizer(analyzer="char_wb", ngram_range=(3, 5), n_features=self.dim,
                                   alternate_sign=False, norm="l2")

    def embed(self, texts: list[str]) -> np.ndarray:
        return self.v.transform([t.lower() for t in texts]).toarray().astype(np.float32)


class AzureOpenAIEmbedder:
    def __init__(self, deployment: str | None = None, batch: int = 64):
        from ..llm import embeddings_client

        self.client = embeddings_client()
        self.deployment = deployment or settings.aoai_embed_deployment
        self.name = f"aoai-{self.deployment}"
        self.batch = batch
        self.dim = None

    def embed(self, texts: list[str]) -> np.ndarray:
        out = []
        for i in range(0, len(texts), self.batch):
            chunk = texts[i:i + self.batch]
            for attempt in range(5):
                try:
                    resp = self.client.embeddings.create(model=self.deployment, input=chunk)
                    out.extend(d.embedding for d in resp.data)
                    break
                except Exception:  # 429 / transient -> exponential backoff
                    if attempt == 4:
                        raise
                    time.sleep(2 ** attempt)
        arr = np.asarray(out, dtype=np.float32)
        arr /= np.linalg.norm(arr, axis=1, keepdims=True) + 1e-9
        self.dim = arr.shape[1]
        return arr


class CachedEmbedder:
    def __init__(self, inner, cache_dir: Path | None = None):
        self.inner = inner
        self.name = inner.name
        self.dir = (cache_dir or settings.data_dir / "emb_cache") / inner.name
        self.dir.mkdir(parents=True, exist_ok=True)
        self.index_path = self.dir / "index.json"
        self.index: dict[str, int] = json.loads(self.index_path.read_text()) if self.index_path.exists() else {}
        self.mat_path = self.dir / "vectors.npy"
        self.mat = np.load(self.mat_path) if self.mat_path.exists() else None

    def embed(self, texts: list[str]) -> np.ndarray:
        keys = [hashlib.sha1(t.encode()).hexdigest() for t in texts]
        missing = [(k, t) for k, t in dict(zip(keys, texts)).items() if k not in self.index]
        if missing:
            vecs = self.inner.embed([t for _, t in missing])
            start = 0 if self.mat is None else len(self.mat)
            self.mat = vecs if self.mat is None else np.vstack([self.mat, vecs])
            for n, (k, _) in enumerate(missing):
                self.index[k] = start + n
            np.save(self.mat_path, self.mat)
            self.index_path.write_text(json.dumps(self.index))
        return self.mat[[self.index[k] for k in keys]]


def get_embedder(kind: str = "auto"):
    if kind == "hashing" or (kind == "auto" and not settings.llm_enabled):
        return CachedEmbedder(HashingEmbedder())
    return CachedEmbedder(AzureOpenAIEmbedder())
