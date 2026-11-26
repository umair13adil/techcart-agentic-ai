"""Chunk + embed the document corpus into the hybrid index.

    python scripts/build_index.py                  # Azure OpenAI embeddings if configured, else local hashing
    python scripts/build_index.py --embedder hashing
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from techcart.llm import LLMUnavailable  # noqa: E402
from techcart.rag.retriever import build_index  # noqa: E402

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--embedder", default="auto", choices=["auto", "hashing", "azure"])
    try:
        print(json.dumps(build_index(ap.parse_args().embedder), indent=2))
    except LLMUnavailable as e:
        sys.exit(f"\nAzure embeddings unavailable: {e}\n"
                 "Fix .env, or build an offline index for now:  make index ARGS='--embedder hashing'")
