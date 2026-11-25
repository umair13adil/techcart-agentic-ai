"""RAG evaluation.

    python scripts/eval_rag.py                 # retrieval metrics (bm25 vs vector vs hybrid) + security leak check
    python scripts/eval_rag.py --langsmith     # + LLM-judged correctness / citation checks as a LangSmith experiment
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from techcart.rag.evaluate import langsmith_eval, retrieval_metrics  # noqa: E402

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--langsmith", action="store_true")
    ap.add_argument("--min-hit", type=float, default=0.85, help="CI gate on hybrid hit@5")
    a = ap.parse_args()
    m = retrieval_metrics()
    print(json.dumps(m, indent=2))
    if a.langsmith:
        langsmith_eval()
    if m["hybrid"]["hit@5"] < a.min_hit or m["security_leaks"]:
        sys.exit("RAG quality gate failed")
