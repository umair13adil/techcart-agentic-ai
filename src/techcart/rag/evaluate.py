"""RAG evaluation.

Retrieval (deterministic, runs in CI): Hit@k and MRR per retrieval mode (bm25 / vector / hybrid) on
data/rag_eval.jsonl (questions with gold doc ids, generated alongside the data so labels are exact).

Generation (needs Azure OpenAI + LangSmith): pushes the same set to a LangSmith dataset and runs
LLM-as-judge evaluators for correctness (vs reference answer) and faithfulness (answer supported by context).
"""
from __future__ import annotations

import json

from ..config import settings
from .retriever import get_retriever


def load_eval() -> list[dict]:
    return [json.loads(l) for l in (settings.data_dir / "rag_eval.jsonl").read_text().splitlines() if l.strip()]


def retrieval_metrics(k: int = 5, modes=("bm25", "vector", "hybrid")) -> dict:
    r = get_retriever()
    qs = load_eval()
    out = {}
    for mode in modes:
        hits, rr = 0, 0.0
        for q in qs:
            role = "staff" if q["audience"] == "internal" else "customer"
            docs = [h.doc_id for h in r.retrieve(q["q"], role=role, k=k, mode=mode)]
            rank = next((i for i, d in enumerate(docs) if d in q["gold"]), None)
            hits += rank is not None
            rr += 0 if rank is None else 1 / (rank + 1)
        out[mode] = {f"hit@{k}": round(hits / len(qs), 4), "mrr": round(rr / len(qs), 4)}
    # security check: customers must never retrieve internal docs
    leaks = sum(1 for q in qs if q["audience"] == "internal"
                for h in r.retrieve(q["q"], role="customer", k=k) if h.meta.get("audience") != "public")
    out["security_leaks"] = leaks
    out["n_questions"] = len(qs)
    return out


def langsmith_eval(dataset_name: str = "techcart-rag-eval") -> None:  # pragma: no cover - needs keys
    from langsmith import Client, evaluate

    from ..llm import chat_model
    from .retriever import answer

    client = Client()
    if not client.has_dataset(dataset_name=dataset_name):
        ds = client.create_dataset(dataset_name)
        client.create_examples(dataset_id=ds.id,
                               inputs=[{"question": q["q"], "role": "staff" if q["audience"] == "internal" else "customer"} for q in load_eval()],
                               outputs=[{"answer": q["answer"], "gold": q["gold"]} for q in load_eval()])

    def target(inputs: dict) -> dict:
        return answer(inputs["question"], role=inputs["role"])

    judge = chat_model()

    def correctness(run, example) -> dict:
        verdict = judge.invoke(f"Reference: {example.outputs['answer']}\nAnswer: {run.outputs['answer']}\n"
                               "Does the answer contain the reference fact? Reply only YES or NO.").content
        return {"key": "correctness", "score": int("YES" in verdict.upper())}

    def retrieval_hit(run, example) -> dict:
        got = {c["doc_id"] for c in run.outputs.get("citations", [])}
        return {"key": "retrieval_hit", "score": int(bool(got & set(example.outputs["gold"])))}

    def grounded(run, example) -> dict:
        return {"key": "cites_sources", "score": int(run.outputs.get("grounded", False))}

    evaluate(target, data=dataset_name, evaluators=[correctness, retrieval_hit, grounded],
             experiment_prefix="techcart-rag", max_concurrency=4)
