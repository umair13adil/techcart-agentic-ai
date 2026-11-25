"""Fine-tune a small Azure OpenAI model as the tier-2 query parser, and evaluate it against the classical tier-1.

    python scripts/finetune_intent_aoai.py --prepare                 # write train/val JSONL (chat format)
    python scripts/finetune_intent_aoai.py --submit --base gpt-4.1-mini-2025-04-14
    python scripts/finetune_intent_aoai.py --evaluate <deployment>   # after deploying the fine-tuned model

When is fine-tuning worth it here? Only if (a) the base model + few-shot prompt misses the label taxonomy
(e.g. 'quote_request' vs 'purchase' for bulk wording), (b) we need shorter prompts for latency/cost at volume, or
(c) we want consistent JSON without a long schema prompt. Knowledge (prices, policies) stays in RAG - fine-tuning
facts in would go stale with every price change. Check model/region availability for fine-tuning in Foundry first.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from techcart.config import settings  # noqa: E402
from techcart.ml.intent import extract_entities  # noqa: E402
from techcart.ml.intent_data import INTENTS, TEST_SET, generate  # noqa: E402

SYSTEM = ("Classify a shopper message for a Saudi computer store. Reply with JSON only: "
          '{"intent": one of ' + json.dumps(INTENTS) + ', "category": string|null, "max_price": number|null}')
OUT = settings.data_dir / "finetune"


def example(q: str, intent: str) -> dict:
    e = extract_entities(q)
    target = {"intent": intent, "category": e.category, "max_price": e.max_price}
    return {"messages": [{"role": "system", "content": SYSTEM}, {"role": "user", "content": q},
                         {"role": "assistant", "content": json.dumps(target)}]}


def prepare(n_per_intent: int = 120, seed: int = 21) -> tuple[Path, Path]:
    data = generate(n_per_intent=n_per_intent, seed=seed)
    # + human-reviewed low-confidence production queries would be appended here (active-learning loop)
    random.Random(seed).shuffle(data)
    cut = int(len(data) * 0.9)
    OUT.mkdir(parents=True, exist_ok=True)
    tr, va = OUT / "train.jsonl", OUT / "val.jsonl"
    tr.write_text("\n".join(json.dumps(example(q, i)) for q, i in data[:cut]) + "\n")
    va.write_text("\n".join(json.dumps(example(q, i)) for q, i in data[cut:]) + "\n")
    print(f"wrote {cut} train / {len(data) - cut} val examples to {OUT}")
    return tr, va


def submit(base: str) -> None:
    from techcart.llm import embeddings_client  # same AzureOpenAI client

    client = embeddings_client()
    tr, va = prepare()
    ftr = client.files.create(file=tr.open("rb"), purpose="fine-tune")
    fva = client.files.create(file=va.open("rb"), purpose="fine-tune")
    job = client.fine_tuning.jobs.create(model=base, training_file=ftr.id, validation_file=fva.id,
                                         suffix="techcart-intent", hyperparameters={"n_epochs": 3}, seed=7)
    print("submitted:", job.id, "- monitor in Foundry; deploy the resulting model, then run --evaluate")


def evaluate(deployment: str) -> None:
    from techcart.llm import embeddings_client
    from techcart.ml.tracking import Registry

    client = embeddings_client()
    clf = Registry().load("intent")
    rows = []
    for q, gold in TEST_SET:
        t = time.perf_counter()
        r = client.chat.completions.create(model=deployment, messages=[{"role": "system", "content": SYSTEM},
                                                                       {"role": "user", "content": q}], max_tokens=60)
        ms = (time.perf_counter() - t) * 1000
        try:
            pred = json.loads(r.choices[0].message.content)["intent"]
        except Exception:
            pred = "PARSE_ERROR"
        rows.append({"q": q, "gold": gold, "ft": pred, "clf": clf.predict([q])[0], "ms": ms})
    acc = lambda k: sum(r[k] == r["gold"] for r in rows) / len(rows)  # noqa: E731
    print(json.dumps({"finetuned_accuracy": round(acc("ft"), 4), "classifier_accuracy": round(acc("clf"), 4),
                      "finetuned_p50_ms": sorted(r["ms"] for r in rows)[len(rows) // 2]}, indent=2))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--prepare", action="store_true")
    ap.add_argument("--submit", action="store_true")
    ap.add_argument("--base", default="gpt-4.1-mini-2025-04-14")
    ap.add_argument("--evaluate", metavar="DEPLOYMENT")
    a = ap.parse_args()
    if a.prepare:
        prepare()
    if a.submit:
        submit(a.base)
    if a.evaluate:
        evaluate(a.evaluate)
