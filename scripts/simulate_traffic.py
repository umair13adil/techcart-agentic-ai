"""Backfill ~30 days of realistic search traffic into search_log (so the admin search-insights panel and the
drift monitor have data on first run). The last 7 days deliberately shift toward gaming + quote requests
(White Friday season) so the drift monitor has something to detect.

    python scripts/simulate_traffic.py
"""
from __future__ import annotations

import random
import sys
from datetime import datetime, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from techcart.db import connect  # noqa: E402
from techcart.ml.intent import QueryUnderstander  # noqa: E402
from techcart.ml.intent_data import generate  # noqa: E402
from techcart.ml.tracking import Registry  # noqa: E402


def main(n: int = 2500, seed: int = 5) -> None:
    rng = random.Random(seed)
    qu = QueryUnderstander(Registry().load("intent"), use_llm=False)
    pool = generate(n_per_intent=400, seed=99)
    by_intent: dict[str, list[str]] = {}
    for q, i in pool:
        by_intent.setdefault(i, []).append(q)
    base = {"product_search": 0.38, "comparison": 0.08, "recommendation": 0.07, "order_status": 0.14,
            "return_refund": 0.06, "policy_question": 0.13, "purchase": 0.08, "quote_request": 0.02, "other": 0.04}
    shifted = {**base, "product_search": 0.42, "quote_request": 0.09, "order_status": 0.2, "policy_question": 0.08,
               "recommendation": 0.04}
    now = datetime.now()
    conn = connect()
    conn.execute("DELETE FROM search_log")
    rows = []
    for _ in range(n):
        ts = now - timedelta(days=rng.random() * 30, hours=rng.random() * 3)
        mix = shifted if (now - ts).days < 7 else base
        intent = rng.choices(list(mix), weights=list(mix.values()))[0]
        q = rng.choice(by_intent[intent])
        if (now - ts).days < 7 and intent == "product_search" and rng.random() < 0.5:
            q = rng.choice(["gaming laptop white friday deal", "rtx 4070 laptop offer", "best gaming monitor sale",
                            "cheap gaming headset", "ps5 monitor 144hz"])
        u = qu.understand(q)
        rows.append((ts.isoformat(timespec="seconds"), f"sim{rng.randint(1, 900)}", q, u.intent, u.confidence,
                     int(u.confidence < 0.55)))
    conn.executemany("INSERT INTO search_log VALUES (?,?,?,?,?,?)", rows)
    conn.commit()
    print(f"inserted {len(rows)} search events")


if __name__ == "__main__":
    main()
