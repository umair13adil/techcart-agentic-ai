"""Seed the TechCart database + RAG documents from the Hugging Face catalog.

    python scripts/seed.py              # first run streams Amazon-Reviews-2023 (Electronics) from Hugging Face
                                        # and caches it to .cache/catalog_hf.jsonl; later runs use the cache
    python scripts/seed.py --refresh    # re-download the catalog
"""
from __future__ import annotations

import argparse
import json
import logging
import shutil
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from techcart.config import settings  # noqa: E402
from techcart.data.catalog import load_catalog  # noqa: E402
from techcart.data.generate import Generator, build_rag_eval, write_db, write_documents  # noqa: E402
from techcart.db import connect  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--refresh", action="store_true", help="ignore the cached catalog and re-stream from HF")
    ap.add_argument("--scan-limit", type=int, default=1_500_000, help="max HF records to scan")
    ap.add_argument("--days", type=int, default=540)
    ap.add_argument("--customers", type=int, default=3200)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    t0 = time.time()
    products = load_catalog(scan_limit=args.scan_limit, refresh=args.refresh)
    gen = Generator(products, seed=args.seed, days=args.days, n_customers=args.customers)
    gen.make_customers()
    gen.make_orders()
    supply = gen.make_supply()

    settings.data_dir.mkdir(parents=True, exist_ok=True)
    for suffix in ("", "-wal", "-shm"):  # stop the API before re-seeding
        f = settings.db_path.with_name(settings.db_path.name + suffix)
        if f.exists():
            f.unlink()
    conn = connect()
    write_db(conn, gen, supply)
    conn.close()

    if settings.docs_dir.exists():
        shutil.rmtree(settings.docs_dir)
    manifest = write_documents(settings.docs_dir, gen, supply)
    evalset = build_rag_eval(gen, supply)
    (settings.data_dir / "rag_eval.jsonl").write_text("\n".join(json.dumps(q) for q in evalset) + "\n")

    revenue = sum(o["total"] for o in gen.orders if o["status"] != "cancelled")
    print(json.dumps({
        "products": len(products), "customers": len(gen.customers), "orders": len(gen.orders),
        "order_items": len(gen.items), "anomalies": sum(o["is_anomaly"] for o in gen.orders),
        "purchase_orders": len(supply["purchase_orders"]), "quotations": len(supply["quotations"]),
        "receipts": len(supply["receipts"]), "documents": len(manifest), "rag_eval_questions": len(evalset),
        "revenue_sar": round(revenue), "seconds": round(time.time() - t0, 1), "db": str(settings.db_path),
    }, indent=2))


if __name__ == "__main__":
    main()
