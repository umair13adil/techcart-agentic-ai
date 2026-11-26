"""Set up the store: real products from Hugging Face, opening stock, supplier list, RAG documents.

    python scripts/seed.py                      # first run streams Amazon-Reviews-2023 (Electronics) from Hugging
                                                # Face and caches it to .cache/catalog_hf_<size>.jsonl
    python scripts/seed.py --per-category 400   # bigger catalog (~6,400 products)
    python scripts/seed.py --refresh            # re-download the catalog

No customers, orders, searches or chats are generated: those come only from real use of the app.
Re-seeding wipes the database (including real activity), so stop the API first.
"""
from __future__ import annotations

import argparse
import json
import logging
import shutil
import sys
import time
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from techcart.config import settings  # noqa: E402
from techcart.data.catalog import load_catalog  # noqa: E402
from techcart.data.store_setup import build_inventory, build_rag_eval, supplier_rows, write_documents  # noqa: E402
from techcart.db import connect, init_schema  # noqa: E402


def insert(conn, table: str, recs: list[dict]) -> None:
    if recs:
        cols = list(recs[0])
        conn.executemany(f"INSERT INTO {table} ({','.join(cols)}) VALUES ({','.join('?' * len(cols))})",
                         [tuple(r[c] for c in cols) for r in recs])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-category", type=int, default=150,
                    help="products per category (laptops x2, monitors x1.25); 150 -> ~2,400 products")
    ap.add_argument("--scan-limit", type=int, default=2_000_000, help="max HF records to scan")
    ap.add_argument("--refresh", action="store_true", help="ignore the cached catalog and re-stream from HF")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    t0 = time.time()
    products = load_catalog(per_category=args.per_category, scan_limit=args.scan_limit, refresh=args.refresh)

    settings.data_dir.mkdir(parents=True, exist_ok=True)
    for suffix in ("", "-wal", "-shm"):
        f = settings.db_path.with_name(settings.db_path.name + suffix)
        if f.exists():
            f.unlink()
    conn = connect()
    init_schema(conn)
    insert(conn, "products", products)
    inventory = build_inventory(products)
    insert(conn, "inventory", inventory)
    insert(conn, "suppliers", supplier_rows())
    conn.commit()
    conn.close()

    if settings.docs_dir.exists():
        shutil.rmtree(settings.docs_dir)
    manifest = write_documents(settings.docs_dir, products)
    evalset = build_rag_eval(products)
    (settings.data_dir / "rag_eval.jsonl").write_text("\n".join(json.dumps(q) for q in evalset) + "\n")

    print(json.dumps({
        "products": len(products), "per_category": dict(Counter(p["category"] for p in products)),
        "units_in_stock": sum(i["stock_on_hand"] for i in inventory),
        "out_of_stock_skus": sum(i["stock_on_hand"] == 0 for i in inventory),
        "suppliers": len(supplier_rows()), "documents": len(manifest), "rag_eval_questions": len(evalset),
        "customers": 0, "orders": 0, "seconds": round(time.time() - t0, 1), "db": str(settings.db_path),
    }, indent=2))


if __name__ == "__main__":
    main()
