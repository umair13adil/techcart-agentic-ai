"""Build one isolated dataset per test session from the Hugging Face catalog.

The catalog is streamed from Hugging Face on first run and cached at .cache/catalog_hf.jsonl (gitignored, cached in
CI), so later runs need no network. Embeddings are local (hashing) and the assistant is
Foundry-only, so it reports itself unavailable here - no keys and no fake replies.
"""
import os
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
_tmp = tempfile.mkdtemp(prefix="techcart-test-")
os.environ["TECHCART_DATA_DIR"] = _tmp
os.environ["AZURE_OPENAI_ENDPOINT"] = ""      # keep tests free of LLM calls
os.environ["AZURE_OPENAI_API_KEY"] = ""
os.environ["ADMIN_EMAILS"] = "admin@techcart.example"
os.environ["ENTRA_CLIENT_ID"] = ""
os.environ["ENTRA_CLIENT_SECRET"] = ""

import pytest  # noqa: E402


@pytest.fixture(scope="session", autouse=True)
def dataset():
    env = {**os.environ, "PYTHONPATH": str(ROOT / "src")}

    def run(*a):
        r = subprocess.run([sys.executable, *a], cwd=ROOT, env=env, capture_output=True, text=True)
        if r.returncode:
            pytest.exit(f"{a[0]} failed (Hugging Face reachable?):\n{r.stderr[-2000:]}", returncode=1)

    run("scripts/seed.py", "--per-category", "40")
    run("scripts/train_all.py")
    run("scripts/build_index.py", "--embedder", "hashing")
    import sqlite3
    con = sqlite3.connect(Path(_tmp) / "techcart.db")
    tables = ("products", "suppliers", "customers", "orders", "receipts", "quotations", "purchase_orders",
              "search_log", "chat_log")
    counts = {t: con.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0] for t in tables}
    con.close()
    yield {"dir": Path(_tmp), "seed_counts": counts}
