"""Build one isolated dataset per test session from the Hugging Face catalog.

The catalog is streamed from Hugging Face on first run and cached at .cache/catalog_hf.jsonl (gitignored, cached in
CI), so later runs need no network. Embeddings are local (hashing) and the agent is the rule agent, so no keys.
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
os.environ["AGENT_MODE"] = "rules"
os.environ["SIMULATE"] = "0"

import pytest  # noqa: E402


@pytest.fixture(scope="session", autouse=True)
def dataset():
    env = {**os.environ, "PYTHONPATH": str(ROOT / "src")}

    def run(*a):
        r = subprocess.run([sys.executable, *a], cwd=ROOT, env=env, capture_output=True, text=True)
        if r.returncode:
            pytest.exit(f"{a[0]} failed (Hugging Face reachable?):\n{r.stderr[-2000:]}", returncode=1)

    run("scripts/seed.py", "--days", "240", "--customers", "1200")
    run("scripts/train_all.py")
    run("scripts/build_index.py", "--embedder", "hashing")
    run("scripts/simulate_traffic.py")
    yield Path(_tmp)
