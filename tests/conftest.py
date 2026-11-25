"""Build a small, isolated dataset once per test session (offline catalog, hashing embeddings, no LLM)."""
import os
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
_tmp = tempfile.mkdtemp(prefix="techcart-test-")
os.environ["TECHCART_DATA_DIR"] = _tmp
os.environ["AZURE_OPENAI_ENDPOINT"] = ""      # force offline paths in tests
os.environ["AZURE_OPENAI_API_KEY"] = ""
os.environ["AGENT_MODE"] = "rules"
os.environ["SIMULATE"] = "0"

import pytest  # noqa: E402


@pytest.fixture(scope="session", autouse=True)
def dataset():
    env = {**os.environ, "PYTHONPATH": str(ROOT / "src")}
    run = lambda *a: subprocess.run([sys.executable, *a], cwd=ROOT, env=env, check=True, capture_output=True)  # noqa: E731
    run("scripts/seed.py", "--source", "offline", "--days", "240", "--customers", "1200")
    run("scripts/train_all.py")
    run("scripts/build_index.py", "--embedder", "hashing")
    run("scripts/simulate_traffic.py")
    yield Path(_tmp)
