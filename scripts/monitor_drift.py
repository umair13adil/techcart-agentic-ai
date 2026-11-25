"""Drift check for scheduled CI. Exit code 3 when retraining is recommended (the workflow then runs training).

    python scripts/monitor_drift.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from techcart.db import connect  # noqa: E402
from techcart.ml.drift import drift_report  # noqa: E402

if __name__ == "__main__":
    rep = drift_report(connect())
    print(json.dumps(rep, indent=2))
    sys.exit(3 if rep["retrain_recommended"] else 0)
