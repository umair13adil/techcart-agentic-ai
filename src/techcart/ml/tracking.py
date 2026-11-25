"""Experiment tracking + model registry.

Uses MLflow when installed (set MLFLOW_TRACKING_URI to an Azure ML workspace URI to log there - Azure ML
is MLflow-native), otherwise a tiny file-based registry so the project runs anywhere.

Registry layout (file backend):
    data/artifacts/registry.json
    data/artifacts/<model_name>/v<N>/model.joblib + metrics.json + card.json
Each model has a "production" alias that the API loads; promotion is gated on metrics (see promote()).
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any

import joblib

from ..config import settings

try:  # optional
    import mlflow  # type: ignore
except Exception:  # pragma: no cover
    mlflow = None


class Registry:
    def __init__(self, root: Path | None = None):
        self.root = root or settings.artifacts_dir
        self.root.mkdir(parents=True, exist_ok=True)
        self.index_path = self.root / "registry.json"
        self.index: dict[str, Any] = json.loads(self.index_path.read_text()) if self.index_path.exists() else {}

    def _save(self) -> None:
        self.index_path.write_text(json.dumps(self.index, indent=2))

    def log_model(self, name: str, model: Any, metrics: dict, params: dict | None = None,
                  card: dict | None = None) -> int:
        entry = self.index.setdefault(name, {"versions": [], "production": None})
        version = len(entry["versions"]) + 1
        vdir = self.root / name / f"v{version}"
        vdir.mkdir(parents=True, exist_ok=True)
        joblib.dump(model, vdir / "model.joblib")
        (vdir / "metrics.json").write_text(json.dumps(metrics, indent=2))
        (vdir / "card.json").write_text(json.dumps(card or {}, indent=2))
        entry["versions"].append({"version": version, "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
                                  "metrics": metrics, "params": params or {}})
        self._save()
        if mlflow is not None and os.environ.get("MLFLOW_TRACKING_URI"):
            mlflow.set_experiment(f"techcart-{name}")
            with mlflow.start_run(run_name=f"{name}-v{version}"):
                mlflow.log_params({k: str(v)[:500] for k, v in (params or {}).items()})
                mlflow.log_metrics({k: float(v) for k, v in metrics.items() if isinstance(v, (int, float))})
                mlflow.log_artifacts(str(vdir))
        return version

    def promote(self, name: str, version: int, gate: dict[str, float] | None = None) -> bool:
        """Promote to production only if every gate metric >= threshold and beats current prod."""
        entry = self.index[name]
        cand = entry["versions"][version - 1]["metrics"]
        for k, thr in (gate or {}).items():
            if cand.get(k, float("-inf")) < thr:
                print(f"[registry] {name} v{version} blocked: {k}={cand.get(k)} < {thr}")
                return False
        prod = entry.get("production")
        if prod and gate:
            cur = entry["versions"][prod - 1]["metrics"]
            key = next(iter(gate))
            if cand.get(key, 0) + 1e-9 < cur.get(key, 0) - 0.01:  # allow tiny regressions only
                print(f"[registry] {name} v{version} blocked: worse than prod v{prod} on {key}")
                return False
        entry["production"] = version
        self._save()
        return True

    def load(self, name: str, version: int | None = None) -> Any:
        entry = self.index.get(name)
        if not entry:
            raise FileNotFoundError(f"model {name} not registered - run scripts/train_all.py")
        v = version or entry["production"] or len(entry["versions"])
        return joblib.load(self.root / name / f"v{v}" / "model.joblib")

    def info(self) -> dict:
        return {k: {"production": v["production"], "latest": len(v["versions"]),
                    "metrics": v["versions"][(v["production"] or len(v["versions"])) - 1]["metrics"]}
                for k, v in self.index.items()}
