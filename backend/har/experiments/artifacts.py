"""Artifact writing/loading.

Each experiment produces::

    artifacts/{run_id}/
        config.yaml            resolved configuration
        pipeline.json          window params, TDA cfg, labels, channels
        model.pt               best model state (torch) or SVM pickle
        model.scaler.pkl       window scaler (train-only fit)
        tda_scaler.pkl         scaler for TDA feature vectors (train-only fit)
        metrics.json           fold summary metrics
        confusion_matrix.json
        training_history.json
        environment.json       versions / hardware / seeds / git commit
"""

from __future__ import annotations

import json
import os as os_module
import platform
import subprocess
from pathlib import Path
from typing import Any

import numpy as np
import yaml


def write_yaml(run_dir: Path, name: str, payload: dict[str, Any]) -> None:
    with (run_dir / name).open("w", encoding="utf-8") as fh:
        yaml.safe_dump(payload, fh, sort_keys=False, allow_unicode=True)


def write_json(run_dir: Path, name: str, payload: Any) -> None:
    with (run_dir / name).open("w", encoding="utf-8") as fh:
        json.dump(_to_plain(payload), fh, indent=2, ensure_ascii=False, default=str)


def _to_plain(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {key: _to_plain(value) for key, value in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_plain(x) for x in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating, np.integer, np.bool_)):
        return obj.item()
    return obj


def write_environment_info(run_dir: Path, seed: int | None = None) -> dict[str, Any]:
    info = collect_environment_info(seed)
    write_json(run_dir, "environment.json", info)
    return info


def collect_environment_info(seed: int | None = None) -> dict[str, Any]:
    import sys

    import scipy
    import sklearn

    info: dict[str, Any] = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "machine": platform.machine(),
        "cpu_count": os_module.cpu_count(),
        "processor": platform.processor(),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "scikit_learn": sklearn.__version__,
        "seed": seed,
    }
    try:
        import torch

        info["torch"] = torch.__version__
        info["cuda_available"] = bool(torch.cuda.is_available())
        info["gpu_available"] = info["cuda_available"]
        if torch.cuda.is_available():
            info["cuda_version"] = torch.version.cuda
            info["gpu"] = torch.cuda.get_device_name(0)
    except ImportError:
        info["torch"] = None
        info["cuda_available"] = False
    try:
        import gudhi

        info["gudhi"] = gudhi.__version__
    except ImportError:
        info["gudhi"] = None
    try:
        import psutil

        vm = psutil.virtual_memory()
        info["ram_total_gb"] = round(vm.total / 1024**3, 2)
        info["ram_available_gb"] = round(vm.available / 1024**3, 2)
    except ImportError:
        pass
    commit = get_git_commit()
    if commit:
        info["git_commit"] = commit
    return info


def get_git_commit() -> str | None:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if out.returncode == 0:
            return out.stdout.strip() or None
    except (subprocess.SubprocessError, OSError, FileNotFoundError):
        return None
    return None
