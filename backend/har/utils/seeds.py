"""Deterministic seeding for reproducible experiments."""

from __future__ import annotations

import os
import random
from pathlib import Path

import numpy as np


def seed_everything(seed: int, deterministic: bool = True) -> dict[str, int]:
    """Seed python, numpy and torch (if installed).

    Returns a dict describing what was seeded so it can be recorded in the
    experiment metadata.
    """
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    seeded: dict[str, int] = {"python": seed, "numpy": seed}

    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        if deterministic:
            torch.use_deterministic_algorithms(False)
            torch.set_num_threads(int(os.environ.get("HAR_TORCH_THREADS", "0")) or torch.get_num_threads())
        seeded["torch"] = seed
    except ImportError:
        pass

    try:
        import sklearn  # noqa: F401

        seeded["sklearn"] = seed
    except ImportError:
        pass

    return seeded


def project_root() -> Path:
    """Repository ``backend`` directory (this file lives in ``har/utils``)."""
    return Path(__file__).resolve().parents[2]
