"""Shared service singletons for the FastAPI app."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from har.db import ExperimentStore
from har.utils.seeds import project_root


@lru_cache(maxsize=1)
def get_store() -> ExperimentStore:
    return ExperimentStore(project_root() / "artifacts" / "har_experiments.db")


def artifacts_root() -> Path:
    return project_root() / "artifacts"
