"""Dataset + model service layer for the API (cached registries)."""

from __future__ import annotations

import threading
from typing import Any

from har.config.schema import Config, DatasetConfig, config_from_dict
from har.datasets import get_adapter
from har.models.loader import LoadedRun, load_run
from har.preprocessing.pipeline import Preprocessor
from har.utils.logging import get_logger
from har.utils.seeds import project_root

logger = get_logger(__name__)


class ServiceError(RuntimeError):
    pass


class DatasetService:
    """Lazy registry of loaded datasets + their preprocessed series."""

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._datasets: dict[str, Any] = {}

    def list_adapters(self) -> list[str]:
        from har.datasets import ADAPTERS

        return sorted(ADAPTERS)

    def load(self, dc: DatasetConfig, cache: bool = True) -> Any:
        key = self._key(dc)
        if cache:
            with self._lock:
                cached = self._datasets.get(key)
                if cached is not None:
                    return cached
        adapter = get_adapter(dc.name, dc)
        dataset = adapter.load()
        if cache:
            with self._lock:
                self._datasets[key] = (adapter, dataset)
        return adapter, dataset

    @staticmethod
    def _key(dc: DatasetConfig) -> str:
        from dataclasses import asdict
        import hashlib
        import json

        payload = {key: value for key, value in asdict(dc).items()}
        return (
            dc.name
            + ":"
            + hashlib.sha1(json.dumps(payload, sort_keys=True, default=str).encode()).hexdigest()[:12]
        )

    def dataset_info(self, dc: DatasetConfig) -> dict[str, Any]:
        adapter, dataset = self.load(dc)
        return {
            "name": dataset.name,
            "kind": dataset.kind,
            "sampling_rate": dataset.sampling_rate,
            "channel_names": dataset.channel_names,
            "label_names": dataset.label_names,
            "subjects": dataset.subjects,
            "source_path": dataset.source_path,
            "notes": dataset.notes,
        }

    def dataset_info_from_name(self, name: str) -> dict[str, Any]:
        return self.dataset_info(DatasetConfig(name=name))


class ModelService:
    """Registry run_id -> LoadedRun (loaded lazily, kept in memory)."""

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._runs: dict[str, LoadedRun] = {}

    def get(self, run_id: str) -> LoadedRun:
        with self._lock:
            cached = self._runs.get(run_id)
        if cached is not None:
            return cached
        run = load_run(project_root() / "artifacts", run_id)
        with self._lock:
            self._runs[run_id] = run
        return run

    def available_run_ids(self) -> list[str]:
        from har.db import ExperimentStore

        store = ExperimentStore(project_root() / "artifacts" / "har_experiments.db")
        return [r.run_id for r in store.summary_rows()]


_dataset_service = DatasetService()
_model_service = ModelService()


def dataset_service() -> DatasetService:
    return _dataset_service


def model_service() -> ModelService:
    return _model_service
