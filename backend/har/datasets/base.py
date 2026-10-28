"""Common dataset interface.

Every adapter normalises a benchmark dataset into :class:`LoadedDataset`,
which is a list of :class:`SubjectSeries` objects plus metadata.

Two internal representations exist:

* ``continuous`` -- one array ``(n_samples, n_channels)`` per subject
  (timestamped or index-implied timestamps) that is later cut into sliding
  windows.
* ``presegmented`` -- the dataset itself ships ready-cut windows of shape
  ``(n_windows, window_samples, n_channels)`` (this is exactly what the
  official UCI HAR distribution provides: 2.56 s @ 50 Hz rows).  Windows
  cannot be re-cut in this mode; they are validated / cropped instead, and
  the overlap is defined by the dataset, not by us.
* ``feature_rows`` -- pre-engineered feature vectors (561 columns for UCI
  HAR).  Kept strictly separate; consumed only by the SVM baseline and
  clearly rejected with a loud error by window/TDA models.
"""

from __future__ import annotations

import abc
from dataclasses import dataclass, field
from typing import Any, Iterator

import numpy as np


@dataclass
class SubjectSeries:
    """All measurements of one subject, normalised by the adapter.

    Continuous series: ``values`` has shape ``(n_samples, n_channels)``.
    Presegmented series: ``values`` has shape ``(n_windows, T, C)``.
    Feature rows: ``values`` has shape ``(n_rows, n_features)``.
    """

    subject_id: str
    values: np.ndarray
    labels: np.ndarray | None = None
    sampling_rate: float = 0.0
    channel_names: list[str] = field(default_factory=list)
    kind: str = "continuous"
    window_shape: tuple[int, int] | None = None
    timestamps: np.ndarray | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def n_samples(self) -> int:
        return int(self.values.shape[0])

    @property
    def n_channels(self) -> int:
        if self.kind == "presegmented":
            return int(self.values.shape[2])
        return int(self.values.shape[1])


@dataclass
class LoadedDataset:
    name: str
    sampling_rate: float
    channel_names: list[str]
    label_names: list[str]
    kind: str
    series: list[SubjectSeries]
    source_path: str
    window_shape: tuple[int, int] | None = None
    extra: dict[str, Any] = field(default_factory=dict)
    notes: str = ""

    @property
    def subjects(self) -> list[str]:
        return [s.subject_id for s in self.series]

    def n_windows_estimate(self) -> int:
        total = 0
        for s in self.series:
            if self.kind == "presegmented":
                total += s.n_samples
            elif self.kind == "continuous":
                pass  # not estimated; windows depend on config
        return total


class DatasetAdapter(abc.ABC):
    """Abstract dataset adapter."""

    name: str = "abstract"

    @abc.abstractmethod
    def load(self) -> LoadedDataset:
        """Load the dataset into the common representation."""

    @abc.abstractmethod
    def get_subjects(self) -> list[str]:
        """Subject ids available in the dataset."""

    @abc.abstractmethod
    def get_sampling_rate(self) -> float:
        """Sampling rate in Hz."""

    @abc.abstractmethod
    def get_sensor_columns(self) -> list[str]:
        """Sensor channel names (after configured selection)."""

    @abc.abstractmethod
    def get_labels(self) -> list[str]:
        """Activity label names indexed by label id."""

    def iter_subject_data(self) -> Iterator[SubjectSeries]:
        for s in self.load().series:
            yield s


ADAPTERS: dict[str, type[DatasetAdapter]] = {}


def register_adapter(cls: type[DatasetAdapter]) -> type[DatasetAdapter]:
    ADAPTERS[cls.name] = cls
    return cls


def get_adapter(name: str, cfg: "DatasetConfig | None" = None) -> DatasetAdapter:
    if name not in ADAPTERS:
        raise KeyError(
            f"unknown dataset adapter {name!r}; available: {sorted(ADAPTERS)}"
        )
    cls = ADAPTERS[name]
    import inspect

    sig = inspect.signature(cls.__init__)
    requires_cfg = any(
        p.name == "cfg" and p.default is inspect.Parameter.empty
        for p in sig.parameters.values()
    )
    if requires_cfg:
        if cfg is None:
            from ..config.schema import DatasetConfig

            cfg = DatasetConfig(name=name)
        return cls(cfg)
    return cls()
