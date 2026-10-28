"""Dataset replay: emits dataset windows like a simulated live sensor.

Order: dataset -> windows (same windowing/preprocessing semantics as
training) -> prediction -> events consumed by the API/WebSocket.

Replay uses no physical sensors; every consumer of a replay payload knows
(``source`` is the dataset).  Pacing is handled by the replay manager, not
this module, so tests can consume events without sleeping.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator

import numpy as np

from ..utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class ReplayEvent:
    index: int
    subject_id: str
    window_id: int
    window: np.ndarray
    timestamp: float
    true_label: str | None
    true_label_idx: int | None


class ReplaySource:
    """Fixed window sequence the replay walks through."""

    def __init__(
        self,
        windows: np.ndarray,
        labels: np.ndarray | None,
        subjects: np.ndarray,
        window_ids: np.ndarray,
        timestamps: np.ndarray | None,
        label_names: list[str],
    ) -> None:
        self.windows = windows
        self.labels = labels
        self.subjects = subjects
        self.window_ids = window_ids
        self.timestamps = timestamps
        self.label_names = label_names

    def __len__(self) -> int:
        return len(self.windows)

    def label_name(self, idx: int | None) -> str | None:
        if idx is None or idx < 0 or idx >= len(self.label_names):
            return None
        return self.label_names[idx]

    def __iter__(self) -> Iterator[ReplayEvent]:
        for i, window in enumerate(self.windows):
            label_idx = int(self.labels[i]) if self.labels is not None else None
            yield ReplayEvent(
                index=i,
                subject_id=str(self.subjects[i]),
                window_id=int(self.window_ids[i]),
                window=np.asarray(window, dtype=np.float32),
                timestamp=(
                    float(self.timestamps[i]) if self.timestamps is not None else float(i)
                ),
                true_label=self.label_name(label_idx),
                true_label_idx=label_idx,
            )


def dataset_into_source(dataset, preprocessed, params, max_windows: int | None = None) -> ReplaySource:
    """Build a replay source from a loaded dataset + configured pipeline."""
    from ..windowing.windowing import build_window_arrays

    arrays = build_window_arrays(
        preprocessed,
        params,
        max_windows=max_windows,
    )
    return ReplaySource(
        windows=arrays.data,
        labels=arrays.labels,
        subjects=arrays.subject_ids,
        window_ids=arrays.window_ids,
        timestamps=arrays.timestamps,
        label_names=dataset.label_names,
    )
