"""Sliding windows with streaming generation.

Window/step sizes are derived from the *actual* sampling rate; nothing is
hardcoded::

    window_size = round(window_duration * sampling_rate)
    step        = max(1, round(window_size * (1 - overlap)))

Two dataset categories are supported (see ``datasets.base``):

* ``continuous``    -- true sliding windows over each subject's series.
                       Windows never cross subject boundaries.  Windows
                       containing invalid samples (missing data outside the
                       interpolated gap budget) are dropped.
* ``presegmented``  -- the dataset already ships fixed windows (official
  UCI HAR rows are 2.56s/128-samples @ 50 Hz).  Re-cutting is impossible
  (the algorithm's parameters only allow cropping to a *shorter* duration
  to be recorded and then... documented); overlap is dataset-defined.
* ``feature_rows``  -- one row == one sample vector; not windowed.

Windows are generated lazily; only training/evaluation materialises them
into float32 arrays (memory-bounded by ``max_windows``).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterator

import numpy as np

from ..config.schema import Config, WindowConfig
from ..datasets.base import LoadedDataset, SubjectSeries
from ..preprocessing.pipeline import ProcessedSeries


class WindowingError(ValueError):
    pass


@dataclass
class WindowParams:
    window_size: int
    step: int
    overlap: float | None
    window_duration: float
    sampling_rate: float
    label_strategy: str = "mode"
    crop_from_presegmented: bool = False
    notes: str = ""

    @classmethod
    def from_config(cls, cfg: Config, sampling_rate: float) -> "WindowParams":
        win: WindowConfig = cfg.window["default"]
        window_size = int(round(win.duration_seconds * sampling_rate))
        if window_size < 8:
            raise WindowingError(
                f"window_duration={win.duration_seconds}s covers only {window_size} "
                f"samples at {sampling_rate} Hz; increase it"
            )
        if win.overlap is None:
            # dataset-defined (presegmented rows); a nominal step keeps the
            # continuous path usable while overlap is reported truthfully
            step = max(1, int(round(window_size * 0.5)))
        else:
            step = int(round(window_size * (1.0 - win.overlap)))
        if step < 1:
            raise WindowingError(
                f"overlap={win.overlap} yields a non-positive step {step}"
            )
        step = max(1, step)
        return cls(
            window_size=window_size,
            step=step,
            overlap=win.overlap,
            window_duration=win.duration_seconds,
            sampling_rate=sampling_rate,
            label_strategy=win.label_strategy,
        )

    def as_dict(self) -> dict[str, object]:
        return {
            "window_size": self.window_size,
            "step": self.step,
            "overlap": self.overlap,
            "window_duration": self.window_duration,
            "sampling_rate": self.sampling_rate,
            "label_strategy": self.label_strategy,
            "overlap_note": (
                "None = dataset-defined (presegmented rows); nominal step kept for the continuous path"
                if self.overlap is None
                else ""
            ),
        }


@dataclass
class WindowArrays:
    """Materialised windows for one subject subset (train/val/test)."""

    data: np.ndarray = field(default_factory=lambda: np.empty(0))  # (N, T, C) float32
    labels: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=np.int64))
    subject_ids: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=object))
    window_ids: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=np.int64))
    timestamps: np.ndarray | None = None
    meta: dict[str, object] = field(default_factory=dict)

    def __len__(self) -> int:
        return int(self.data.shape[0])

    @property
    def n_channels(self) -> int:
        return int(self.data.shape[2]) if self.data.ndim == 3 else 0

    @property
    def window_size(self) -> int:
        return int(self.data.shape[1]) if self.data.ndim == 3 else 0


@dataclass
class PerWinRecord:
    """Streaming window record (before materialisation)."""

    data: np.ndarray  # (T, C)
    label: int
    subject_id: str
    global_index: int
    timestamp: float
    valid: bool


def label_from_window(labels: np.ndarray, strategy: str) -> int:
    labels = np.atleast_1d(np.asarray(labels))
    if labels.size == 1:
        return int(labels[0])
    if strategy == "center":
        return int(labels[labels.size // 2])
    values, counts = np.unique(labels, return_counts=True)
    return int(values[np.argmax(counts)])


def iter_windows_from_continuous_series(
    series: SubjectSeries,
    invalid_mask: np.ndarray | None,
    params: WindowParams,
    *,
    start_index: int = 0,
) -> Iterator[tuple[np.ndarray, int, int, float]]:
    """Sliding windows over one continuous subject series.

    Yields ``(window (T,C), label, window_index, timestamp)``.  Windows with
    invalid samples are skipped (missing data is never fabricated).
    """
    values = np.asarray(series.values)
    labels = series.labels if series.labels is not None else np.zeros(len(values), dtype=np.int64)
    n = len(values)
    window = params.window_size
    step = params.step
    if n < window:
        return
    index = start_index
    t0 = float(series.timestamps[0]) if series.timestamps is not None else 0.0
    for start in range(0, n - window + 1, step):
        if invalid_mask is not None and invalid_mask[start : start + window].any():
            continue
        chunk = values[start : start + window]
        label = label_from_window(labels[start : start + window], params.label_strategy)
        timestamp = t0 + start / params.sampling_rate
        yield chunk, label, index, timestamp
        index += 1


def iter_windows_from_presegmented_series(
    processed: ProcessedSeries,
    params: WindowParams,
    *,
    start_index: int = 0,
) -> Iterator[tuple[np.ndarray, int, int, float]]:
    """Yield dataset-provided windows, optionally cropped to a shorter
    window duration (the top slice of each window).  The dataset's own
    overlap is preserved and recorded; re-cutting with custom overlap is
    impossible honestly and is refused."""
    series = processed.series
    values = np.asarray(series.values)
    if values.ndim != 3:
        raise WindowingError("presegmented series must be (n_windows, T, C)")
    if series.labels is None:
        raise WindowingError("presegmented series without labels")
    win_size = params.window_size
    t = values.shape[1]
    n_channels = values.shape[2]
    expected_t = params.window_size
    invalid_mask = processed.invalid_mask
    index = start_index
    for i in range(values.shape[0]):
        if invalid_mask is not None and i < len(invalid_mask) and invalid_mask[i]:
            continue
        chunk: np.ndarray
        if t == expected_t:
            chunk = values[i]
        elif t > expected_t:
            # crop: honest ablation = take the last `expected_t` samples
            chunk = values[i, t - expected_t : t, :]
        else:
            raise WindowingError(
                f"presegmented windows are {t} samples but the requested window "
                f"size is {expected_t}; the dataset does not provide longer raw "
                "streams, so this window length cannot be measured on this mode. "
                "Reduce window.duration_seconds below the dataset window."
            )
        label = label_from_window(series.labels[i], params.label_strategy)
        timestamp = (series.timestamps[i] if series.timestamps is not None else float(i))
        yield chunk, label, index, float(timestamp)
        index += 1


def _iter_feature_rows(
    series: SubjectSeries,
) -> Iterator[tuple[np.ndarray, int, int, float]]:
    """One "window" per engineered-feature row (SVM path only)."""
    values = np.asarray(series.values)
    labels = np.asarray(series.labels)
    if labels.shape[0] != values.shape[0]:
        raise WindowingError("feature rows: label count mismatch")
    t0 = float(series.timestamps[0]) if series.timestamps is not None else 0.0
    for i in range(values.shape[0]):
        yield values[i : i + 1], int(labels[i]), i, t0 + float(i) / max(
            1e-9, series.sampling_rate
        )


def windows_stream(
    processed_series: list[ProcessedSeries],
    params: WindowParams,
    subjects: list[str] | None = None,
) -> Iterator[tuple[np.ndarray, int, str, int, float]]:
    """Stream windows across subjects.

    Yields ``(window (T,C), label, subject_id, window_index, timestamp)``.
    ``window_index`` is a per-subject running index (stable and easy to
    inspect; the evaluation log references it together with the subject).
    """
    wanted = set(subjects) if subjects else None
    for processed in processed_series:
        sid = processed.series.subject_id
        if wanted is not None and sid not in wanted:
            continue
        if processed.series.kind == "presegmented":
            iterator = iter_windows_from_presegmented_series(processed, params)
        elif processed.series.kind == "continuous":
            iterator = iter_windows_from_continuous_series(
                processed.series, processed.invalid_mask, params
            )
        elif processed.series.kind == "feature_rows":
            iterator = _iter_feature_rows(processed.series)
        else:
            raise WindowingError(
                f"subject {sid}: unsupported series kind {processed.series.kind!r}; "
                "engineered features must flow through the SVM path only"
            )
        for chunk, label, w_index, timestamp in iterator:
            yield chunk, label, sid, w_index, timestamp


def build_window_arrays(
    processed_series: list[ProcessedSeries],
    params: WindowParams,
    subjects: list[str] | None = None,
    max_windows: int | None = None,
) -> WindowArrays:
    """Materialise windows into float32 arrays (bounded by ``max_windows``)."""
    windows: list[np.ndarray] = []
    labels: list[int] = []
    subjects_out: list[str] = []
    window_ids: list[int] = []
    timestamps_out: list[float] = []

    for chunk, label, sid, w_index, timestamp in windows_stream(processed_series, params, subjects):
        windows.append(chunk)
        labels.append(label)
        subjects_out.append(sid)
        window_ids.append(w_index)
        timestamps_out.append(timestamp)
        if max_windows is not None and len(windows) >= max_windows:
            break

    if not windows:
        raise WindowingError("no windows were generated; check window parameters/subjects")

    data = np.stack(windows).astype(np.float32)
    arrays = WindowArrays(
        data=data,
        labels=np.asarray(labels, dtype=np.int64),
        subject_ids=np.asarray(subjects_out, dtype=object),
        window_ids=np.asarray(window_ids, dtype=np.int64),
        timestamps=np.asarray(timestamps_out, dtype=np.float64),
        meta={
            "params": params.as_dict(),
            "n_windows": len(data),
            "channels": processed_series[0].series.channel_names,
        },
    )
    return arrays


def describe_window_params(params: WindowParams) -> str:
    if params.overlap is None:
        return (
            f"window={params.window_size} samples ({params.window_duration:.2f}s @ "
            f"{params.sampling_rate:.0f}Hz); rows are dataset-presegmented, "
            "overlap is dataset-defined"
        )
    return (
        f"window={params.window_size} samples ({params.window_duration:.2f}s @ "
        f"{params.sampling_rate:.0f}Hz), step={params.step}, overlap={params.overlap:.0%}"
    )
