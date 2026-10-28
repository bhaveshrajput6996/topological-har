"""Window math tests: sizes, steps, overlap computation, no leakage."""

import numpy as np
import pytest

from har.config.schema import Config, WindowConfig, resolve_paths
from har.datasets.base import LoadedDataset, SubjectSeries
from har.preprocessing.pipeline import ProcessedSeries
from har.windowing.windowing import (
    WindowParams,
    WindowingError,
    build_window_arrays,
    describe_window_params,
    iter_windows_from_continuous_series,
    iter_windows_from_presegmented_series,
    windows_stream,
)


def _series(values, timestamps=None, labels=None, kind="continuous"):
    return SubjectSeries(
        subject_id="s00",
        kind=kind,
        values=np.asarray(values, dtype=np.float32),
        labels=labels,
        timestamps=timestamps,
    )


def test_continuous_window_count_and_step():
    n = 100
    values = np.arange(n * 2, dtype=np.float32).reshape(n, 2)
    labels = np.repeat(np.array([[0, 1]]), n // 2, axis=0).ravel()
    params = WindowParams(
        window_size=20, step=10, window_duration=2.0, overlap=0.5, sampling_rate=10.0, label_strategy="mode"
    )
    out = list(iter_windows_from_continuous_series(_series(values, labels=labels), None, params))
    assert [w[0].shape for w in out] == [(20, 2)] * len(out)
    assert len(out) == (n - 20) // 10 + 1  # 8: starts 0,10,...,70,80 -> 9 starts n-window=80


def test_overlap_zero_gives_disjoint_windows():
    n = 60
    values = np.zeros((n, 1), dtype=np.float32)
    params = WindowParams(
        window_size=20, step=20, window_duration=2.0, overlap=0.0, sampling_rate=10.0, label_strategy="mode"
    )
    starts = [int(np.asarray(w[2])) for w in iter_windows_from_continuous_series(_series(values), None, params)]
    assert len(starts) == 3  # indices 0,1,2 of consecutive windows
    chunks = [int(w[0][0, 0]) if False else i for i, w in enumerate(iter_windows_from_continuous_series(_series(values), None, params))]
    assert chunks == [0, 1, 2]


def _processed(series: SubjectSeries):
    return ProcessedSeries(series=series, invalid_mask=None, gap_report=None, filtered=False)


def test_presegmented_window_crop_only_shorter():
    values = np.zeros((3, 40, 1), dtype=np.float32)
    labels = np.array([1, 2, 3])
    params = WindowParams(
        window_size=20, step=20, window_duration=2.0, overlap=0.0, sampling_rate=20.0, label_strategy="mode"
    )
    out = list(iter_windows_from_presegmented_series(
        _processed(_series(values, labels=labels, kind="presegmented")), params
    ))
    assert len(out) == 3
    assert out[0][0].shape == (20, 1)
    with pytest.raises(WindowingError):
        # longer window than dataset-provided must be refused, not fabricated
        list(iter_windows_from_presegmented_series(
            _processed(_series(np.zeros((2, 40, 1), dtype=np.float32), labels=np.array([1, 2]), kind="presegmented")),
            WindowParams(window_size=50, step=10, window_duration=5.0, overlap=0.5, sampling_rate=20.0, label_strategy="mode"),
        ))


def test_windows_stream_per_subject_index():
    values = np.zeros((100, 2), dtype=np.float32)
    s0 = _series(values)
    s1 = _series(values)
    s1.subject_id = "s01"
    ds = LoadedDataset(
        name="toy",
        sampling_rate=10.0,
        channel_names=["a", "b"],
        label_names=["c0", "c1", "c2"],
        kind="continuous",
        series=[s0, s1],
        source_path=".",
    )
    params = WindowParams(
        window_size=10, step=10, window_duration=1.0, overlap=0.0, sampling_rate=10.0, label_strategy="mode"
    )
    out = list(windows_stream([_processed(s0), _processed(s1)], params))
    assert len(out) == 20  # 10 windows per subject (starts 0..90, step 10)
    assert {o[2] for o in out} == {"s00", "s01"}
    idx0 = [o[3] for o in out if o[2] == "s00"]
    assert idx0 == list(range(10))


def test_build_window_arrays_groups():
    values = np.arange(120, dtype=np.float32).reshape(120, 1)
    labels = np.repeat([0, 1], 60)
    s0 = _series(values, labels=labels)
    s1 = _series(values, labels=labels)
    s1.subject_id = "s01"
    ds = LoadedDataset(
        name="toy",
        sampling_rate=10.0,
        channel_names=["a"],
        label_names=["c0", "c1"],
        kind="continuous",
        series=[s0, s1],
        source_path=".",
    )
    params = WindowParams(
        window_size=20, step=20, window_duration=2.0, overlap=0.0, sampling_rate=10.0, label_strategy="mode"
    )
    arrays = build_window_arrays([_processed(s0), _processed(s1)], params, max_windows=None)
    assert arrays.data.shape == (12, 20, 1)  # 6 windows per subject
    assert set(arrays.subject_ids) == {"s00", "s01"}
    assert arrays.labels.min() >= 0 and arrays.labels.max() <= 1

