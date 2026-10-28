"""Dataset adapter tests (synthetic always, UCI when present)."""

import numpy as np
import pytest
from pathlib import Path

from har.config.schema import DatasetConfig
from har.datasets import get_adapter


def test_registry_lists_adapters():
    from har.datasets.base import ADAPTERS

    for expected in ("uci_har", "pamap2", "opportunity", "synthetic"):
        assert expected in ADAPTERS


def test_unknown_adapter_rejected():
    with pytest.raises(KeyError):
        get_adapter("definitely_not_a_dataset")


def test_synthetic_load_shapes():
    ds = get_adapter("synthetic", DatasetConfig(name="synthetic")).load()
    assert len(ds.series) == 2
    assert ds.sampling_rate == 32.0
    for s in ds.series:
        v = np.asarray(s.values)
        assert v.ndim == 2 and v.shape[1] == 3
        assert v.dtype == np.float32
        assert s.labels is not None and len(s.labels) == len(v)


def test_ucihar_load_real_data():
    root = Path(__file__).resolve().parents[1] / "data" / "raw" / "UCI HAR Dataset"
    if not (root / "train" / "X_train.txt").exists():
        pytest.skip("UCI HAR dataset not downloaded")
    ds = get_adapter("uci_har", DatasetConfig(name="uci_har")).load()
    assert len(ds.series) == 30
    assert len(ds.subjects) == 30
    assert ds.label_names == [
        "WALKING",
        "WALKING_UPSTAIRS",
        "WALKING_DOWNSTAIRS",
        "SITTING",
        "STANDING",
        "LAYING",
    ]
    first = ds.series[0]
    assert first.kind == "presegmented"
    assert np.asarray(first.values).shape[1] == 128
    assert np.asarray(first.values).shape[2] == 9
    total = sum(len(s.values) for s in ds.series)
    assert total == 10299


def test_ucihar_subject_filter():
    root = Path(__file__).resolve().parents[1] / "data" / "raw" / "UCI HAR Dataset"
    if not (root / "train" / "X_train.txt").exists():
        pytest.skip("UCI HAR dataset not downloaded")
    ds = get_adapter(
        "uci_har", DatasetConfig(name="uci_har", subjects=["1", "30"])
    ).load()
    assert ds.subjects == ["subject_01", "subject_30"]
    total = sum(len(s.values) for s in ds.series)
    assert 0 < total < 10299
