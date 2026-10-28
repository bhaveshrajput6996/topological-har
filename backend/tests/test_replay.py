"""Replay source tests (library level; API integration in test_prediction_api)."""

import numpy as np
import pytest

from har.config.schema import Config
from har.datasets import get_adapter
from har.preprocessing.pipeline import Preprocessor
from har.replay.replay import ReplaySource, dataset_into_source
from har.windowing.windowing import WindowParams


def _synthetic_source(max_windows=6):
    cfg = Config()
    ds = get_adapter("synthetic", cfg.dataset).load()
    sr = float(cfg.dataset.sampling_rate_override or ds.sampling_rate)
    params = WindowParams.from_config(cfg, sr)
    pre = Preprocessor(cfg, sr).process_dataset(ds)
    return cfg, ds, dataset_into_source(ds, pre, params, max_windows=max_windows)


def test_source_yields_windows_in_order():
    _cfg, _ds, src = _synthetic_source()
    assert len(src) > 0
    it = iter(src)
    first = next(it)
    second = next(it)
    assert first.window.shape[1] == 3
    assert first.true_label is not None
    assert second.index == first.index + 1
    assert first.subject_id == "synthetic_s00"


def test_events_carry_labels_and_subjects():
    _cfg, _ds, src = _synthetic_source()
    events = list(src)
    assert all(e.true_label in src.label_names for e in events)
    assert len(events) == len(src)


def test_label_name_out_of_range_is_none():
    _cfg, _ds, src = _synthetic_source()
    assert src.label_name(999) is None
    assert src.label_name(0) == src.label_names[0]
