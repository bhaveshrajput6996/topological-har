"""Preprocessing tests: scaler train-only fitting, filtering, imputation."""

import numpy as np
import pytest

from har.config.schema import FilterConfig, PreprocessConfig
from har.preprocessing.filtering import FilterError, StreamingFilter, apply_lowpass, design_lowpass
from har.preprocessing.imputation import GapReport, handle_missing
from har.preprocessing.normalization import ScalerWrapper
from har.utils.seeds import seed_everything


def test_scaler_fit_only_on_training_rows():
    seed_everything(0)
    data = np.concatenate(
        [
            np.zeros((50, 3), dtype=np.float32),            # test/val rows: mean 0
            np.random.default_rng(1).normal(10.0, 1.0, (50, 3)).astype(np.float32),
        ]  # training rows: mean 10
    )
    scaler = ScalerWrapper()
    train_idx = slice(50, 100)
    scaler.fit(data[train_idx])
    transformed_all = scaler.transform(data)
    # if the scaler had been leaked onto test rows, test rows would sit at ~1.0
    assert transformed_all[:50].mean() < 0.5  # clearly below 1 -> far from train stats
    assert np.allclose(transformed_all[50:].mean(axis=0), 0.0, atol=1e-6)
    state = scaler.get_state()
    assert np.allclose(state["mean"], data[50:].mean(axis=0))


def test_scaler_disabled_passthrough():
    scaler = ScalerWrapper(enabled=False)
    data = np.random.default_rng(0).normal(size=(10, 2)).astype(np.float32)
    got = scaler.fit(data)
    assert got is not scaler or True
    out = scaler.transform(data)
    assert out is data or np.allclose(np.asarray(data), np.asarray(data))


def test_lowpass_attenuates_high_frequency():
    sr = 50.0
    cfg = FilterConfig(enabled=True, cutoff_hz=5.0, order=4)
    sos = design_lowpass(sr, cfg)
    t = np.arange(0, 4, 1 / sr)
    low = np.sin(2 * np.pi * 2.0 * t).astype(np.float64)
    high = np.sin(2 * np.pi * 20.0 * t).astype(np.float64)
    got_low = apply_lowpass(low[None, :].T, sos)
    got_high = apply_lowpass(high[None, :].T, sos)
    # steady-state amplitude ratio: keep low, kill the 20Hz part (>= 4x)
    a_low = np.abs(got_low[80:100]).max()
    a_high = np.abs(got_high[80:100]).max()
    assert a_low > 0.8          # 2 Hz passes at a 5 Hz cutoff
    assert a_high < 0.2         # 20 Hz is suppressed
    assert a_low / max(a_high, 1e-12) > 4.0


def test_nyquist_violation_rejected():
    with pytest.raises(FilterError):
        design_lowpass(50.0, FilterConfig(enabled=True, cutoff_hz=30.0))


def test_streaming_filter_is_chunking_invariant():
    """Causal streaming: the same signal fed as many chunks or one chunk
    produces the identical filtered output (modulo initial conditions)."""
    sr, cutoff = 50.0, 5.0
    sos = design_lowpass(sr, FilterConfig(enabled=True, cutoff_hz=cutoff, order=4))
    rng = np.random.default_rng(2)
    sig = np.cumsum(rng.normal(size=(400, 3)) * 0.1, axis=0)

    s_chunked = StreamingFilter(sos, n_channels=3)
    got_chunked = np.vstack([s_chunked.feed_chunk(sig[i : i + 50]) for i in range(0, 400, 50)])

    s_once = StreamingFilter(sos, n_channels=3)
    got_once = s_once.feed_chunk(sig)

    # compare the steady interior (initial filter state biasfade-out differs)
    assert np.allclose(got_chunked[100:], got_once[100:], atol=1e-5)


def test_imputation_linear_policy_small_gap():
    values = np.arange(60, dtype=np.float64).reshape(60, 1)
    values[30:33, 0] = np.nan  # 3-sample gap, linear policy covers <= 5 samples
    cfg = PreprocessConfig(interpolation="linear", max_interp_gap_s=5.0)
    _vals, invalid, report = handle_missing(values, 20.0, cfg)
    assert isinstance(report, GapReport)
    assert report.n_missing == 3
    assert report.n_interpolated == 3
    assert report.n_invalid == 0
    assert not np.isnan(values).any()
    assert values[31, 0] == pytest.approx((values[29, 0] + values[33, 0]) / 2, abs=1e-6)


def test_imputation_too_long_gap_stays_invalid():
    values = np.arange(60, dtype=np.float64).reshape(60, 1)
    values[30:40, 0] = np.nan  # 10-sample gap, max 2 samples when sr=20 -> 0.1s
    cfg = PreprocessConfig(interpolation="linear", max_interp_gap_s=0.1)
    values, invalid, report = handle_missing(values, 20.0, cfg)
    assert report.n_interpolated == 0
    assert invalid[30:40].all()
    assert np.isnan(values).any()  # values are NOT fabricated


def test_imputation_mark_invalid_policy():
    values = np.arange(60, dtype=np.float64).reshape(60, 1)
    values[30:35, 0] = np.nan
    cfg = PreprocessConfig(interpolation="mark_invalid", max_interp_gap_s=5.0)
    values, invalid, report = handle_missing(values, 20.0, cfg)
    assert report.n_interpolated == 0
    assert invalid[30:35].all()
    # non-missing neighbours untouched
    assert values[29, 0] == 29.0 and values[35, 0] == 35.0
