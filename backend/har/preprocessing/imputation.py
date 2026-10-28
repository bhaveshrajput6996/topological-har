"""Missing-data handling with explicit, documented policies.

Policies (``preprocessing.interpolation``):

* ``linear``       -- small gaps (duration <= ``max_interp_gap_s``) are
                      linearly interpolated; longer gaps are marked invalid.
* ``drop_window``  -- no interpolation; every missing sample is invalid and
                      windows containing invalid samples are dropped.
* ``mark_invalid`` -- same invalid marking, but the invalid marker is
                      forwarded (windows containing them are still dropped;
                      kept for explicit introspection/reporting).

Invalid samples are represented as a boolean mask aligned with the series.
Leading/trailing gaps cannot be interpolated and are always invalid.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..config.schema import PreprocessConfig


class PreprocessError(ValueError):
    pass


@dataclass
class GapReport:
    n_samples: int
    n_missing: int
    n_gap_runs: int
    max_gap_s: float
    n_interpolated: int
    n_invalid: int
    policy: str


def sample_missing_mask(values: np.ndarray) -> np.ndarray:
    """Boolean mask (n_samples,) True where any channel is missing."""
    if values.ndim != 2:
        raise PreprocessError("missing mask expects (n_samples, n_channels)")
    return ~np.isfinite(values).all(axis=1)


def _gap_runs(mask: np.ndarray) -> list[tuple[int, int]]:
    """Return [(start, end_exclusive, length), ...] for True runs."""
    if mask.size == 0:
        return []
    padded = np.concatenate(([False], mask, [False])).astype(np.int8)
    diff = np.diff(padded)
    starts = np.where(diff == 1)[0]
    ends = np.where(diff == -1)[0]
    return list(zip(starts.tolist(), ends.tolist()))


def handle_missing(
    values: np.ndarray,
    sampling_rate: float,
    cfg: PreprocessConfig,
) -> tuple[np.ndarray, np.ndarray, GapReport]:
    """Apply the configured policy.

    Returns ``(values, invalid_mask, report)``.  ``invalid_mask`` is True for
    samples that must not enter windows (missing, or outside interpolatable
    gaps under the ``linear`` policy).
    """
    if values.ndim != 2:
        raise PreprocessError("handle_missing expects (n_samples, n_channels)")

    missing = sample_missing_mask(values)
    n_missing = int(missing.sum())
    if n_missing == 0:
        report = GapReport(
            n_samples=len(values),
            n_missing=0,
            n_gap_runs=0,
            max_gap_s=0.0,
            n_interpolated=0,
            n_invalid=0,
            policy=cfg.interpolation,
        )
        return values.copy(), np.zeros(len(values), dtype=bool), report

    invalid = missing.copy()
    n_interpolated = 0
    max_gap_s = 0.0
    runs = _gap_runs(missing)

    if cfg.interpolation == "linear":
        max_len = int(round(cfg.max_interp_gap_s * sampling_rate))
        for start, end in runs:
            length = end - start
            max_gap_s = max(max_gap_s, length / sampling_rate)
            if length <= max_len and start > 0 and end < len(values):
                # linear interpolation between the surrounding valid samples
                x0, x1 = start - 1, end
                v0 = values[x0].astype(np.float64)
                v1 = values[x1].astype(np.float64)
                for k, i in enumerate(range(start, end)):
                    w = (k + 1) / (length + 1)
                    values[i] = (v0 * (1.0 - w) + v1 * w).astype(values.dtype)
                n_interpolated += length
                invalid[start:end] = False
            # leading/trailing or too-long gaps stay invalid

    else:
        if runs:
            max_gap_s = max(end - start for start, end in runs) / sampling_rate

    n_invalid = int(invalid.sum())
    report = GapReport(
        n_samples=len(values),
        n_missing=n_missing,
        n_gap_runs=len(runs),
        max_gap_s=round(max_gap_s, 4),
        n_interpolated=n_interpolated,
        n_invalid=n_invalid,
        policy=cfg.interpolation,
    )
    return values, invalid, report


def validate_timestamps(timestamps: np.ndarray | None, sampling_rate: float) -> dict[str, float]:
    """Sanity-check pace of timestamps; returns statistics (no mutation)."""
    if timestamps is None or len(timestamps) < 3:
        return {"available": 0.0, "median_dt_s": 0.0, "monotonic": 0.0}
    dt = np.diff(timestamps)
    monotonic = bool(np.all(dt >= -1e-6))
    return {
        "available": float(len(timestamps)),
        "median_dt_s": float(np.median(dt)) if len(dt) else 0.0,
        "median_dt_vs_sr_diff": float(abs(np.median(dt) - 1.0 / sampling_rate)) if np.median(dt) > 0 else 0.0,
        "monotonic": float(monotonic),
    }
