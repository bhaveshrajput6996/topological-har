"""Butterworth low-pass filtering via second-order sections.

Offline paths use zero-phase ``sosfiltfilt``; a stateful ``StreamingFilter``
is provided for the future live-sensor mode (uses causal ``sosfilt``).
Both validate the cutoff against Nyquist at design time.
"""

from __future__ import annotations

import numpy as np
from scipy.signal import butter, sosfilt, sosfiltfilt

from ..config.schema import FilterConfig


class FilterError(ValueError):
    pass


def design_lowpass(sampling_rate: float, cfg: FilterConfig) -> np.ndarray:
    """Design a Butterworth low-pass SOC filter, validating Nyquist."""
    cutoff = cfg.cutoff_hz
    if cutoff is None:
        raise FilterError(
            "filter.cutoff_hz is unset; the pipeline must resolve a dataset "
            "sampling rate before designing the filter"
        )
    nyquist = sampling_rate / 2.0
    if not 0 < cutoff < nyquist:
        raise FilterError(
            f"filter.cutoff_hz={cutoff} is invalid for sampling_rate={sampling_rate} "
            f"(must satisfy 0 < cutoff < {nyquist})"
        )
    sos = butter(cfg.order, cutoff, btype="lowpass", output="sos", fs=sampling_rate)
    return sos


def apply_lowpass(values: np.ndarray, sos: np.ndarray) -> np.ndarray:
    """Zero-phase filter of ``values`` shaped ``(n_samples, n_channels)``."""
    if values.ndim != 2:
        raise FilterError("apply_lowpass expects a 2-D (n_samples, n_channels) array")
    if values.shape[0] < 2:
        return values.copy()
    return sosfiltfilt(sos, values, axis=0).astype(np.float32)


class StreamingFilter:
    """Stateful filter for (future) real-time sensor streams.

    Reused filter state between ``feed()`` calls; produces causally filtered
    samples with ``sosfilt`` (zero-phase is not possible in real time).
    """

    def __init__(self, sos: np.ndarray, n_channels: int) -> None:
        self.sos = sos
        self.zi: np.ndarray | None = None
        self._n_channels = n_channels
        self._warmed = False

    def feed_chunk(self, chunk: np.ndarray) -> np.ndarray:
        if chunk.ndim != 2:
            raise FilterError("feed_chunk expects (n_samples, n_channels)")
        if self.zi is None:
            zi = np.zeros((self.sos.shape[0], 2, self._n_channels), dtype=np.float64)
        else:
            zi = self.zi
            self._warmed = True
        out, self.zi = sosfilt(self.sos, chunk, axis=0, zi=zi)
        return out.astype(np.float32)

    def reset(self) -> None:
        self.zi = None
        self._warmed = False
