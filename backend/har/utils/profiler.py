"""Cross-platform resource measurement.

What is actually measured (documented for the report):

* ``RSS`` -- resident set size of *the whole Python process* measured with
  ``psutil`` (Windows/Linux/macOS).  This includes interpreter + libraries
  overhead, not only processed tensors.  It is therefore an upper bound of
  "used memory".
* ``tracemalloc`` -- optional Python-level allocation tracking (fine
  grained, but does not see inside native code, so it is *not* used as the
  primary process-level metric).

The :class:`ResourceMonitor` samples RSS in a daemon thread at 50 ms and
records per-phase peak/average values.  For GPU workloads
``torch.cuda.max_memory_allocated`` is recorded when CUDA is available.
"""

from __future__ import annotations

import contextlib
import threading
import time
from dataclasses import dataclass, field
from typing import Any

import psutil

from ..utils.logging import get_logger

logger = get_logger(__name__)

_SAMPLE_INTERVAL_S = 0.05


@dataclass
class PhaseMeasurement:
    peak_rss_mb: float
    avg_rss_mb: float
    start_rss_mb: float


class ResourceMonitor:
    """Background RSS sampler with explicit phase reporting.

    Usage::

        with ResourceMonitor() as monitor:
            with monitor.phase("train"):
                ...
        monitor.report()  # {"train": {"peak_rss_mb": .., "avg_rss_mb": ..}}
    """

    def __init__(self, sample_interval: float = _SAMPLE_INTERVAL_S) -> None:
        self._proc = psutil.Process()
        self._interval = sample_interval
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._lock = threading.Lock()
        self._current_phase: str | None = None
        self._samples: dict[str, list[float]] = {}
        self._phases: dict[str, dict[str, Any]] = {}
        self._base_rss_mb: float = 0.0

    # -- lifecycle ---------------------------------------------------------
    def __enter__(self) -> "ResourceMonitor":
        self.start()
        return self

    def __exit__(self, *exc: Any) -> None:
        self.stop()

    def start(self) -> None:
        self.base_rss_mb = self._rss_mb()
        self._thread = threading.Thread(target=self._run, daemon=True, name="rss-sampler")
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)
        self._thread = None

    # -- phases ------------------------------------------------------------
    @contextlib.contextmanager
    def phase(self, name: str):
        """Context manager reporting one phase."""
        self.begin_phase(name)
        try:
            yield self
        finally:
            self.end_phase()

    def begin_phase(self, name: str) -> None:
        with self._lock:
            if self._current_phase is not None:
                self.end_phase()
            self._current_phase = name

    def end_phase(self) -> None:
        if self._current_phase is None:
            return
        with self._lock:
            name, self._current_phase = self._current_phase, None
        samples = self._samples.pop(name, [])
        if samples:
            start = samples[0]
            self._phases[name] = {
                "start_rss_mb": round(start, 2),
                "peak_rss_mb": round(max(samples), 2),
                "avg_rss_mb": round(sum(samples) / len(samples), 2),
            }

    def report(self) -> dict[str, Any]:
        """Finalise the current phase and return all phase measurements."""
        self.end_phase()
        out: dict[str, Any] = dict(self._phases)
        out["process_baseline_rss_mb"] = round(self.base_rss_mb, 2)
        gpu = self._gpu_report()
        if gpu:
            out["gpu"] = gpu
        return out

    # -- internals ---------------------------------------------------------
    @property
    def base_rss_mb(self) -> float:
        return self._base_rss_mb

    @base_rss_mb.setter
    def base_rss_mb(self, value: float) -> None:
        self._base_rss_mb = value

    def _rss_mb(self) -> float:
        try:
            return self._proc.memory_info().rss / (1024 * 1024)
        except (psutil.Error, OSError):
            return 0.0

    def _run(self) -> None:
        while not self._stop.wait(self._interval):
            value = self._rss_mb()
            with self._lock:
                if self._current_phase is None:
                    continue
                self._samples.setdefault(self._current_phase, []).append(value)

    def _gpu_report(self) -> dict[str, Any]:
        try:
            import torch
        except ImportError:
            return {}
        if not torch.cuda.is_available():
            return {}
        return {
            "available": True,
            "reserved_mb": round(torch.cuda.memory_reserved(0) / (1024 * 1024), 2),
            "max_allocated_mb": round(torch.cuda.max_memory_allocated(0) / (1024 * 1024), 2),
        }

    def snapshot(self) -> PhaseMeasurement:
        value = self._rss_mb()
        return PhaseMeasurement(peak_rss_mb=value, avg_rss_mb=value, start_rss_mb=value)


@dataclass
class LatencyBreakdown:
    tda_ms: float = 0.0
    deep_ms: float = 0.0
    fusion_ms: float = 0.0

    @property
    def total_ms(self) -> float:
        return self.tda_ms + self.deep_ms + self.fusion_ms

    def as_dict(self) -> dict[str, float]:
        return {
            "tda_ms": round(self.tda_ms, 3),
            "deep_ms": round(self.deep_ms, 3),
            "fusion_ms": round(self.fusion_ms, 3),
            "total_ms": round(self.total_ms, 3),
        }


@dataclass
class Stopwatch:
    """Simple cumulative stopwatch used for latency breakdowns."""

    segments: dict[str, float] = field(default_factory=dict)

    def add(self, name: str, seconds: float) -> None:
        self.segments[name] = self.segments.get(name, 0.0) + seconds

    def get(self, name: str) -> float:
        return self.segments.get(name, 0.0)


class Timer:
    """Context manager measuring wall time of a block (seconds)."""

    def __init__(self) -> None:
        self.elapsed: float = 0.0
        self._start: float | None = None

    def __enter__(self) -> "Timer":
        self._start = time.perf_counter()
        return self

    def __exit__(self, *exc: Any) -> None:
        if self._start is not None:
            self.elapsed = time.perf_counter() - self._start
