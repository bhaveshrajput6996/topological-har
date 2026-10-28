"""TDA branch orchestration: point cloud -> subsample -> VR -> diagram ->
persistence image -> (optional) statistics, with latency measurement of
every stage. The full-resolution window stays untouched for the deep
branch; subsampling affects only the TDA branch."""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np

from ..config.schema import Config, TdaConfig
from .diagram import PersistenceDiagram
from .persistence_image import compute_persistence_image
from .stats import persistence_statistics
from .vr import compute_vr_persistence
from ..utils.logging import log_structured, get_logger

logger = get_logger(__name__)


class TdaEngineError(RuntimeError):
    pass


@dataclass
class TdaResult:
    feature_vector: np.ndarray
    """Flattened persistence image (+ optional statistics)."""
    point_cloud: np.ndarray
    """Subsampled point cloud actually used by VR (n, C)."""
    point_indices: np.ndarray
    """Indices of the point-cloud selection inside the window."""
    diagram: PersistenceDiagram
    statistics: dict[str, float] = field(default_factory=dict)
    timings_ms: dict[str, float] = field(default_factory=dict)
    d_max: float = 0.0
    n_simplices: int = 0

    def feature_dict(self) -> dict[str, float]:
        return {
            "feature_dim": int(self.feature_vector.size),
            "n_points": int(self.point_cloud.shape[0]),
            "n_h0": self.diagram.betti_numbers().get(0, 0),
            "n_h1": self.diagram.betti_numbers().get(1, 0),
            "d_max": round(self.d_max, 6),
            "timings_ms": {k: round(v, 3) for k, v in self.timings_ms.items()},
            "n_simplices": int(self.n_simplices),
        }


def subsample_indices(n_total: int, max_points: int, method: str, seed: int = 0) -> np.ndarray:
    if n_total <= max_points:
        return np.arange(n_total, dtype=np.int64)
    if method == "stride":
        idx = np.linspace(0, n_total - 1, max_points)
        return np.unique(idx.astype(np.int64))
    if method == "random":
        rng = np.random.default_rng(seed)
        return np.sort(rng.choice(n_total, size=max_points, replace=False).astype(np.int64))
    raise TdaEngineError(f"unknown subsample method {method!r}")


class TdaEngine:
    """Stateless per-window TDA computation (thread-safe)."""

    def __init__(self, cfg: TdaConfig) -> None:
        self.cfg = cfg
        if not cfg.homology_dimensions:
            raise TdaEngineError("tda.homology_dimensions must not be empty")
        self._seed = 0

    @classmethod
    def from_config(cls, cfg: Config) -> "TdaEngine":
        return cls(cfg.tda)

    def compute(self, window: np.ndarray) -> TdaResult:
        window = np.asarray(window)
        if window.ndim != 2:
            raise TdaEngineError(
                f"window must be (T, C); got shape {window.shape}"
            )
        if not np.isfinite(window).all():
            raise TdaEngineError("window contains non-finite values; check preprocessing")

        t0 = time.perf_counter()
        if self.cfg.subsample.enabled and window.shape[0] > self.cfg.subsample.max_points:
            idx = subsample_indices(
                window.shape[0],
                self.cfg.subsample.max_points,
                self.cfg.subsample.method,
                self._seed,
            )
        else:
            idx = np.arange(window.shape[0], dtype=np.int64)
        point_cloud = window[idx].astype(np.float64)
        t_sub = (time.perf_counter() - t0) * 1000.0

        vr = compute_vr_persistence(point_cloud, self.cfg)

        t1 = time.perf_counter()
        image = compute_persistence_image(
            vr.diagram.dimension_intervals,
            self.cfg.persistence_image,
            vr.diagram.scale_reference,
        )
        t_img = (time.perf_counter() - t1) * 1000.0

        stats: dict[str, float] = {}
        if self.cfg.statistical_features:
            stats = persistence_statistics(vr.diagram.dimension_intervals)

        parts = [image]
        if stats:
            parts.append(np.array([stats[k] for k in sorted(stats)], dtype=np.float64))
        feature_vector = np.concatenate(parts) if parts else np.zeros(0)

        return TdaResult(
            feature_vector=feature_vector.astype(np.float32),
            point_cloud=point_cloud,
            point_indices=idx,
            diagram=vr.diagram,
            statistics=stats,
            timings_ms={
                "subsample_ms": t_sub,
                "rips_ms": vr.rips_ms,
                "image_ms": t_img,
            },
            d_max=vr.d_max,
            n_simplices=vr.n_simplices,
        )
