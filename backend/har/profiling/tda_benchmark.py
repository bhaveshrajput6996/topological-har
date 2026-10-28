"""TDA benchmarking: VR cost vs number of points (the justification for the
subsample default).  Reports points, VR time, total TDA time, H0/H1
feature counts and memory deltas; writes benchmark.json-style dicts."""

from __future__ import annotations

import time
from typing import Any

import numpy as np
import psutil

from ..config.schema import SubsampleConfig, PersistenceImageConfig, TdaConfig
from ..tda.engine import TdaEngine

PROC = psutil.Process()


def _rss_mb() -> float:
    try:
        return PROC.memory_info().rss / (1024**2)
    except psutil.Error:
        return 0.0


def benchmark_tda(
    sample_window: np.ndarray,
    points_list: list[int] | None = None,
    repetitions: int = 3,
    resolution: list[int] | None = None,
    max_features: int = 64,
) -> dict[str, Any]:
    """Benchmark the TDA branch at several point counts.

    Each run subsamples exactly to the requested point count so the
    Vietoris-Rips cost is measured honestly.
    """
    window = np.asarray(sample_window, dtype=np.float32)
    points_list = points_list or [40, 64, 80, 100]
    results: list[dict[str, Any]] = []
    for n_points in points_list:
        cfg = TdaConfig(
            subsample=SubsampleConfig(
                enabled=True, max_points=n_points, method="stride"
            ),
            persistence_image=PersistenceImageConfig(
                resolution=resolution or [20, 20], bandwidth=0.1, max_features=max_features
            ),
        )
        engine = TdaEngine(cfg)
        rss_before = _rss_mb()
        timings = []
        feature_counts: list[dict[str, float]] = []
        for _ in range(max(1, repetitions)):
            t0 = time.perf_counter()
            outcome = engine.compute(window)
            timings.append((time.perf_counter() - t0) * 1000.0)
            betti = outcome.diagram.betti_numbers()
            feature_counts.append(
                {
                    "h0_features": float(betti.get(0, 0)),
                    "h1_features": float(betti.get(1, 0)),
                    "n_simplices": float(outcome.n_simplices),
                    "rips_ms": outcome.timings_ms.get("rips_ms", 0.0),
                    "image_ms": outcome.timings_ms.get("image_ms", 0.0),
                }
            )
        rss_after = _rss_mb()
        results.append(
            {
                "window_shape": [int(x) for x in window.shape],
                "max_points": n_points,
                "avg_total_ms": round(float(np.mean(timings)), 2),
                "min_total_ms": round(float(np.min(timings)), 2),
                "max_total_ms": round(float(np.max(timings)), 2),
                "rss_before_mb": round(rss_before, 1),
                "rss_after_mb": round(rss_after, 1),
                "point_cloud_points": int(outcome.point_cloud.shape[0]),
                "feature_vector_dim": int(outcome.feature_vector.size),
                "first_run_stats": feature_counts[0],
            }
        )
    return {
        "benchmark_type": "tda_vr_points",
        "results": results,
        "config": {
            "resolution": resolution or [20, 20],
            "max_features": max_features,
            "repetitions": repetitions,
        },
    }
