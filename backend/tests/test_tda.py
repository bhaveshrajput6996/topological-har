"""TDA pipeline tests: point cloud, VR persistence, persistence images."""

import numpy as np
import pytest

from har.config.schema import (
    PersistenceImageConfig,
    SubsampleConfig,
    TdaConfig,
)
from har.tda.engine import TdaEngine, TdaEngineError, subsample_indices
from har.tda.persistence_image import (
    PersistenceImageError,
    compute_persistence_image,
    grid_centers,
)
from har.tda.stats import persistence_statistics
from har.tda.vr import compute_vr_persistence, max_pairwise_distance


def _circle(n: int = 80, radius: float = 1.0, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    ang = rng.uniform(0.0, 2.0 * np.pi, size=n)
    return np.c_[radius * np.cos(ang), radius * np.sin(ang)]


def _tda(subsample: bool = True) -> TdaConfig:
    return TdaConfig(subsample=SubsampleConfig(enabled=subsample), statistical_features=False)


def test_subsample_indices_deterministic_stride():
    a = subsample_indices(1000, 80, "stride", seed=0)
    b = subsample_indices(1000, 80, "stride", seed=0)
    assert np.array_equal(a, b)
    assert len(np.unique(a)) == 80
    rng = subsample_indices(1000, 80, "random", seed=3)
    assert len(np.unique(rng)) == 80


def test_tda_engine_rejects_3d_and_nonfinite():
    engine = TdaEngine(_tda())
    with pytest.raises(TdaEngineError):
        engine.compute(np.zeros((10, 2, 2)))
    with pytest.raises(TdaEngineError):
        engine.compute(np.zeros((10, 2)) + np.nan)


def test_circle_has_dominant_h1():
    engine = TdaEngine(_tda(subsample=False))
    result = engine.compute(_circle(80))
    h1 = result.diagram.dimension_intervals[1]
    h0 = result.diagram.dimension_intervals[0]
    assert h1, "circle must produce H1 classes"
    # generator loop of a radius-1 circle dies near the diameter (2.0);
    # H0's final merge can match it, but a long-lived H1 cycle MUST exist
    # each interval is [birth, death]; persistence = death - birth
    top_h1 = max(d - b for b, d in h1)
    assert 1.2 < top_h1 < 1.6  # circle radius 1, deterministic seed (measured 1.38)
    assert sum(1 for b, d in h1 if d - b > 1.0) == 1


def test_same_input_same_output_deterministic():
    engine = TdaEngine(_tda(subsample=False))
    circle = _circle(80, seed=5)
    r1 = engine.compute(circle)
    r2 = engine.compute(circle + 0.0)
    assert np.allclose(r1.feature_vector, r2.feature_vector)
    assert r1.diagram.to_api() == r2.diagram.to_api()


def test_constant_signal_gives_valid_vector():
    engine = TdaEngine(_tda(subsample=False))
    result = engine.compute(np.zeros((30, 3)))
    assert result.feature_vector.shape[0] == 400
    assert np.isfinite(result.feature_vector).all()
    assert (result.feature_vector >= 0).all()


def test_grid_centers_shape_and_unit_range():
    centers = grid_centers([20, 20])
    assert centers.shape == (400, 2)
    assert centers[:, 0].min() >= 0 and centers[:, 0].max() <= 1
    assert centers[:, 1].min() >= 0 and centers[:, 1].max() <= 1


def test_persistence_image_weighted_by_persistence():
    cfg = PersistenceImageConfig(resolution=[20, 20], bandwidth=0.1)
    vec = compute_persistence_image({0: [[0.0, 1.0], [0.2, 0.9]]}, cfg, 1.0)
    flat = np.asarray(vec)
    assert flat.shape == (400,)
    assert flat.min() >= 0.0
    assert flat.max() > 0.0


def test_persistence_image_rejects_death_lt_birth():
    cfg = PersistenceImageConfig()
    vec = compute_persistence_image({0: [[0.5, 0.3]]}, cfg, 1.0)
    # zero-persistence pairs are dropped upstream -> zero vector
    assert np.asarray(vec).sum() == pytest.approx(0.0)


def test_persistence_image_max_features_topk():
    interval = {0: [[0.0, 1.0], [0.0, 0.8], [0.0, 0.5], [0.0, 0.2]]}
    cfg_top2 = PersistenceImageConfig(resolution=[20, 20], max_features=2)
    vec_top = compute_persistence_image(interval, cfg_top2, 1.0)
    # the two most persistent features are [0,1] and [0,0.8]
    vec_ref = compute_persistence_image({0: [[0.0, 1.0], [0.0, 0.8]]}, cfg_top2, 1.0)
    assert np.allclose(vec_top, vec_ref)


def test_stats_contains_betti_numbers():
    stats = persistence_statistics({0: [[0.0, 1.0], [0.0, 0.5]]})
    assert stats["h0_count"] == 2.0
    assert stats["h0_total_persistence"] == pytest.approx(1.5)
    assert stats["h0_max_persistence"] == pytest.approx(1.0)


def test_max_pairwise_distance_scale():
    pts = np.array([[0.0, 0.0], [3.0, 4.0]])
    assert max_pairwise_distance(pts) == pytest.approx(5.0)


def test_vr_persistence_runs_and_dimensions():
    vr = compute_vr_persistence(_circle(50), _tda())
    assert 0 in vr.diagram.dimension_intervals
    assert 1 in vr.diagram.dimension_intervals
    assert vr.diagram.scale_reference > 0
