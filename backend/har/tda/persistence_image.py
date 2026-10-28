"""Persistence images (Adams et al., 2017).

Each diagram point (b, d) is mapped to (b, p = d - b) and normalised by the
cloud diameter (``scale_reference``) into the unit square [0,1]^2.  The
image is the sum over features of ``weight * Gaussian_kernels`` evaluated
on a fixed grid, with weight = persistence (default) or 1.  The result is
deterministic and depends only on the diagram + configuration; it is a
dimension-agnostic weighted sum over H0 + H1 by default (this keeps the
20x20 = 400-d embedding requested by the project plan).
"""

from __future__ import annotations

import numpy as np

from ..config.schema import PersistenceImageConfig


class PersistenceImageError(ValueError):
    pass


def grid_centers(resolution: list[int]) -> np.ndarray:
    """Cell-centre coordinates of the [0,1]^2 grid (rows x cols)."""
    rows, cols = resolution
    xs = (np.arange(cols) + 0.5) / cols
    ys = (np.arange(rows) + 0.5) / rows
    gy, gx = np.meshgrid(ys, xs, indexing="ij")
    return np.stack([gx.ravel(), gy.ravel()], axis=1)  # (rows*cols, 2)


def compute_persistence_image(
    intervals: dict[int, list[list[float]]],
    cfg: PersistenceImageConfig,
    scale_reference: float,
) -> np.ndarray:
    """Compute the flattened persistence image vector.

    Parameters
    ----------
    intervals
        dimension -> list of finite [birth, death] pairs
        (zero-persistence features already removed upstream).
    scale_reference
        finite positive number; the coordinates are divided by it before
        rasterisation so every window is represented on a comparable axis.
    """
    rows, cols = cfg.resolution
    if scale_reference <= 0:
        raise PersistenceImageError("scale_reference must be positive")
    centers = grid_centers([rows, cols])

    births: list[np.ndarray] = []
    deaths: list[np.ndarray] = []
    for dim, pairs in intervals.items():
        if not pairs:
            continue
        arr = np.asarray(pairs, dtype=np.float64)
        births.append(arr[:, 0])
        deaths.append(arr[:, 1])
    if not births:
        return np.zeros(rows * cols, dtype=np.float64)

    birth = np.concatenate(births)
    death = np.concatenate(deaths)
    persistence = death - birth
    ok = persistence > 0
    birth, persistence = birth[ok], persistence[ok]
    if birth.size == 0:
        return np.zeros(rows * cols, dtype=np.float64)

    b_n = np.clip(birth / scale_reference, 0.0, 1.0)
    p_n = np.clip(persistence / scale_reference, 0.0, 1.0)

    sigma = cfg.bandwidth
    weight = persistence if cfg.weight == "persistence" else np.ones_like(b_n)
    if cfg.max_features > 0 and weight.size > cfg.max_features:
        # keep only the most persistent features (deterministic top-K);
        # less persistent features hold negligible image energy and dominate
        # the computation otherwise (documented ablation knob)
        top = np.argsort(weight)[::-1][: cfg.max_features]
        b_n, p_n, weight = b_n[top], p_n[top], weight[top]

    diff = centers[None, :, :] - np.stack([b_n, p_n], axis=1)[:, None, :]
    sq = (diff ** 2).sum(axis=2)
    kernel = np.exp(-sq / (2.0 * sigma ** 2))
    image = (weight[:, None] * kernel).sum(axis=0)
    return image.reshape(rows, cols).ravel().astype(np.float64)
