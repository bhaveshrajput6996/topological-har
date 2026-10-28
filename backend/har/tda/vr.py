"""Vietoris-Rips persistent homology computation via GUDHI.

Verified against the installed gudhi API (3.13), in particular:

* ``gudhi.RipsComplex(points=..., max_edge_length=...)`` and
  ``create_simplex_tree(max_dimension=1)`` for the 1-skeleton,
* ``SimplexTree.collapse_edges()`` + ``expansion(max_dim + 1)`` for the exact,
  lossless edge-collapsed flag complex (fast path; H1 needs the
  2-skeleton, i.e. triangles filled in),
* ``persistence(persistence_dim_max=True, homology_coeff_field=2,
  min_persistence=...)`` -- field with 2 elements = Z/2Z,
* ``persistence_intervals_in_dimension(dim)`` for H0/H1 intervals.

H0 = connected components, H1 = loops/cycles.  Essential (infinite
death-time) features are clipped to ``scale_reference`` (the maximum edge
length considered, approximated by the point cloud diameter), and this is
recorded, not silently hidden.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np

from ..config.schema import TdaConfig
from .diagram import PersistenceDiagram, make_diagram


class TdaError(RuntimeError):
    pass


@dataclass
class VrResult:
    diagram: PersistenceDiagram
    rips_ms: float
    n_simplices: int
    d_max: float


def max_pairwise_distance(points: np.ndarray) -> float:
    """Diameter of the (small) point cloud (n <= ~150 here, so O(n^2) is fine)."""
    n = len(points)
    if n < 2:
        return 0.0
    diff = points[:, None, :] - points[None, :, :]
    return float(np.max(np.linalg.norm(diff, axis=2)))


def compute_vr_persistence(
    points: np.ndarray,
    cfg: TdaConfig,
) -> VrResult:
    """Compute VR persistence (H0/H1) of ``points`` with Euclidean metric."""
    import gudhi  # local import: keeps torch-free paths importable when absent

    if points.ndim != 2:
        raise TdaError("points must be (n, d)")
    n = points.shape[0]
    max_dim = max(cfg.homology_dimensions)
    if max_dim > 1:
        raise TdaError("only H0 and H1 are supported by this engine")

    t_start = time.perf_counter()
    max_edge = max_pairwise_distance(points)
    if n <= max_dim:  # degenerate clouds give empty diagrams
        empty_diag = make_diagram(
            {d: [] for d in cfg.homology_dimensions},
            scale_reference=max_edge if max_edge > 0 else 1.0,
        )
        return VrResult(diagram=empty_diag, rips_ms=0.0, n_simplices=0, d_max=max_edge)

    rips = gudhi.RipsComplex(points=points, max_edge_length=float("inf"))
    st = rips.create_simplex_tree(max_dimension=1)
    if cfg.use_edge_collapse and n >= 10:
        # To compute homology up to dimension k, gudhi requires the
        # (k+1)-skeleton: expansion(2) fills in triangles for H1.
        st.collapse_edges()
        st.expansion(max_dim + 1)
    else:
        st = rips.create_simplex_tree(max_dimension=max_dim + 1)
    n_simplices = st.num_simplices()
    st.persistence(
        persistence_dim_max=True,
        homology_coeff_field=2,  # Z/2Z coefficients
        min_persistence=cfg.min_persistence,
    )
    intervals: dict[int, list[list[float]]] = {}
    for dim in cfg.homology_dimensions:
        iv = st.persistence_intervals_in_dimension(dim)
        iv = [] if iv is None or len(iv) == 0 else np.asarray(iv, dtype=np.float64).tolist()
        intervals[dim] = iv
    rips_ms = (time.perf_counter() - t_start) * 1000.0

    diagram = make_diagram(
        {d: intervals[d] for d in cfg.homology_dimensions},
        scale_reference=max_edge if max_edge > 0 else 1.0,
    )
    return VrResult(diagram=diagram, rips_ms=rips_ms, n_simplices=int(n_simplices), d_max=max_edge)
