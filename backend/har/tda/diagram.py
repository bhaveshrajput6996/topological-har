"""Persistence diagrams returned by persistent homology computations."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class PersistenceDiagramPoint:
    """One (birth, death) feature of a persistence diagram.

    ``persistence = death - birth``; essential features (death = ?) are
    clipped to a finite scale and recorded as essential (they are still
    exactly identifiable).
    """

    dimension: int
    birth: float
    death: float
    persistence: float
    essential: bool = False

    def as_dict(self) -> dict[str, float]:
        return {
            "dimension": self.dimension,
            "birth": self.birth,
            "death": self.death,
            "persistence": self.persistence,
        }


@dataclass
class PersistenceDiagram:
    """Full persistence diagram of one point cloud."""

    dimension_intervals: dict[int, list[float]]
    """id -> [n,2] array-like intervals ordered [birth, death] (finite)."""

    n_essential: dict[int, int]  # per-dimension count of infinite-death features
    points: list[PersistenceDiagramPoint]
    scale_reference: float  # the d_max used to clip essential pieces

    def to_api(self) -> dict[str, list[list[float]]]:
        """The JSON shape the frontend needs.

        ``{"H0": [[birth, death], ...], "H1": [[birth, death], ...]}``
        """
        out: dict[str, list[list[float]]] = {}
        for dim in sorted(self.dimension_intervals):
            intervals = self.dimension_intervals[dim]
            out[f"H{dim}"] = [[float(b), float(d)] for b, d in intervals]
        return out

    def betti_numbers(self) -> dict[int, int]:
        return {dim: len(v) for dim, v in self.dimension_intervals.items()}


def make_diagram(
    intervals_by_dim: dict[int, tuple[float, float] | list[list[float]]],
    scale_reference: float,
) -> PersistenceDiagram:
    """Assemble a :class:`PersistenceDiagram` from raw (birth, death) pairs.

    ``scale_reference`` (the maximum filtration scale considered) must be
    finite and > 0; essential (infinite-death) features are clipped to it.
    """
    if not (scale_reference > 0) or not (scale_reference < float("inf")):
        raise ValueError("scale_reference must be a positive finite number")

    finite: dict[int, list[list[float]]] = {}
    n_essential: dict[int, int] = {}
    points: list[PersistenceDiagramPoint] = []
    for dim, intervals in intervals_by_dim.items():
        finite[dim] = []
        n_essential[dim] = 0
        for birth, death in intervals:
            if death == float("inf"):
                n_essential[dim] += 1
                death_f = scale_reference
                essential = True
            else:
                death_f = float(death)
                essential = False
            birth_f = float(birth)
            persistence = death_f - birth_f
            if persistence <= 0.0:
                continue  # meaningless zero-persistence feature
            finite[dim].append([birth_f, death_f])
            points.append(
                PersistenceDiagramPoint(
                    dimension=dim,
                    birth=birth_f,
                    death=death_f,
                    persistence=persistence,
                    essential=essential,
                )
            )
    return PersistenceDiagram(
        dimension_intervals=finite,
        n_essential=n_essential,
        points=points,
        scale_reference=scale_reference,
    )
