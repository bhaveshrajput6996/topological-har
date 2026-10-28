"""Optional topological statistics (H0/H1 feature counts, total persistence,
persistence entropy, ...).  Useful for research analysis and debugging;
only fed to the classifier when ``tda.statistical_features`` is enabled."""

from __future__ import annotations

import numpy as np


def persistence_statistics(diagram_intervals: dict[int, list[list[float]]]) -> dict[str, float]:
    stats: dict[str, float] = {}
    for dim, intervals in sorted(diagram_intervals.items()):
        if not intervals:
            stats[f"h{dim}_count"] = 0.0
            stats[f"h{dim}_total_persistence"] = 0.0
            stats[f"h{dim}_max_persistence"] = 0.0
            stats[f"h{dim}_mean_persistence"] = 0.0
            stats[f"h{dim}_entropy"] = 0.0
            continue
        arr = np.asarray(intervals, dtype=np.float64)
        pers = arr[:, 1] - arr[:, 0]
        pers = pers[pers > 0]
        total = float(pers.sum())
        stats[f"h{dim}_count"] = float(len(pers))
        stats[f"h{dim}_total_persistence"] = total
        stats[f"h{dim}_max_persistence"] = float(pers.max()) if len(pers) else 0.0
        stats[f"h{dim}_mean_persistence"] = float(pers.mean()) if len(pers) else 0.0
        if total > 0:
            probs = pers / total
            entropy = -float((probs * np.log(probs + 1e-12)).sum())
        else:
            entropy = 0.0
        stats[f"h{dim}_entropy"] = entropy
    return stats
