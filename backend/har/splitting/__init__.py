"""Subject-aware splitting with enforced leakage checks.

The research rule: **all windows of one subject stay in the same fold.**
``sklearn.model_selection.StratifiedGroupKFold`` is used with

* ``groups`` = subject id,
* stratification ``y`` = window activity label (majority within each
  subject/group is approximated by sklearn).

An optional *validation* subset is carved out of each training fold by a
second StratifiedGroupKFold restricted to the training subjects, so
train/val/test are always subject-disjoint (leakage fails loudly).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from sklearn.model_selection import StratifiedGroupKFold


class SplitError(RuntimeError):
    """Raised when a split strategy cannot be executed or leaks subjects."""


@dataclass
class SubjectSplit:
    fold: int
    train_subjects: list[str]
    val_subjects: list[str]
    test_subjects: list[str]
    n_windows: dict[str, int] = field(default_factory=dict)

    def assert_disjoint(self) -> None:
        train = set(self.train_subjects)
        val = set(self.val_subjects)
        test = set(self.test_subjects)
        for name, a, b in (
            ("train/val", train, val),
            ("train/test", train, test),
            ("val/test", val, test),
        ):
            overlap = a & b
            if overlap:
                raise SplitError(
                    f"subject leakage detected between {name}: {sorted(overlap)}. "
                    "Refusing to train on a leaking split."
                )

    def as_dict(self) -> dict[str, object]:
        return {
            "fold": self.fold,
            "train_subjects": self.train_subjects,
            "val_subjects": self.val_subjects,
            "test_subjects": self.test_subjects,
        }


def make_subject_folds(
    window_subjects: np.ndarray,
    window_labels: np.ndarray,
    n_splits: int,
    val_splits: int,
    folds_to_run: list[int] | None = None,
    seed: int = 42,
) -> list[SubjectSplit]:
    """Create subject-disjoint folds via StratifiedGroupKFold.

    Parameters
    ----------
    window_subjects
        Subject id (object array) for every window.
    window_labels
        Activity label for every window.
    n_splits
        Outer CV folds (test hat).
    val_splits
        Inner splits used to carve one validation hat out of the training hat.
    folds_to_run
        Restrict to these outer fold numbers (0-based).  ``None`` = all.
    """
    if len(window_subjects) != len(window_labels):
        raise SplitError("window_subjects/window_labels length mismatch")

    subjects = np.asarray(window_subjects, dtype=object)
    labels = np.asarray(window_labels)
    groups, group_index = np.unique(subjects, return_inverse=True)

    if n_splits > len(groups):
        raise SplitError(
            f"evaluation.cv_folds={n_splits} exceeds the number of subjects "
            f"({len(groups)}); every fold would have an empty training set"
        )

    gkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    folds: list[SubjectSplit] = []
    for fold, (train_idx, test_idx) in enumerate(gkf.split(np.zeros(len(labels)), labels, groups=group_index)):
        if folds_to_run is not None and fold not in folds_to_run:
            continue
        test_subjects = [str(groups[i]) for i in np.unique(group_index[test_idx])]
        train_subjects = [str(groups[i]) for i in np.unique(group_index[train_idx])]

        val_subjects = choose_val_subjects(
            subjects[train_idx],
            labels[train_idx],
            train_subjects,
            val_splits,
            seed,
        ) if val_splits > 1 else []
        if val_subjects:
            val_set = set(val_subjects)
            train_subjects = [s for s in train_subjects if s not in val_set]

        split = SubjectSplit(
            fold=fold,
            train_subjects=train_subjects,
            val_subjects=val_subjects,
            test_subjects=test_subjects,
        )
        split.assert_disjoint()
        if not split.train_subjects:
            raise SplitError(
                f"fold {fold}: training subjects became empty after carving the "
                "validation set; use more subjects or training.val_splits=1"
            )
        folds.append(split)
    if not folds:
        raise SplitError("no folds were produced (check evaluation.folds)")
    return folds


def choose_val_subjects(
    subjects: np.ndarray,
    labels: np.ndarray,
    train_subjects: list[str],
    val_splits: int,
    seed: int,
) -> list[str]:
    """Pick validation subjects from the training subjects (stratified-approx)."""
    groups, group_index = np.unique(np.asarray(subjects, dtype=object), return_inverse=True)
    gkf = StratifiedGroupKFold(n_splits=val_splits, shuffle=True, random_state=seed + 1000)
    try:
        train_idx, val_idx = next(iter(gkf.split(np.zeros(len(labels)), labels, groups=group_index)))
    except ValueError:
        # a single class or too few groups: fall back to a subject-level round split
        shuffled = sorted(range(len(groups)))
        rng = np.random.default_rng(seed)
        rng.shuffle(shuffled)
        k_val = max(1, len(groups) // val_splits)
        val_groups = {groups[i] for i in shuffled[:k_val]}
        return sorted(str(g) for g in val_groups if str(g) in train_subjects)

    val_groups = {groups[i] for i in np.unique(group_index[val_idx])}
    val_subjects = [str(g) for g in val_groups]
    if not val_subjects:
        raise SplitError("could not carve a validation subject set")
    return sorted(val_subjects)


def leakage_report(splits: list[SubjectSplit]) -> dict[str, object]:
    """Return a machine-readable report of subject disjointness."""
    for split in splits:
        split.assert_disjoint()
    return {
        "n_folds": len(splits),
        "subject_sets_disjoint": True,
        "folds": [s.as_dict() for s in splits],
    }
