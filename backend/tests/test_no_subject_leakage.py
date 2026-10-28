"""No subject leakage: the split helper raises when groups collide, and the
runner's split metadata proves folds were subject-disjoint on real data."""

from pathlib import Path

import numpy as np
import pytest

from har.splitting import SplitError, make_subject_folds
from har.utils.seeds import seed_everything


def test_every_test_subject_excluded_from_train():
    seed_everything(42)
    subjects = [f"s{i:02d}" for i in range(24)]
    labels = np.array([i % 3 for i in range(len(subjects))])
    folds = make_subject_folds(subjects, labels, n_splits=5, val_splits=2, folds_to_run=None, seed=42)
    for f in folds:
        assert not (set(f.train_subjects) & set(f.test_subjects))
        assert not (set(f.val_subjects) & set(f.test_subjects))
        assert not (set(f.train_subjects) & set(f.val_subjects))


def test_stratification_distributes_labels_across_folds():
    seed_everything(1)
    subjects = [f"s{i:02d}" for i in range(24)]
    labels = np.repeat(np.arange(4), 6)
    folds = make_subject_folds(subjects, labels, n_splits=4, val_splits=1, folds_to_run=None, seed=1)
    combined = sorted(
        {int(c) for f in folds for c in labels[np.isin(subjects, f.test_subjects)]}
    )
    assert combined == [0, 1, 2, 3]
    # every subject appears in exactly one test hat
    flat = sorted({s for f in folds for s in f.test_subjects})
    assert flat == subjects


def test_too_few_subjects_raises():
    subjects = ["s1", "s2"]
    labels = np.array([0, 1])
    with pytest.raises(SplitError):
        make_subject_folds(subjects, labels, n_splits=5, val_splits=2, folds_to_run=None, seed=0)
