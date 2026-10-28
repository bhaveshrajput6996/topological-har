"""Splitting tests: folds disjoint, cover all subjects, no leakage."""

import numpy as np
import pytest

from har.splitting import SplitError, SubjectSplit, make_subject_folds
from har.utils.seeds import seed_everything


def _subjects(n: int):
    return [f"subject_{i:02d}" for i in range(1, n + 1)]


def test_folds_cover_all_subjects_exactly_once():
    seed_everything(7)
    subjects = _subjects(20)
    labels = np.repeat(np.arange(4), 5)
    folds = make_subject_folds(subjects, labels, n_splits=5, val_splits=2, folds_to_run=None, seed=7)
    test_union = sorted({s for f in folds for s in f.test_subjects})
    assert test_union == subjects
    for f in folds:
        f.assert_disjoint()  # raises on overlap


def test_val_carve_is_subtrain_disjoint():
    seed_everything(11)
    subjects = _subjects(16)
    labels = np.repeat(np.arange(4), 4)
    folds = make_subject_folds(subjects, labels, n_splits=4, val_splits=2, folds_to_run=None, seed=11)
    for f in folds:
        assert not (set(f.train_subjects) & set(f.val_subjects))
        assert not (set(f.train_subjects) & set(f.test_subjects))
        assert not (set(f.val_subjects) & set(f.test_subjects))
        assert not (set(f.val_subjects) & set(f.train_subjects))
        if f.val_subjects:
            assert set(f.val_subjects) <= set(subjects)


def test_single_val_split_skips_carve():
    seed_everything(3)
    subjects = _subjects(8)
    labels = np.repeat(np.arange(4), 2)
    folds = make_subject_folds(subjects, labels, n_splits=2, val_splits=1, folds_to_run=None, seed=3)
    for f in folds:
        assert f.val_subjects == []


def test_reproducibility_with_seed():
    subjects = _subjects(18)
    labels = np.array([i % 4 for i in range(len(subjects))])
    folds_a = make_subject_folds(subjects, labels, n_splits=3, val_splits=2, folds_to_run=None, seed=99)
    folds_b = make_subject_folds(subjects, labels, n_splits=3, val_splits=2, folds_to_run=None, seed=99)
    assert [f.test_subjects for f in folds_a] == [f.test_subjects for f in folds_b]


def test_leakage_detected_by_assert_disjoint():
    s = SubjectSplit(fold=0, train_subjects=["a"], val_subjects=[], test_subjects=["a"])
    with pytest.raises(SplitError):
        s.assert_disjoint()
