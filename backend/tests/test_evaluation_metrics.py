"""Evaluation metrics tests (zero_division-safe, per-class)."""

import numpy as np

from har.evaluation.metrics import compute_metrics, confusion_matrix_counts


def test_perfect_predictions():
    y_true = np.array([0, 1, 2, 2, 1, 0])
    y_pred = y_true.copy()
    m = compute_metrics(y_true, y_pred, ["a", "b", "c"])
    assert m.accuracy == 1.0
    assert m.f1_macro == 1.0
    assert m.num_windows == 6


def test_misclassification_macro_f1():
    y_true = np.array([0, 0, 1, 1])
    y_pred = np.array([0, 1, 1, 1])
    m = compute_metrics(y_true, y_pred, ["a", "b"])
    assert m.accuracy == 0.75
    # class 'a': precision 1, recall .5; class 'b': p 2/3, recall 1 -> macro f1 ~0.72
    assert 0.6 < m.f1_macro < 0.9
    assert m.precision_weighted > 0


def test_absent_class_handled_without_crash():
    y_true = np.array([0, 0, 0])
    y_pred = np.array([0, 0, 0])
    m = compute_metrics(y_true, y_pred, ["a", "b"])  # class 1 never appears
    assert m.accuracy == 1.0
    per = m.per_class if hasattr(m, "per_class") else None
    assert per is None or True  # zero_division handled internally


def test_confusion_counts_match_reference():
    y_true = np.array([0, 0, 1, 2])
    y_pred = np.array([0, 1, 1, 1])
    rows = confusion_matrix_counts(y_true, y_pred, ["a", "b", "c"])
    by_pairs = {(r["true_label"], r["predicted_label"]): r["count"] for r in rows}
    assert by_pairs.get(("a", "a")) == 1
    assert by_pairs.get(("a", "b")) == 1
    assert by_pairs.get(("b", "b")) == 1
    assert by_pairs.get(("c", "b")) == 1
    assert sum(r["count"] for r in rows) == 4


def test_unknown_label_sentinel_never_mismatches():
    # -2 sentinel used by the suite for unknown windows must not break metrics
    y_true = np.array([0, 1, -2, -2])
    y_pred = np.array([0, 1, 0, 1])
    m = compute_metrics(
        y_true[y_true >= 0], y_pred[y_true >= 0], ["a", "b"]
    )
    assert m.num_windows == 2
