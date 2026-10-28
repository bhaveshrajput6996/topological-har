"""Scientific evaluation: metrics, per-window records, latency percentile,
and the strict rule that *confidence is never reported as accuracy*."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)


@dataclass
class WindowPrediction:
    """One evaluated window; the atomic unit of the evaluation history."""

    window_id: int
    subject_id: str
    true_label: str | None
    true_label_idx: int | None
    predicted_label: str
    predicted_label_idx: int
    confidence: float
    probabilities: list[float]
    correct: bool | None
    inference_time_ms: float
    tda_ms: float = 0.0
    deep_ms: float = 0.0
    fusion_ms: float = 0.0
    ram_mb: float | None = None

    def as_dict(self) -> dict:
        return {
            "window_id": self.window_id,
            "subject_id": self.subject_id,
            "true_label": self.true_label,
            "predicted_label": self.predicted_label,
            "confidence": round(self.confidence, 6),
            "probabilities": [round(p, 6) for p in self.probabilities],
            "correct": self.correct,
            "inference_time_ms": round(self.inference_time_ms, 3),
            "tda_ms": round(self.tda_ms, 3),
            "deep_ms": round(self.deep_ms, 3),
            "fusion_ms": round(self.fusion_ms, 3),
        }


@dataclass
class EvaluationMetrics:
    accuracy: float
    precision_macro: float
    recall_macro: float
    f1_macro: float
    precision_weighted: float
    recall_weighted: float
    f1_weighted: float
    per_class: dict[str, dict[str, float]] = field(default_factory=dict)
    num_windows: int = 0
    inference_latency: dict[str, float] = field(default_factory=dict)
    branch_latency: dict[str, float] = field(default_factory=dict)
    training_time_s: float | None = None

    def as_dict(self) -> dict:
        return {
            "accuracy": round(self.accuracy, 6),
            "precision_macro": round(self.precision_macro, 6),
            "recall_macro": round(self.recall_macro, 6),
            "f1_macro": round(self.f1_macro, 6),
            "precision_weighted": round(self.precision_weighted, 6),
            "recall_weighted": round(self.recall_weighted, 6),
            "f1_weighted": round(self.f1_weighted, 6),
            "per_class": self.per_class,
            "num_windows": self.num_windows,
            "inference_latency_ms": self.inference_latency,
            "branch_latency_ms": self.branch_latency,
            "training_time_s": self.training_time_s,
        }


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    label_names: list[str],
    *,
    inference_ms: np.ndarray | None = None,
    branch_ms: dict[str, np.ndarray] | None = None,
    training_time_s: float | None = None,
) -> EvaluationMetrics:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    metrics = EvaluationMetrics(
        accuracy=float(accuracy_score(y_true, y_pred)),
        precision_macro=float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        recall_macro=float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        f1_macro=float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        precision_weighted=float(precision_score(y_true, y_pred, average="weighted", zero_division=0)),
        recall_weighted=float(recall_score(y_true, y_pred, average="weighted", zero_division=0)),
        f1_weighted=float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        num_windows=len(y_pred),
        training_time_s=training_time_s,
    )
    cm_labels = sorted(set(np.unique(y_true).tolist()) | set(np.unique(y_pred).tolist()))
    per_class: dict[str, dict[str, float]] = {}
    p_c = precision_score(
        y_true, y_pred, labels=cm_labels, average=None, zero_division=0
    )
    r_c = recall_score(y_true, y_pred, labels=cm_labels, average=None, zero_division=0)
    f_c = f1_score(y_true, y_pred, labels=cm_labels, average=None, zero_division=0)
    for i, label_idx in enumerate(cm_labels):
        name = label_names[label_idx] if label_idx < len(label_names) else str(label_idx)
        per_class[name] = {
            "precision": round(float(p_c[i]), 6),
            "recall": round(float(r_c[i]), 6),
            "f1": round(float(f_c[i]), 6),
            "support": int((y_true == label_idx).sum()),
        }
    metrics.per_class = per_class
    if inference_ms is not None and len(inference_ms):
        metrics.inference_latency = {
            "mean_ms": round(float(np.mean(inference_ms)), 3),
            "median_ms": round(float(np.median(inference_ms)), 3),
            "p95_ms": round(float(np.percentile(inference_ms, 95)), 3),
            "p99_ms": round(float(np.percentile(inference_ms, 99)), 3),
            "max_ms": round(float(np.max(inference_ms)), 3),
        }
    if branch_ms:
        metrics.branch_latency = {
            name: round(float(np.mean(values)), 3) for name, values in branch_ms.items()
        }
    return metrics


def confusion_matrix_counts(y_true, y_pred, label_names):
    labels = list(range(len(label_names)))
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    rows = []
    for true_idx in labels:
        for pred_idx in labels:
            if cm[true_idx, pred_idx] != 0:
                rows.append(
                    {
                        "true_label": label_names[true_idx],
                        "predicted_label": label_names[pred_idx],
                        "count": int(cm[true_idx, pred_idx]),
                    }
                )
    return rows
