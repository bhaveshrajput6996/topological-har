"""Evaluation suite: per-window prediction records, batched metrics,
latency distribution, confusion matrix.  Used by the experiment runner and
the API."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch

from ..models.svm import SklearnSvm
from ..inference.predictor import WindowPredictor
from ..tda.engine import TdaEngine
from .metrics import WindowPrediction, compute_metrics, confusion_matrix_counts


@dataclass
class FoldEvaluation:
    metrics: Any  # EvaluationMetrics
    window_records: list[dict[str, Any]]
    confusion_rows: list[dict[str, Any]]
    y_true: np.ndarray
    y_pred: np.ndarray
    latencies: dict[str, float]
    tda_time_s: float = 0.0

    def summary_dict(self) -> dict[str, Any]:
        return {
            "metrics": self.metrics.as_dict(),
            "latency": self.latencies,
            "num_windows": int(len(self.y_pred)),
            "tda_time_s": round(self.tda_time_s, 3),
        }


def evaluate_windows(
    model: Any,
    data_test: np.ndarray,
    labels_test: np.ndarray,
    subjects_test: np.ndarray,
    window_ids_test: np.ndarray,
    timestamps_test: np.ndarray | None,
    label_names: list[str],
    tda_engine: TdaEngine | None,
    window_scaler,
    tda_scaler,
    train_time_s: float | None = None,
    warmup: int = 5,
) -> FoldEvaluation:
    """Per-window (batch=1) evaluation with honest latency measurement.

    The windows passed here must NOT be scaled yet (the predictor applies
    the training-fitted scaler exactly like during training).  TDA features
    are computed fresh per window so TDA latency is measured, too.
    """
    n = len(data_test)
    if n == 0:
        raise ValueError("evaluation set is empty")

    dummy_window = data_test[0]
    if isinstance(model, torch.nn.Module):
        predictor = WindowPredictor(
            model=model,
            tda_engine=tda_engine,
            window_scaler=window_scaler,
            tda_scaler=tda_scaler,
            warmup=warmup,
        )
        predictor.warm_up(dummy_window)

    records: list[WindowPrediction] = []
    probs_all: list[np.ndarray] = []
    t_total = time.perf_counter()
    tda_total_s = 0.0

    if isinstance(model, SklearnSvm):
        for i in range(n):
            window = data_test[i : i + 1]
            t0 = time.perf_counter()
            probs, pred = model.predict_proba(window)
            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            probs_i = probs[0]
            pred_i = int(pred[0])
            records.append(
                _make_record(
                    i, labels_test[i], subjects_test[i], window_ids_test[i],
                    label_names, pred_i, probs_i, elapsed_ms,
                    tda_ms=0.0, deep_ms=0.0, fusion_ms=elapsed_ms,
                    timestamps_test=timestamps_test,
                )
            )
            probs_all.append(probs_i)
    else:
        fuse = tda_engine is not None
        for i in range(n):
            window = data_test[i]
            t0 = time.perf_counter()
            if fuse:
                # TDA with the preprocessing-scaled window coordinate system
                t_start = time.perf_counter()
                scaled = window
                if window_scaler is not None and window_scaler.enabled:
                    scaled = window_scaler.transform(window[None, ...])[0]
                tda_result = tda_engine.compute(np.asarray(scaled, dtype=np.float32))
                t_features = tda_result.feature_vector[None, :].astype(np.float32)
                if tda_scaler is not None:
                    t_features = tda_scaler.transform(t_features)
                tda_ms = (time.perf_counter() - t_start) * 1000.0
                tda_total_s += (time.perf_counter() - t_start)
                with torch.no_grad():
                    x = torch.from_numpy(scaled[None, ...]).float()
                    t_feat = torch.from_numpy(t_features).float()
                    t_deep0 = time.perf_counter()
                    deep = model.encoder.forward_features(x)
                    t_deep = (time.perf_counter() - t_deep0) * 1000.0
                    t_fus0 = time.perf_counter()
                    logits = model.head(torch.cat([deep, t_feat], dim=1))
                    t_fusion = (time.perf_counter() - t_fus0) * 1000.0
                    probs_i = torch.softmax(logits, dim=1)[0].numpy()
                inference_ms = (time.perf_counter() - t0) * 1000.0
                records.append(
                    _make_record(
                        i, labels_test[i], subjects_test[i], window_ids_test[i],
                        label_names, int(probs_i.argmax()), probs_i, inference_ms,
                        tda_ms=tda_ms, deep_ms=t_deep, fusion_ms=t_fusion,
                        timestamps_test=timestamps_test,
                    )
                )
                probs_all.append(probs_i)
            else:
                predictor_result = predictor.predict_window(window)
                probs_i = predictor_result.probabilities
                records.append(
                    _make_record(
                        i, labels_test[i], subjects_test[i], window_ids_test[i],
                        label_names, predictor_result.predicted_idx, probs_i,
                        predictor_result.total_ms,
                        tda_ms=predictor_result.tda_ms,
                        deep_ms=predictor_result.deep_ms,
                        fusion_ms=predictor_result.fusion_ms,
                        timestamps_test=timestamps_test,
                    )
                )
                probs_all.append(probs_i)

    total_eval_s = time.perf_counter() - t_total
    y_true = np.asarray(labels_test)
    probs_matrix = np.vstack(probs_all)
    y_pred = probs_matrix.argmax(axis=1)

    inference_ms_array = np.array([r.inference_time_ms for r in records], dtype=np.float64)
    branch = {
        "tda_ms": np.array([r.tda_ms for r in records], dtype=np.float64),
        "deep_ms": np.array([r.deep_ms for r in records], dtype=np.float64),
        "fusion_ms": np.array([r.fusion_ms for r in records], dtype=np.float64),
    }
    metrics = compute_metrics(
        y_true,
        y_pred,
        label_names,
        inference_ms=inference_ms_array,
        branch_ms=branch,
        training_time_s=train_time_s,
    )
    confusion_rows = confusion_matrix_counts(y_true, y_pred, label_names)

    latency = {
        "mean_ms": metrics.inference_latency.get("mean_ms", 0.0),
        "p95_ms": metrics.inference_latency.get("p95_ms", 0.0),
        "median_ms": metrics.inference_latency.get("median_ms", 0.0),
        "p99_ms": metrics.inference_latency.get("p99_ms", 0.0),
        "max_ms": metrics.inference_latency.get("max_ms", 0.0),
        "wall_eval_s_per_window": round(total_eval_s / max(n, 1), 4),
    }

    return FoldEvaluation(
        metrics=metrics,
        window_records=[r.as_dict() for r in records],
        confusion_rows=confusion_rows,
        y_true=y_true,
        y_pred=y_pred,
        latencies=latency,
        tda_time_s=tda_total_s,
    )


def _make_record(
    i: int,
    label_true: int,
    subject: str,
    window_id: int,
    label_names: list[str],
    pred_idx: int,
    probs_i: np.ndarray,
    inference_ms: float,
    tda_ms: float,
    deep_ms: float,
    fusion_ms: float,
    timestamps_test: np.ndarray | None,
) -> WindowPrediction:
    confidence = float(probs_i[pred_idx])
    true_optional = int(label_true) if label_true is not None else None
    correct: bool | None
    if true_optional is None:
        correct = None
    else:
        correct = bool(true_optional == pred_idx)
    timestamp = (
        float(timestamps_test[i]) if timestamps_test is not None else None
    )
    return WindowPrediction(
        window_id=int(window_id),
        subject_id=str(subject),
        true_label=label_names[true_optional] if true_optional is not None else None,
        true_label_idx=true_optional,
        predicted_label=label_names[pred_idx],
        predicted_label_idx=pred_idx,
        confidence=confidence,
        probabilities=[float(p) for p in probs_i],
        correct=correct,
        inference_time_ms=inference_ms,
        tda_ms=tda_ms,
        deep_ms=deep_ms,
        fusion_ms=fusion_ms,
    )
