"""Window inspection: sensors, point cloud, PCA projection, persistence
diagram/image, Betti statistics, and (optional) a saved-model prediction
with ground truth when available."""

from __future__ import annotations

import json
from typing import Any

import numpy as np

from ..config.schema import Config, DatasetConfig, config_from_dict
from ..datasets import get_adapter
from ..models.loader import load_run
from ..preprocessing.pipeline import Preprocessor
from ..tda.engine import TdaEngine, TdaEngineError
from ..tda.stats import persistence_statistics
from ..utils.logging import get_logger
from ..utils.seeds import project_root
from ..windowing.windowing import WindowParams, build_window_arrays

logger = get_logger(__name__)


class InspectionError(RuntimeError):
    pass


def representative_window(
    dataset_name: str,
    config_path: str | None,
    subject: str | None = None,
    window_index: int = 0,
) -> np.ndarray:
    """Return one preprocessed window (used by the benchmark CLI)."""
    cfg = _resolve_config(config_path, dataset_name)
    (
        dataset,
        _pre,
        params,
        window,
        label,
        sid,
        w_index,
        t,
    ) = _dataset_and_windows(cfg, subject, window_index, 1)
    return window


def _resolve_config(config_path: str | None, dataset_name: str | None) -> Config:
    if config_path:
        return config_from_dict(_load_yaml(config_path))
    cfg = Config()
    if dataset_name:
        cfg.dataset = DatasetConfig(name=dataset_name, signal_mode=None)
    return cfg


def _load_yaml(path: str) -> dict[str, Any]:
    import yaml

    with open(path, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def _dataset_and_windows(
    cfg: Config,
    subject: str | None,
    window_index: int,
    n_windows: int,
):
    adapter = get_adapter(cfg.dataset.name, cfg.dataset)
    dataset = adapter.load()
    sampling_rate = float(
        cfg.dataset.sampling_rate_override or adapter.get_sampling_rate()
    )
    params = WindowParams.from_config(cfg, sampling_rate)
    preprocessor = Preprocessor(cfg, sampling_rate)
    preprocessed = preprocessor.process_dataset(dataset)
    taken: list[tuple] = []
    from ..windowing.windowing import windows_stream, ProcessedSeries  # local import

    for window, label, sid, w_index, t in windows_stream(preprocessed, params, [subject] if subject else None):
        taken.append((window, label, sid, w_index, t))
        if len(taken) >= n_windows + window_index:
            break
    if len(taken) <= window_index:
        raise InspectionError(
            f"window_index {window_index} out of range ({len(taken)} windows available)"
        )
    window, label, sid, w_index, t = taken[window_index]
    return dataset, preprocessed, params, np.asarray(window, dtype=np.float32), label, sid, w_index, t


def inspect_dataset_window(
    dataset_name: str | None,
    subject: str | None,
    window_index: int,
    run_id: str | None,
    config_path: str | None,
) -> dict[str, Any]:
    cfg = _resolve_config(config_path, dataset_name)
    (
        dataset,
        _pre,
        params,
        window,
        label,
        sid,
        w_index,
        t,
    ) = _dataset_and_windows(cfg, subject, window_index, 1)

    tda_engine = TdaEngine(cfg.tda)
    tda_result = tda_engine.compute(window)

    # PCA is ONLY for visualisation; the actual TDA runs in sensor space
    pca_coords = _pca_2d(tda_result.point_cloud)

    payload: dict[str, Any] = {
        "dataset": cfg.dataset.name,
        "subject_id": sid,
        "window_index": int(w_index),
        "window_shape": [int(x) for x in window.shape],
        "sampling_rate": params.sampling_rate,
        "duration_s": round(window.shape[0] / params.sampling_rate, 3),
        "channel_names": list(dataset.channel_names),
        "sensor_series": {
            name: [round(float(v), 5) for v in window[:, c].tolist()]
            for c, name in enumerate(dataset.channel_names)
        },
        "ground_truth": {
            "label": dataset.label_names[label] if label is not None and 0 <= label < len(dataset.label_names) else None,
            "label_idx": int(label) if label is not None else None,
        },
        "point_cloud": {
            "n_points": int(tda_result.point_cloud.shape[0]),
            "n_dims": int(tda_result.point_cloud.shape[1]),
            "point_indices": [int(x) for x in tda_result.point_indices],
            "pcaviz2": [[round(float(x), 5) for x in row] for row in pca_coords],
            "note": "PCA is visualization-only; TDA ran on the full dimensional point cloud",
        },
        "persistence_diagram": {
            "api": tda_result.diagram.to_api(),
            "points": [p.as_dict() for p in tda_result.diagram.points[:200]],
            "n_essential": {str(k): v for k, v in tda_result.diagram.n_essential.items()},
            "scale_reference": round(tda_result.diagram.scale_reference, 6),
        },
        "persistence_image": {
            "resolution": list(cfg.tda.persistence_image.resolution),
            "bandwidth": cfg.tda.persistence_image.bandwidth,
            "vector": [round(float(x), 6) for x in tda_result.feature_vector.tolist()],
        },
        "betti_statistics": persistence_statistics(tda_result.diagram.dimension_intervals),
        "timings_ms": {k: round(v, 3) for k, v in tda_result.timings_ms.items()},
    }

    if run_id:
        run = load_run(project_root() / "artifacts", run_id)
        predictor = _predictor_for_run(run)
        result = predictor.predict_window(window)
        payload["prediction"] = {
            **result.as_dict(),
            "predicted_label": run.label_names[result.predicted_idx],
            "probabilities": {
                name: round(float(p), 4)
                for name, p in zip(run.label_names, result.probabilities)
            },
            "true_label": dataset.label_names[label] if label is not None else None,
            "correct": (
                (dataset.label_names[label] == run.label_names[result.predicted_idx])
                if label is not None
                else None
            ),
            "model_run_id": run_id,
        }

    return payload


def _predictor_for_run(run):
    from ..inference.predictor import WindowPredictor

    return WindowPredictor(
        model=run.model,
        tda_engine=run.tda_engine,
        window_scaler=run.window_scaler,
        tda_scaler=run.tda_scaler,
    )


def _pca_2d(points: np.ndarray) -> np.ndarray:
    from sklearn.decomposition import PCA

    if points.shape[0] < 2:
        return np.zeros((points.shape[0], 2))
    pca = PCA(n_components=2, random_state=0)
    return pca.fit_transform(points)
