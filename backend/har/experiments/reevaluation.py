"""Re-evaluation of a stored run on its own saved test split.

Reads artifacts/{run_id}/pipeline.json (label names, split, channels),
rebuilds the dataset with the SAME configuration and confirms the split
identically through the saved seed, then re-evaluates the stored model on
the test windows.  This is used by `evaluate --run-id ...` and the API.
"""

from __future__ import annotations

from typing import Any

from ..config.schema import Config, DatasetConfig, config_from_dict
from ..datasets import get_adapter
from ..db import ExperimentStore
from ..evaluation.suite import evaluate_windows
from ..models.loader import load_run
from ..preprocessing.pipeline import Preprocessor
from ..preprocessing.normalization import ScalerWrapper
from ..splitting import make_subject_folds
from ..utils.logging import get_logger
from ..windowing.windowing import WindowParams, build_window_arrays
from ..utils.seeds import project_root

logger = get_logger(__name__)


class ReevaluationError(RuntimeError):
    pass


def evaluate_stored_run(run_id: str) -> dict[str, Any]:
    run = load_run(project_root() / "artifacts", run_id)
    cfg = run.cfg
    pipeline = run.pipeline

    adapter = get_adapter(cfg.dataset.name, cfg.dataset)
    dataset = adapter.load()
    sampling_rate = float(
        cfg.dataset.sampling_rate_override or adapter.get_sampling_rate()
    )
    params = WindowParams.from_config(cfg, sampling_rate)
    preprocessor = Preprocessor(cfg, sampling_rate)
    processed = preprocessor.process_dataset(dataset)

    arrays = build_window_arrays(
        processed, params, max_windows=cfg.window["default"].max_windows
    )

    # reproduce the subject split deterministically
    folds = make_subject_folds(
        arrays.subject_ids,
        arrays.labels,
        n_splits=cfg.evaluation.cv_folds,
        val_splits=cfg.training.val_splits,
        folds_to_run=[int(pipeline["split"]["fold"])],
        seed=cfg.training.seed,
    )
    split = folds[0]
    test_mask = _mask_of(arrays.subject_ids, split.test_subjects)
    data_test = arrays.data[test_mask]
    if run.is_svm:
        # SVM scales internally; hand it the raw windows
        window_scaler = ScalerWrapper(enabled=False)
    else:
        window_scaler = run.window_scaler

    store = ExperimentStore(cfg.paths.db_path)
    previous = store.get_summary(run_id)

    evaluation = evaluate_windows(
        model=run.model,
        data_test=data_test,
        labels_test=arrays.labels[test_mask],
        subjects_test=arrays.subject_ids[test_mask],
        window_ids_test=arrays.window_ids[test_mask],
        timestamps_test=arrays.timestamps[test_mask],
        label_names=run.label_names,
        tda_engine=run.tda_engine,
        window_scaler=(ScalerWrapper(enabled=False) if run.is_svm else window_scaler),
        tda_scaler=run.tda_scaler,
        train_time_s=None if previous is None else previous.get("training_time_s"),
    )

    return {
        "run_id": run_id,
        "test_subjects": split.test_subjects,
        "metrics": evaluation.metrics.as_dict(),
        "latency": evaluation.latencies,
        "num_windows": int(len(evaluation.y_pred)),
    }


def _mask_of(subjects, wanted: list[str]):
    import numpy as np

    return np.isin(subjects, wanted)
