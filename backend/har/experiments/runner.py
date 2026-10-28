"""Experiment orchestration.

One :meth:`ExperimentRunner.run` executes the full research pipeline per
(config, fold)::

    dataset -> preprocess -> windows -> subject-aware split
            -> scaler fit on TRAIN ONLY -> TDA features (train-fit scaler)
            -> model training (torch CNN/LSTM/DCL/fusion) or SVM
            -> per-window evaluation with latency breakdowns
            -> metrics + confusion matrix
            -> artifacts/ + SQLite rows (never overwritten)

Every fold gets its own run_id.
"""

from __future__ import annotations

import dataclasses
import pathlib
import time
from typing import Any

import numpy as np
import torch

from ..config.schema import Config, config_to_dict, validate as validate_config
from ..datasets import LoadedDataset, get_adapter
from ..db import ExperimentStore
from ..evaluation.suite import evaluate_windows
from ..models.factory import build_model
from ..models.svm import SklearnSvm
from ..preprocessing.normalization import ScalerWrapper
from ..preprocessing.pipeline import Preprocessor
from ..splitting import SubjectSplit, make_subject_folds
from ..tda.engine import TdaEngine
from ..training.trainer import Trainer, WindowTorchDataset
from ..utils.logging import get_logger, log_structured
from ..utils.profiler import ResourceMonitor
from ..utils.seeds import seed_everything
from ..windowing.windowing import (
    WindowParams,
    build_window_arrays,
    describe_window_params,
)
from .artifacts import write_environment_info, write_json, write_yaml

logger = get_logger(__name__)


class ExperimentError(RuntimeError):
    pass


class FoldRunResult:
    def __init__(self, run_id: str, fold: int, artifacts_dir: str, accuracy: float, f1_macro: float) -> None:
        self.run_id = run_id
        self.fold = fold
        self.artifacts_dir = artifacts_dir
        self.accuracy = accuracy
        self.f1_macro = f1_macro

    def as_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "fold": self.fold,
            "accuracy": round(self.accuracy, 6),
            "f1_macro": round(self.f1_macro, 6),
            "artifacts_dir": self.artifacts_dir,
        }


def _tda_features(engine: TdaEngine, windows: np.ndarray) -> np.ndarray:
    vectors = [
        engine.compute(np.ascontiguousarray(windows[i])).feature_vector
        for i in range(len(windows))
    ]
    dim = vectors[0].size if vectors else 0
    if dim == 0:
        return np.zeros((len(windows), 0), dtype=np.float32)
    return np.vstack(vectors).astype(np.float32, copy=False)


class ExperimentRunner:
    def __init__(
        self,
        cfg: Config,
        experiment_name: str | None = None,
        store: ExperimentStore | None = None,
    ) -> None:
        self.cfg = cfg
        self.experiment_name = experiment_name or f"{cfg.dataset.name}_{cfg.model.type}"
        self.store = store or ExperimentStore(cfg.paths.db_path)
        self.runs: list[FoldRunResult] = []

    # ------------------------------------------------------------------
    def run(self) -> list[FoldRunResult]:
        seed_everything(self.cfg.training.seed)
        adapter = get_adapter(self.cfg.dataset.name, self.cfg.dataset)
        dataset = adapter.load()
        sampling_rate = float(
            self.cfg.dataset.sampling_rate_override or adapter.get_sampling_rate()
        )

        validate_config(self.cfg, sampling_rate)
        params = WindowParams.from_config(self.cfg, sampling_rate)
        log_structured(logger, 20, "windowing", description=describe_window_params(params))

        preprocessor = Preprocessor(self.cfg, sampling_rate)
        processed = preprocessor.process_dataset(dataset)

        max_windows = self.cfg.window["default"].max_windows
        all_windows = build_window_arrays(processed, params, max_windows=max_windows)
        log_structured(
            logger, 20, "windows", n=int(len(all_windows)), labels=dataset.label_names
        )

        folds = make_subject_folds(
            all_windows.subject_ids,
            all_windows.labels,
            n_splits=self.cfg.evaluation.cv_folds,
            val_splits=self.cfg.training.val_splits,
            folds_to_run=self.cfg.evaluation.folds,
            seed=self.cfg.training.seed,
        )

        results: list[FoldRunResult] = []
        for split in folds:
            results.append(
                self._run_fold(dataset, all_windows, params, split, sampling_rate)
            )
        self.runs = results
        # free the full window matrix only after every fold has consumed it
        all_windows.data = None
        return results

    # ------------------------------------------------------------------
    def _run_fold(
        self,
        dataset: LoadedDataset,
        all_windows,
        params: WindowParams,
        split: SubjectSplit,
        sampling_rate: float,
    ) -> FoldRunResult:
        run_id = self.store.next_run_id()
        self.store.insert_experiment(
            self._experiment_record(run_id, dataset, params, split, sampling_rate)
        )

        artifacts_dir = pathlib.Path(self.cfg.paths.artifacts_dir) / run_id
        artifacts_dir.mkdir(parents=True, exist_ok=True)
        self.store.insert_split_meta(
            run_id,
            split.fold,
            {
                "strategy": "StratifiedGroupKFold(subjects) + inner val carve",
                **split.as_dict(),
            },
            passed=True,
        )

        train_mask = np.isin(all_windows.subject_ids, split.train_subjects)
        val_mask = (
            np.isin(all_windows.subject_ids, split.val_subjects)
            if split.val_subjects
            else None
        )
        test_mask = np.isin(all_windows.subject_ids, split.test_subjects)

        log_structured(
            logger,
            20,
            "split",
            run_id=run_id,
            train_windows=int(train_mask.sum()),
            val_windows=int(val_mask.sum()) if val_mask is not None else 0,
            test_windows=int(test_mask.sum()),
        )

        label_names = dataset.label_names
        n_classes = len(label_names)
        n_channels = all_windows.n_channels

        window_scaler = Preprocessor.new_scaler(self.cfg)
        tda_engine = TdaEngine(self.cfg.tda) if self.cfg.tda.enabled else None

        monitor = ResourceMonitor()
        monitor.start()
        model: Any = None
        train_time_s = 0.0
        history: list[dict[str, Any]] = []
        tda_module_feature_dim = 0
        tda_scaler: ScalerWrapper | None = None
        tda_time_s = 0.0

        is_svm = self.cfg.model.type == "svm"

        with monitor.phase("train"):
            if is_svm:
                # SVM standardises internally (scaler saved inside model.pkl)
                data_train = np.ascontiguousarray(all_windows.data[train_mask])
                data_val = (
                    np.ascontiguousarray(all_windows.data[val_mask])
                    if val_mask is not None and int(val_mask.sum())
                    else np.empty(0, dtype=np.float32)
                )
                data_test = np.ascontiguousarray(all_windows.data[test_mask])
            else:
                # CRITICAL: fit ONLY on training windows
                window_scaler.fit(all_windows.data[train_mask])
                data_train = window_scaler.transform(all_windows.data[train_mask])
                data_val = (
                    window_scaler.transform(all_windows.data[val_mask])
                    if val_mask is not None and int(val_mask.sum())
                    else np.empty(0, dtype=np.float32)
                )
                data_test = window_scaler.transform(all_windows.data[test_mask])

            if is_svm:
                svm = SklearnSvm(self.cfg)
                train_stats = svm.fit(data_train.copy(), all_windows.labels[train_mask])
                model = svm
                train_time_s = float(train_stats["training_time_s"])
                svm.save(artifacts_dir / "model.pkl")
                tda_module_feature_dim = 0
            elif tda_engine is not None:
                t0 = time.perf_counter()
                tda_train = _tda_features(tda_engine, data_train)
                tda_val = (
                    _tda_features(tda_engine, data_val)
                    if len(data_val)
                    else np.zeros((0, tda_train.shape[1]), dtype=np.float32)
                )
                tda_time_s = time.perf_counter() - t0

                tda_scaler = ScalerWrapper(enabled=self.cfg.model.apply_scaler_to_tda)
                tda_scaler.fit(tda_train)
                tda_train = tda_scaler.transform(tda_train)
                tda_val = tda_scaler.transform(tda_val)
                tda_module_feature_dim = int(tda_train.shape[1])

                model = build_model(self.cfg, n_classes, n_channels, tda_module_feature_dim)
                trainer = Trainer(self.cfg)
                train_ds = WindowTorchDataset(
                    data_train, all_windows.labels[train_mask], tda_train
                )
                val_ds = (
                    WindowTorchDataset(
                        data_val, all_windows.labels[val_mask], tda_val
                    )
                    if len(data_val)
                    else None
                )
                result = trainer.fit(model, train_ds, val_ds)
                train_time_s = result.training_time_s + tda_time_s
                history = result.history_dicts()
                torch.save(
                    {"model_state": result.best_state, "epoch": result.best_epoch},
                    artifacts_dir / "model.pt",
                )
            else:
                model = build_model(self.cfg, n_classes, n_channels, 0)
                trainer = Trainer(self.cfg)
                train_ds = WindowTorchDataset(
                    data_train, all_windows.labels[train_mask]
                )
                val_ds = (
                    WindowTorchDataset(data_val, all_windows.labels[val_mask])
                    if len(data_val)
                    else None
                )
                result = trainer.fit(model, train_ds, val_ds)
                train_time_s = result.training_time_s
                history = result.history_dicts()
                torch.save(
                    {"model_state": result.best_state, "epoch": result.best_epoch},
                    artifacts_dir / "model.pt",
                )

            window_scaler.save(artifacts_dir / "model.scaler.pkl")
            (artifacts_dir / "tda_scaler.pkl").unlink(missing_ok=True)
            if tda_scaler is not None:
                tda_scaler.save(artifacts_dir / "tda_scaler.pkl")
            write_environment_info(artifacts_dir, self.cfg.training.seed)

        with monitor.phase("evaluate"):
            evaluation = evaluate_windows(
                model=model,
                data_test=data_test,
                labels_test=all_windows.labels[test_mask],
                subjects_test=all_windows.subject_ids[test_mask],
                window_ids_test=all_windows.window_ids[test_mask],
                timestamps_test=all_windows.timestamps[test_mask],
                label_names=label_names,
                tda_engine=tda_engine,
                window_scaler=window_scaler,
                tda_scaler=tda_scaler,
                train_time_s=train_time_s,
            )
        monitor.stop()
        resources = monitor.report()

        metrics_obj = evaluation.metrics
        self.store.insert_summary(
            {
                "run_id": run_id,
                "model_name": self.cfg.model.type,
                "accuracy": metrics_obj.accuracy,
                "precision_macro": metrics_obj.precision_macro,
                "recall_macro": metrics_obj.recall_macro,
                "f1_macro": metrics_obj.f1_macro,
                "precision_weighted": metrics_obj.precision_weighted,
                "recall_weighted": metrics_obj.recall_weighted,
                "f1_weighted": metrics_obj.f1_weighted,
                "avg_inference_ms": evaluation.latencies.get("mean_ms"),
                "p95_inference_ms": evaluation.latencies.get("p95_ms"),
                "peak_ram_mb": max(
                    [
                        float(p.get("peak_rss_mb", 0.0))
                        for p in resources.values()
                        if isinstance(p, dict) and "peak_rss_mb" in p
                    ]
                    or [0.0]
                ),
                "training_time_s": round(train_time_s, 3),
                "num_windows": int(len(evaluation.y_pred)),
                "fold": split.fold,
                "train_subjects": split.train_subjects,
                "val_subjects": split.val_subjects,
                "test_subjects": split.test_subjects,
            }
        )
        self.store.insert_window_evaluations(run_id, evaluation.window_records)
        self.store.insert_training_history(run_id, history)
        self.store.insert_resource_measurements(run_id, resources)
        self.store.insert_confusion_matrix(run_id, evaluation.confusion_rows)

        # artifacts --------------------------------------------------------
        write_json(
            artifacts_dir,
            "metrics.json",
            {
                "run_id": run_id,
                "fold": split.fold,
                **evaluation.summary_dict(),
                "train_time_s": round(train_time_s, 3),
                "resources": resources,
            },
        )
        write_json(artifacts_dir, "confusion_matrix.json", evaluation.confusion_rows)
        write_json(artifacts_dir, "training_history.json", history)
        write_yaml(artifacts_dir, "config.yaml", config_to_dict(self.cfg))
        write_json(
            artifacts_dir,
            "pipeline.json",
            {
                "run_id": run_id,
                "dataset": self.cfg.dataset.name,
                "signal_mode": self.cfg.dataset.signal_mode,
                "label_names": label_names,
                "channels": dataset.channel_names,
                "sampling_rate": sampling_rate,
                "window_params": params.as_dict(),
                "window_shape": [int(x) for x in data_test.shape[1:]],
                "tda_enabled": bool(tda_engine is not None),
                "tda_feature_dim": tda_module_feature_dim,
                "model_type": self.cfg.model.type,
                "preprocessing": dataclasses.asdict(self.cfg.preprocessing),
                "split": split.as_dict(),
                "dataset_notes": dataset.notes,
            },
        )

        log_structured(
            logger,
            20,
            "run_complete",
            run_id=run_id,
            accuracy=round(metrics_obj.accuracy, 4),
            f1_macro=round(metrics_obj.f1_macro, 4),
            avg_ms=evaluation.latencies.get("mean_ms"),
        )
        run = FoldRunResult(
            run_id=run_id,
            fold=split.fold,
            artifacts_dir=str(artifacts_dir),
            accuracy=metrics_obj.accuracy,
            f1_macro=metrics_obj.f1_macro,
        )
        self.runs.append(run)
        return run

    # ------------------------------------------------------------------
    def _experiment_record(
        self,
        run_id: str,
        dataset: LoadedDataset,
        params: WindowParams,
        split: SubjectSplit,
        sampling_rate: float,
    ) -> dict[str, Any]:
        tda_points = (
            self.cfg.tda.subsample.max_points if self.cfg.tda.enabled else None
        )
        return {
            "run_id": run_id,
            "experiment_name": self.experiment_name,
            "dataset": self.cfg.dataset.name,
            "model_name": self.cfg.model.type,
            "window_duration": params.window_duration,
            "overlap": params.overlap,
            "window_size": params.window_size,
            "step_size": params.step,
            "sampling_rate": sampling_rate,
            "tda_enabled": int(self.cfg.tda.enabled),
            "tda_points": tda_points,
            "tda_resolution": "{}x{}".format(*self.cfg.tda.persistence_image.resolution),
            "config_json": config_to_dict(self.cfg),
        }
