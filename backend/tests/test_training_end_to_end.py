"""End-to-end runner test on the synthetic dataset (tiny budgets)."""

import json
import pathlib

from har.config.schema import Config, SubsampleConfig, TdaConfig, resolve_paths
from har.experiments.runner import ExperimentRunner
from har.utils.seeds import project_root

PROJECT_ROOT = project_root()


def _tiny_config(tmp: pathlib.Path) -> Config:
    cfg = Config()
    cfg.dataset.name = "synthetic"
    cfg.paths.artifacts_dir = str(tmp / "artifacts")
    cfg.paths.db_path = str(tmp / "test.db")
    cfg.tda = TdaConfig(
        subsample=SubsampleConfig(enabled=True, max_points=60),
        statistical_features=True,
    )
    cfg.model.conv_channels = [16, 32, 32, 64]
    cfg.model.lstm_hidden = 32
    cfg.model.head_hidden = [64, 32]
    cfg.training.epochs = 2
    cfg.training.val_splits = 1
    cfg.training.batch_size = 32
    cfg.evaluation.cv_folds = 2
    cfg.evaluation.folds = [0]
    cfg.window["default"].duration_seconds = 2.0
    resolve_paths(cfg)
    return cfg


def test_runner_produces_complete_artifacts(tmp_path):
    runner = ExperimentRunner(_tiny_config(tmp_path), "e2e-tiny")
    results = runner.run()
    assert len(results) == 1
    r = results[0]
    assert r.run_id.startswith("HAR-2026-")

    art = pathlib.Path(r.artifacts_dir)
    for name in ("metrics.json", "confusion_matrix.json", "training_history.json",
                 "pipeline.json", "config.yaml", "environment.json", "model.pt",
                 "model.scaler.pkl", "tda_scaler.pkl"):
        assert (art / name).exists(), f"missing artifact {name}"

    metrics = json.loads((art / "metrics.json").read_text())
    inner = metrics["metrics"]
    assert 0.0 <= inner["accuracy"] <= 1.0
    assert inner["f1_macro"] >= 0.0
    assert "latency" in metrics

    # SQLite rows recorded
    from har.db import ExperimentStore

    store = ExperimentStore(str(tmp_path / "test.db"))
    assert store.get_summary(r.run_id) is not None
    windows = store.get_window_history(r.run_id)
    assert len(windows) > 0
    assert any(w["tda_ms"] is not None for w in windows)
    cm = store.get_confusion_matrix(r.run_id)
    sum_points = sum(row["count"] for row in cm)
    n_sum = store.get_summary(r.run_id)["num_windows"]
    assert sum_points == n_sum
