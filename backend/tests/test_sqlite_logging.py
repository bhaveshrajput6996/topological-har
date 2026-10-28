"""SQLite experiment store tests (schema, ids, never-overwrite rule)."""

import pytest

from har.db import ExperimentStore


@pytest.fixture()
def store(tmp_path):
    return ExperimentStore(tmp_path / "test_har.db")


_EXPR_RECORD = {
    "run_id": "HAR-2026-001",
    "experiment_name": "e",
    "dataset": "d",
    "model_name": "m",
    "window_duration": 2.0,
    "overlap": 0.5,
    "window_size": 64,
    "step_size": 32,
    "sampling_rate": 32.0,
    "tda_enabled": 1,
    "tda_points": 80,
    "tda_resolution": "20x20",
    "config_json": {},
}


def _expr(record_from: str = ""):
    rec = dict(_EXPR_RECORD)
    rec["run_id"] = record_from or "HAR-2026-001"
    return rec


def test_run_ids_increase_with_year_prefix(store):
    assert store.next_run_id() == "HAR-2026-001"
    store.insert_experiment(_expr("HAR-2026-001"))
    assert store.next_run_id() == "HAR-2026-002"


def test_summary_roundtrip(store):
    store.insert_experiment(_expr())
    store.insert_summary(
        {"run_id": "HAR-2026-001", "model_name": "tda_deepconvlstm", "accuracy": 0.9,
         "precision_macro": 0.9, "recall_macro": 0.9, "f1_macro": 0.9,
         "precision_weighted": 0.9, "recall_weighted": 0.9, "f1_weighted": 0.9,
         "avg_inference_ms": 10.0, "p95_inference_ms": 20.0, "peak_ram_mb": 500.0,
         "training_time_s": 1.0, "num_windows": 100, "fold": 0,
         "train_subjects": ["s0"], "val_subjects": [], "test_subjects": ["s1"]}
    )
    rows = store.summary_rows()
    assert len(rows) == 1
    r = rows[0]
    assert r.run_id == "HAR-2026-001" and abs(r.accuracy - 0.9) < 1e-9
    assert store.get_experiment("HAR-2026-001") is not None
    assert store.get_summary("HAR-2026-001")["accuracy"] == 0.9


def test_window_history_roundtrip(store):
    store.insert_experiment(_expr())
    recs = [
        {
            "window_id": f"s{i}-{i}",
            "subject_id": f"s{i}",
            "predicted_label": "SITTING",
            "true_label": "SITTING",
            "confidence": 0.9,
            "correct": i % 2,
            "inference_time_ms": 10.0 + i,
            "tda_ms": 5.0,
            "deep_ms": 4.0,
            "fusion_ms": 1.0,
            "probabilities": [0.1, 0.2, 0.7],
            "timestamp": 0.0,
        }
        for i in range(3)
    ]
    store.insert_window_evaluations("HAR-2026-001", recs)
    rows = store.get_window_history("HAR-2026-001")
    assert len(rows) == 3
    got = store.get_window_history("HAR-2026-001", subject_id="s1")
    assert len(got) == 1


def test_confusion_and_training_roundtrip(store):
    store.insert_experiment(_expr())
    store.insert_confusion_matrix(
        "HAR-2026-001",
        [{"true_label": "a", "predicted_label": "a", "count": 3}],
    )
    store.insert_training_history(
        "HAR-2026-001",
        [{"epoch": 1, "train_loss": 1.0, "val_loss": 0.9, "train_accuracy": 0.5, "val_accuracy": 0.5, "epoch_time_s": 1.0}],
    )
    cm = store.get_confusion_matrix("HAR-2026-001")
    assert cm[0]["true_label"] == "a"
    hist = store.get_training_history("HAR-2026-001")
    assert hist[0]["epoch"] == 1


def test_never_overwrite_experiment_row(store):
    store.insert_experiment(_expr())
    with pytest.raises(ValueError):
        store.insert_experiment(_expr())
