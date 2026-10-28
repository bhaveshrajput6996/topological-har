"""REST API integration tests (TestClient; heavy endpoints budgeted)."""

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

UCI_READY = (Path(__file__).resolve().parents[1] / "data" / "raw" / "UCI HAR Dataset"
             / "train" / "X_train.txt").exists()


@pytest.fixture(scope="module")
def client():
    import warnings

    warnings.filterwarnings("ignore")
    from app.main import create_app

    with TestClient(create_app()) as c:
        yield c


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert "version" in body


def test_root_lists_docs(client):
    r = client.get("/")
    assert r.status_code == 200
    assert "docs" in r.json()


def test_list_datasets(client):
    r = client.get("/datasets")
    assert r.status_code == 200
    assert "uci_har" in r.json()["adapters"]


def test_unknown_dataset_subjects_404(client):
    r = client.get("/datasets/not_a_real_one/subjects")
    assert r.status_code == 404


def test_dataset_load_rejects_bad_name(client):
    r = client.post("/datasets/load", json={"name": "nope", "signal_mode": None})
    assert r.status_code == 400


@pytest.mark.skipif(not UCI_READY, reason="UCI HAR dataset not downloaded")
def test_ucihar_load_and_subjects(client):
    r = client.post("/datasets/load", json={"name": "uci_har", "subjects": ["1", "2"]})
    assert r.status_code == 200
    body = r.json()
    assert sorted(body["subjects"]) == ["subject_01", "subject_02"]
    assert body["kind"] == "presegmented"
    r = client.get("/datasets/uci_har/subjects")
    assert r.status_code == 200
    assert len(r.json()["subjects"]) == 30


def test_window_inspect_bare_shape(client):
    r = client.get("/windows/inspect", params={"dataset": "synthetic", "window_index": 0})
    assert r.status_code == 200
    body = r.json()
    assert body["window_shape"][-1] == 3
    assert "persistence_diagram" in body and "persistence_image" in body
    assert len(body["persistence_image"]["vector"]) == 400


def test_predict_requires_payload_choice(client):
    r = client.post("/predict", json={"run_id": "HAR-0000-000"})
    assert r.status_code == 400


def test_predict_unknown_run_404(client):
    r = client.post("/predict", json={
        "run_id": "HAR-0000-999", "dataset": "synthetic",
        "subject": "synthetic_s00", "window_index": 0,
    })
    assert r.status_code == 404


def test_experiments_listing(client):
    r = client.get("/experiments")
    assert r.status_code == 200
    assert "experiments" in r.json()


def test_unknown_experiment_404(client):
    r = client.get("/experiments/HAR-0000-999")
    assert r.status_code == 404


def test_replay_requires_existing_session(client):
    r = client.post("/replay/step", json={"session_id": "nope"})
    assert r.status_code == 404


def test_replay_start_stop_synthetic(client):
    r = client.post("/replay/start", json={
        "dataset": "synthetic", "max_windows": 3, "speed": 1.0,
    })
    assert r.status_code == 200
    sid = r.json()["session_id"]
    r = client.post("/replay/step", json={"session_id": sid})
    assert r.status_code == 200
    body = r.json()
    # without run_id there is no trained model -> no prediction piggybacked
    assert body.get("prediction") is None
    assert body["session_id"] == sid
    assert body["subject_id"].startswith("synthetic_")
    r = client.post("/replay/stop", json={"session_id": sid})
    assert r.status_code == 200


def test_experiment_status_unknown_404(client):
    r = client.get("/experiments/status/nope")
    assert r.status_code == 404


def test_websocket_replay_unknown_session(client):
    with client.websocket_connect("/ws/replay/unknown-session") as ws:
        msg = ws.receive_json()
        assert "error" in msg
