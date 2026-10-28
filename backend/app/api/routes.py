"""REST + WebSocket routes for the TDL-HAR backend."""

from __future__ import annotations

import json
import threading
from typing import Any

import numpy as np
from fastapi import APIRouter, HTTPException, Query, WebSocket, WebSocketDisconnect

from har.config.schema import ConfigError, DatasetConfig, config_from_dict
from har.db import ExperimentStore
from har.models.loader import LoadedRun
from har.utils.seeds import project_root
from har.experiments.inspection import inspect_dataset_window
from har.utils.logging import get_logger, log_structured

from ..schemas import (
    DatasetLoadRequest,
    ExperimentRequest,
    PredictionRequest,
    ReplayControlRequest,
    ReplayStartRequest,
)
from ..services.registry import dataset_service, model_service
from ..services.replay import ReplayError, replay_manager

logger = get_logger(__name__)

router = APIRouter()

run_statuses: dict[str, dict[str, Any]] = {}
run_lock = threading.Lock()


def _http(status_code: int, error: str) -> HTTPException:
    return HTTPException(status_code=status_code, detail=error)


# ---------------------------------------------------------------------------
# health / datasets
# ---------------------------------------------------------------------------
@router.get("/health")
def health() -> dict[str, Any]:
    import torch
    from har.version import __version__

    return {
        "status": "ok",
        "version": __version__,
        "torch_cuda": bool(torch.cuda.is_available()),
    }


@router.get("/datasets")
def list_datasets() -> dict[str, Any]:
    return {"adapters": dataset_service().list_adapters()}


@router.post("/datasets/load")
def load_dataset(payload: DatasetLoadRequest) -> dict[str, Any]:
    dc = DatasetConfig(
        name=payload.name,
        root=payload.root,
        signal_mode=payload.signal_mode,
        subjects=payload.subjects,
        channels=payload.channels,
        activity_filter=payload.activity_filter,
    )
    try:
        info = dataset_service().dataset_info(dc)
    except Exception as exc:  # adapter errors carry user-friendly text
        raise _http(400, f"dataset load failed: {exc}")
    return info


@router.get("/datasets/{name}/subjects")
def dataset_subjects(name: str) -> dict[str, Any]:
    try:
        info = dataset_service().dataset_info(DatasetConfig(name=name))
    except Exception as exc:
        raise _http(404, f"unknown/failed dataset: {exc}")
    return {"dataset": name, "subjects": info["subjects"]}


# ---------------------------------------------------------------------------
# experiments (training + history)
# ---------------------------------------------------------------------------
@router.get("/experiments")
def list_experiments() -> dict[str, Any]:
    store = ExperimentStore(project_root() / "artifacts" / "har_experiments.db")
    rows = store.summary_rows()
    return {
        "experiments": [
            {
                "run_id": r.run_id,
                "experiment_name": r.experiment_name,
                "dataset": r.dataset,
                "model": r.model_name,
                "fold": r.fold,
                "accuracy": r.accuracy,
                "f1_macro": r.f1_macro,
                "f1_weighted": r.f1_weighted,
                "avg_inference_ms": r.avg_inference_ms,
                "training_time_s": r.training_time_s,
                "peak_ram_mb": r.peak_ram_mb,
                "num_windows": r.num_windows,
                "created_at": r.created_at,
            }
            for r in rows
        ]
    }


@router.post("/experiments")
def create_experiment(payload: ExperimentRequest) -> dict[str, Any]:
    """Start a training run in the background (status in /experiments/status)."""
    try:
        if payload.config:
            cfg = config_from_dict(payload.config)
        else:
            from har.config.schema import Config, resolve_paths

            cfg = resolve_paths(Config())
    except Exception as exc:
        raise _http(400, f"invalid config: {exc}")
    experiment_id = f"exp-{int(threading.get_ident())}-{int(__import__('time').time()) % 1000000}"
    with run_lock:
        run_statuses[experiment_id] = {"state": "submitted", "runs": []}

    def _worker() -> None:
        from har.experiments.runner import ExperimentRunner

        try:
            with run_lock:
                run_statuses[experiment_id] = {"state": "running", "runs": []}
            runner = ExperimentRunner(cfg, experiment_name=payload.name or "api_run")
            results = runner.run()
            with run_lock:
                run_statuses[experiment_id] = {
                    "state": "done",
                    "runs": [r.as_dict() for r in results],
                }
        except Exception as exc:  # keep the API alive after training errors
            logger.exception("api training failed")
            with run_lock:
                run_statuses[experiment_id] = {"state": "failed", "error": str(exc)}

    threading.Thread(target=_worker, daemon=True, name=f"train-{experiment_id}").start()
    return {"experiment_id": experiment_id, "state": "submitted"}


@router.get("/experiments/status/{experiment_id}")
def experiment_status(experiment_id: str) -> dict[str, Any]:
    with run_lock:
        if experiment_id not in run_statuses:
            raise _http(404, f"unknown experiment_id {experiment_id}")
        return run_statuses[experiment_id]


@router.get("/experiments/{run_id}")
def experiment_detail(run_id: str) -> dict[str, Any]:
    store = ExperimentStore(project_root() / "artifacts" / "har_experiments.db")
    record = store.get_experiment(run_id)
    if record is None:
        raise _http(404, f"unknown run_id {run_id}")
    summary = store.get_summary(run_id)
    split_meta = store.get_split_meta(run_id)
    return {
        "run_id": run_id,
        "experiment": record,
        "summary": summary,
        "split": split_meta,
    }


@router.get("/experiments/{run_id}/metrics")
def experiment_metrics(run_id: str) -> dict[str, Any]:
    store = ExperimentStore(project_root() / "artifacts" / "har_experiments.db")
    summary = store.get_summary(run_id)
    if summary is None:
        raise _http(404, f"no evaluation summary stored for {run_id}")
    metrics_path = project_root() / "artifacts" / run_id / "metrics.json"
    metrics_json = (
        json.loads(metrics_path.read_text(encoding="utf-8")) if metrics_path.exists() else None
    )
    return {"run_id": run_id, "summary": summary, "metrics": metrics_json}


@router.get("/experiments/{run_id}/confusion-matrix")
def experiment_confusion(run_id: str) -> dict[str, Any]:
    store = ExperimentStore(project_root() / "artifacts" / "har_experiments.db")
    return {"run_id": run_id, "confusion_matrix": store.get_confusion_matrix(run_id)}


@router.get("/experiments/{run_id}/history")
def experiment_history(
    run_id: str,
    limit: int = Query(default=200, ge=1, le=50000),
    subject: str | None = Query(default=None),
    correct: bool | None = Query(default=None),
) -> dict[str, Any]:
    store = ExperimentStore(project_root() / "artifacts" / "har_experiments.db")
    rows = store.get_window_history(run_id, limit=limit, subject_id=subject, correct=correct)
    return {"run_id": run_id, "windows": rows, "count": len(rows)}


@router.get("/experiments/{run_id}/training-history")
def experiment_training_history(run_id: str) -> dict[str, Any]:
    store = ExperimentStore(project_root() / "artifacts" / "har_experiments.db")
    return {
        "run_id": run_id,
        "epochs": store.get_training_history(run_id),
        "resources": store.get_resource_measurements(run_id),
    }


# ---------------------------------------------------------------------------
# prediction
# ---------------------------------------------------------------------------
@router.post("/predict")
def predict(payload: PredictionRequest) -> dict[str, Any]:
    if not payload.data and payload.dataset is None and payload.subject is None:
        raise _http(400, "provide 'data' windows or dataset+subject+window_index")
    try:
        run = model_service().get(payload.run_id)
    except Exception as exc:
        raise _http(404, f"cannot load run {payload.run_id}: {exc}")

    true_label = None
    correct = None
    window_id = payload.window_id if payload.window_id is not None else payload.window_index
    confidence_payload: dict[str, Any]

    if payload.data:
        window = np.asarray(payload.data, dtype=np.float32)
        if window.ndim == 3 and window.shape[0] == 1:
            window = window[0]
        if window.ndim != 2:
            raise _http(400, "data must be a (T, C) window (or list with one)")
        expected = run.window_shape
        if list(window.shape) != expected:
            raise _http(
                400,
                f"window shape {list(window.shape)} does not match model input {expected}",
            )
        predictor = _predictor_for(run)
        result = predictor.predict_window(window)
        confidence_payload = result.as_dict()
    else:
        if payload.subject is None or payload.window_index is None:
            raise _http(400, "subject + window_index required without inline data")
        try:
            outcome = inspect_dataset_window(
                dataset_name=payload.dataset,
                subject=payload.subject,
                window_index=payload.window_index,
                run_id=payload.run_id,
                config_path=None,
            )
        except Exception as exc:
            raise _http(400, f"prediction from dataset failed: {exc}")
        prediction = outcome.get("prediction")
        if prediction is None:
            raise _http(503, "run has no model prediction capability")
        confidence_payload = {
            "predicted_label_idx": outcome["prediction"]["predicted_label_idx"],
            "confidence": outcome["prediction"]["confidence"],
            "probabilities": list(outcome["prediction"]["probabilities"].values()),
            "inference_time_ms": outcome["prediction"]["inference_time_ms"],
            "tda_ms": outcome["prediction"]["tda_ms"],
            "deep_ms": outcome["prediction"]["deep_ms"],
            "fusion_ms": outcome["prediction"]["fusion_ms"],
        }
        true_label = outcome["ground_truth"]["label"]
        predicted_label = outcome["prediction"]["predicted_label"]
        correct = (true_label == predicted_label) if true_label else None

    from har.inference.predictor import WindowPredictor  # noqa: F401

    predicted_label = run.label_names[confidence_payload["predicted_label_idx"]]
    return {
        "run_id": payload.run_id,
        "model": run.pipeline["model_type"],
        "window_id": window_id,
        "predicted_activity": predicted_label,
        "confidence": confidence_payload["confidence"],
        "probabilities": {
            name: round(float(p), 6)
            for name, p in zip(run.label_names, confidence_payload["probabilities"])
        },
        "inference_time_ms": confidence_payload["inference_time_ms"],
        "tda_time_ms": confidence_payload.get("tda_ms", 0.0),
        "deep_time_ms": confidence_payload.get("deep_ms", 0.0),
        "fusion_time_ms": confidence_payload.get("fusion_ms", 0.0),
        "true_activity": true_label,
        "correct": correct,
    }


# ---------------------------------------------------------------------------
# window inspection
# ---------------------------------------------------------------------------
@router.get("/windows/inspect")
def windows_inspect(
    dataset: str = Query(...),
    subject: str | None = Query(None),
    window_index: int = Query(0, ge=0),
    run_id: str | None = Query(None),
    config_path: str | None = Query(None),
) -> dict[str, Any]:
    try:
        return inspect_dataset_window(
            dataset_name=dataset,
            subject=subject,
            window_index=window_index,
            run_id=run_id,
            config_path=config_path,
        )
    except Exception as exc:
        raise _http(400, f"inspection failed: {exc}")


# ---------------------------------------------------------------------------
# replay
# ---------------------------------------------------------------------------
@router.post("/replay/start")
def replay_start(payload: ReplayStartRequest) -> dict[str, Any]:
    try:
        return replay_manager().start(
            run_id=payload.run_id,
            dataset_name=payload.dataset,
            config_override=payload.config,
            speed=payload.speed,
            subject=payload.subject,
            max_windows=payload.max_windows,
        )
    except ReplayError as exc:
        raise _http(400, str(exc))
    except Exception as exc:
        logger.exception("replay start failed")
        raise _http(400, f"replay start failed: {exc}")


@router.get("/replay/{session_id}/window/{index}")
def replay_window(session_id: str, index: int) -> dict[str, Any]:
    try:
        session = replay_manager().get_session(session_id)
        event = session.window_at(index)
        return session.predict(event)
    except ReplayError as exc:
        raise _http(404, str(exc))


@router.post("/replay/step")
def replay_step(payload: ReplayControlRequest) -> dict[str, Any]:
    try:
        session = replay_manager().get_session(payload.session_id)
    except ReplayError as exc:
        raise _http(404, str(exc))
    session.index = max(0, session.index)
    event = session.window_at(min(session.index, len(session.source) - 1))
    payload_out = session.predict(event)
    session.push(payload_out)
    session.index += 1
    return payload_out


@router.post("/replay/control")
def replay_control(payload: ReplayControlRequest) -> dict[str, Any]:
    if payload.speed is None:
        raise _http(400, "speed is required")
    try:
        return replay_manager().set_speed(payload.session_id, payload.speed)
    except ReplayError as exc:
        raise _http(404, str(exc))


@router.post("/replay/play")
def replay_play(payload: ReplayControlRequest) -> dict[str, Any]:
    try:
        session = replay_manager().get_session(payload.session_id)
        session.autoplay(payload.speed)
        return {"session_id": payload.session_id, "autoplays": True, "speed": session.speed}
    except ReplayError as exc:
        raise _http(404, str(exc))


@router.post("/replay/stop")
def replay_stop(payload: ReplayControlRequest) -> dict[str, Any]:
    try:
        return replay_manager().stop(payload.session_id)
    except ReplayError as exc:
        raise _http(404, str(exc))


@router.get("/replay/{session_id}/events")
def replay_events(session_id: str) -> dict[str, Any]:
    try:
        return {"session_id": session_id, "events": replay_manager().drain(session_id)}
    except ReplayError as exc:
        raise _http(404, str(exc))


# ---------------------------------------------------------------------------
# WebSocket streaming replay
# ---------------------------------------------------------------------------
@router.websocket("/ws/replay/{session_id}")
async def replay_ws(websocket: WebSocket, session_id: str) -> None:
    await websocket.accept()
    try:
        session = replay_manager().get_session(session_id)
    except ReplayError:
        await websocket.send_json({"error": f"unknown replay session {session_id}"})
        await websocket.close()
        return

    try:
        while True:
            events = replay_manager().drain(session_id)
            for event_payload in events:
                await websocket.send_json(event_payload)
            n = len(events)
            if n == 0:
                if session.finished:
                    await websocket.send_json({"event": "end", "session_id": session_id})
                    break
                await websocket.send_json({"event": "heartbeat", "session_id": session_id})
            try:
                await websocket.receive_text()
            except WebSocketDisconnect:
                break
    except WebSocketDisconnect:
        pass
    finally:
        try:
            await websocket.close()
        except Exception:
            pass


def _predictor_for(run: LoadedRun):
    from har.inference.predictor import WindowPredictor

    return WindowPredictor(
        model=run.model,
        tda_engine=run.tda_engine,
        window_scaler=run.window_scaler,
        tda_scaler=run.tda_scaler,
    )
