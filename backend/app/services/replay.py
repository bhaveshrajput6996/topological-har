"""Replay manager for the API: session registry, autoplay thread, WebSocket
event queue.  Sessions replay dataset windows through a trained model with
configured speed (0.25x..5x); correctness/ground-truth passed through."""

from __future__ import annotations

import itertools
import queue
import threading
import time
import uuid
from typing import Any

import numpy as np
from har.config.schema import Config, DatasetConfig, config_from_dict
from har.db import ExperimentStore
from har.inference.predictor import WindowPredictor, WindowPredictionResult
from har.models.loader import LoadedRun
from har.preprocessing.pipeline import Preprocessor
from har.replay.replay import ReplayEvent, ReplaySource, dataset_into_source
from har.utils.logging import get_logger, log_structured
from har.utils.seeds import project_root
from har.windowing.windowing import WindowParams, build_window_arrays

from .registry import dataset_service, model_service

logger = get_logger(__name__)


class ReplayError(RuntimeError):
    pass


class ReplaySession:
    def __init__(
        self,
        session_id: str,
        source: ReplaySource,
        predictor: WindowPredictor | None,
        label_names: list[str],
        dataset_name: str,
        speed: float = 1.0,
        loop: bool = False,
    ) -> None:
        self.session_id = session_id
        self.source = source
        self.predictor = predictor
        self.label_names = label_names
        self.dataset_name = dataset_name
        self.speed = max(0.25, min(5.0, speed))
        self.loop = loop
        self.index = 0
        self.finished = False
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.events: "queue.Queue[dict[str, Any]]" = queue.Queue(maxsize=500)

    # ------------------------------------------------------------------
    def window_at(self, index: int) -> ReplayEvent:
        if not (0 <= index < len(self.source)):
            raise ReplayError(f"invalid window index {index} (0..{len(self.source)-1})")
        window = np.asarray(self.source.windows[index], dtype=np.float32)
        label_idx = (
            int(self.source.labels[index]) if self.source.labels is not None else None
        )
        return ReplayEvent(
            index=index,
            subject_id=str(self.source.subjects[index]),
            window_id=int(self.source.window_ids[index]),
            window=window,
            timestamp=(float(self.source.timestamps[index]) if self.source.timestamps is not None else float(index)),
            true_label=self.source.label_name(label_idx),
            true_label_idx=label_idx,
        )

    # ------------------------------------------------------------------
    def predict(self, event: ReplayEvent) -> dict[str, Any]:
        payload = {
            "session_id": self.session_id,
            "dataset": self.dataset_name,
            "source": "replay(dataset=%s)" % self.dataset_name,
            "index": event.index,
            "window_id": event.window_id,
            "subject_id": event.subject_id,
            "timestamp": event.timestamp,
            "window_shape": [int(x) for x in event.window.shape],
            "sensor_sample": event.window[: min(8, len(event.window)), 0].round(4).tolist(),
            "true_activity": event.true_label,
            "true_label_idx": event.true_label_idx,
            "correct": None,
            "prediction": None,
        }
        if self.predictor is not None:
            result = self.predictor.predict_window(event.window)
            predicted_label = (
                self.label_names[result.predicted_idx]
                if result.predicted_idx < len(self.label_names)
                else str(result.predicted_idx)
            )
            payload["prediction"] = {
                **result.as_dict(),
                "predicted_activity": predicted_label,
                "probabilities": {
                    name: round(float(p), 4)
                    for name, p in zip(self.label_names, result.probabilities)
                },
            }
            if event.true_label_idx is not None:
                payload["correct"] = result.predicted_idx == event.true_label_idx
        return payload

    # ------------------------------------------------------------------
    def autoplay(self, speed: float | None = None) -> None:
        if speed is not None:
            self.speed = max(0.25, min(5.0, speed))
        if self._thread is None or not self._thread.is_alive():
            self._stop.clear()
            self._thread = threading.Thread(
                target=self._run, name=f"replay-{self.session_id}", daemon=True
            )
            self._thread.start()

    def stop(self) -> None:
        self._stop.set()

    def _run(self) -> None:
        delay_s = max(0.0, self.speed and (1.0 / self.speed))
        delay = min(2.5, max(0.02, delay_s * 0.05))  # 20ms per window at 1x
        while not self._stop.is_set() and not self.finished:
            index = self.index
            if index >= len(self.source):
                if self.loop:
                    self.index = 0
                    continue
                self.finished = True
                break
            event = self.window_at(index)
            payload = self.predict(event)
            self.push(payload)
            self.index += 1
            time.sleep(delay)

    def push(self, payload: dict[str, Any]) -> None:
        try:
            self.events.put_nowait(payload)
        except queue.Full:
            # never drop silently while animating; consumers are slower than
            # autoplay, so backfilling is acceptable for the demo (documented)
            try:
                self.events.get_nowait()
                self.events.put_nowait(payload)
            except (queue.Empty, queue.Full):
                pass


class ReplayManager:
    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._sessions: dict[str, ReplaySession] = {}

    # ------------------------------------------------------------------
    def _build_source(
        self,
        run_id: str | None,
        dataset_name: str | None,
        config_override: dict[str, Any] | None,
        subject: str | None,
        max_windows: int | None,
    ) -> tuple[ReplaySource, WindowPredictor | None, str, list[str]]:
        if run_id:
            run = model_service().get(run_id)
            cfg = run.cfg
            predictor = WindowPredictor(
                model=run.model,
                tda_engine=run.tda_engine,
                window_scaler=run.window_scaler,
                tda_scaler=run.tda_scaler,
            )
        else:
            from har.config.schema import Config, resolve_paths

            cfg = resolve_paths(Config())
            dataset_name = dataset_name or "synthetic"
            cfg.dataset.name = str(dataset_name)
            if config_override:
                known = {
                    "sampling_rate_override",
                    "subjects",
                    "channels",
                    "signal_mode",
                }
                for key, value in config_override.items():
                    if key in known:
                        setattr(cfg.dataset, key, value)
                    else:
                        raise ReplayError(
                            f"unsupported replay config key {key!r}; allowed: {sorted(known)}"
                        )
            predictor = None
        dataset_name = cfg.dataset.name
        svc = dataset_service()
        adapter, dataset = svc.load(cfg.dataset, cache=True)
        sampling_rate = float(
            cfg.dataset.sampling_rate_override or adapter.get_sampling_rate()
        )
        params = WindowParams.from_config(cfg, sampling_rate)
        preprocessor = Preprocessor(cfg, sampling_rate)
        processed = preprocessor.process_dataset(dataset)
        source = dataset_into_source(
            dataset, processed, params, max_windows=max_windows
        )
        subjects = list(dict.fromkeys([str(s) for s in source.subjects]))
        if subject:
            keep = {subject}
            mask = np.fromiter((str(s) in keep for s in source.subjects), dtype=bool, count=len(source.subjects))
            source = ReplaySource(
                windows=source.windows[mask],
                labels=source.labels[mask] if source.labels is not None else None,
                subjects=source.subjects[mask],
                window_ids=source.window_ids[mask],
                timestamps=source.timestamps[mask] if source.timestamps is not None else None,
                label_names=source.label_names,
            )
        return source, predictor, cfg.dataset.name, src_labels(source, dataset)

    # ------------------------------------------------------------------
    def start(
        self,
        run_id: str | None,
        dataset_name: str | None,
        config_override: dict[str, Any] | None = None,
        speed: float = 1.0,
        subject: str | None = None,
        max_windows: int | None = None,
    ) -> dict[str, Any]:
        if run_id is None and (dataset_name is None and config_override is None):
            raise ReplayError("either run_id or dataset must be specified")
        source, predictor, dataset_name, label_names = self._build_source(
            run_id, dataset_name, config_override, subject, max_windows
        )
        if len(source) == 0:
            raise ReplayError("replay would produce zero windows; widen the window config")
        session_id = uuid.uuid4().hex[:10]
        session = ReplaySession(
            session_id=session_id,
            source=source,
            predictor=predictor,
            label_names=label_names,
            dataset_name=dataset_name,
            speed=speed,
        )
        with self._lock:
            self._sessions[session_id] = session
        predictor.warm_up(source.windows[0]) if predictor is not None else None
        return {
            "session_id": session_id,
            "dataset": dataset_name,
            "n_windows": len(source),
            "subjects": list(dict.fromkeys([str(s) for s in source.subjects])),
            "speed": session.speed,
            "source": "replay(dataset=%s)" % dataset_name,
        }

    # ------------------------------------------------------------------
    def set_speed(self, session_id: str, speed: float) -> dict[str, Any]:
        session = self.get_session(session_id)
        session.speed = max(0.25, min(5.0, speed))
        return {"session_id": session_id, "speed": session.speed}

    def stop(self, session_id: str) -> dict[str, Any]:
        session = self.get_session(session_id)
        session.stop()
        with self._lock:
            self._sessions.pop(session_id, None)
        return {"session_id": session_id, "stopped": True}

    def get_session(self, session_id: str) -> ReplaySession:
        with self._lock:
            if session_id not in self._sessions:
                raise ReplayError(f"unknown replay session {session_id}")
            return self._sessions[session_id]

    def drain(self, session_id: str) -> list[dict[str, Any]]:
        session = self.get_session(session_id)
        out: list[dict[str, Any]] = []
        while True:
            try:
                out.append(session.events.get_nowait())
            except queue.Empty:
                break
        return out


def src_labels(source: ReplaySource, dataset) -> list[str]:
    return dataset.label_names


_manager = ReplayManager()


def replay_manager() -> ReplayManager:
    return _manager
