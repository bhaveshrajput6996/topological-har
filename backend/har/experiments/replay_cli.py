"""Terminal replay: walk dataset windows through a trained model and print
predictions + latency (REST/WebSocket replay is available through the API)."""

from __future__ import annotations

import json
import time
from typing import Any

from ..config.schema import Config, DatasetConfig, config_from_dict
from ..datasets import get_adapter
from ..models.loader import load_run
from ..preprocessing.pipeline import Preprocessor
from ..utils.logging import get_logger
from ..utils.seeds import project_root
from ..windowing.windowing import WindowParams, build_window_arrays

logger = get_logger(__name__)


def run_replay_cli(
    run_id: str | None,
    dataset_name: str | None,
    limit: int = 25,
    speed: float = 4.0,
) -> None:
    cfg: Config
    if run_id:
        run = load_run(project_root() / "artifacts", run_id)
        cfg = run.cfg
    else:
        cfg = config_from_dict({"dataset": {"name": dataset_name or "synthetic"}})
        dataset_name = cfg.dataset.name
        run = None

    adapter = get_adapter(cfg.dataset.name, cfg.dataset)
    dataset = adapter.load()
    sampling_rate = float(
        cfg.dataset.sampling_rate_override or adapter.get_sampling_rate()
    )
    params = WindowParams.from_config(cfg, sampling_rate)
    preprocessor = Preprocessor(cfg, sampling_rate)
    processed = preprocessor.process_dataset(dataset)

    from ..inference.predictor import WindowPredictor

    predictor = None
    if run is not None:
        predictor = WindowPredictor(
            model=run.model,
            tda_engine=run.tda_engine,
            window_scaler=run.window_scaler,
            tda_scaler=run.tda_scaler,
        )

    arrays = build_window_arrays(processed, params, max_windows=None)
    n = min(limit, len(arrays))
    delay = max(0.0, 1.0 / max(speed, 1e-6) * (params.window_duration) if predictor else 0.0)

    from ..windowing.windowing import windows_stream

    count = 0
    for window, label, sid, w_index, t in windows_stream(processed, params):
        if count >= n:
            break
        count += 1
        out: dict[str, Any] = {
            "asof": round(t, 2),
            "subject": sid,
            "window_id": w_index,
            "dataset": cfg.dataset.name,
        }
        if dataset.label_names and label is not None:
            out["truth"] = dataset.label_names[label]
        else:
            out["truth"] = None
        if predictor is not None:
            result = predictor.predict_window(window)
            out["pred"] = run.label_names[result.predicted_idx]
            out["conf"] = round(result.confidence, 3)
            out["ms"] = round(result.total_ms, 1)
            out["tda_ms"] = round(result.tda_ms, 1)
        print(json.dumps(out, ensure_ascii=False))
        if delay:
            time.sleep(delay)
    print(f"replay done: {count} windows")
