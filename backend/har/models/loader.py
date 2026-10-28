"""Artifact loading: rebuild a runnable model bundle from artifacts/{run_id}."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml

from ..config.schema import (
    Config,
    TdaConfig,
    config_from_dict,
    parse_tda,
)
from ..models.factory import build_model
from ..models.svm import SklearnSvm
from ..preprocessing.normalization import ScalerWrapper
from ..tda.engine import TdaEngine
from ..utils.logging import get_logger

logger = get_logger(__name__)


class ArtifactError(RuntimeError):
    pass


@dataclass
class LoadedRun:
    run_id: str
    artifacts_dir: Path
    cfg: Config
    pipeline: dict[str, Any]
    model: Any  # torch nn.Module or SklearnSvm
    window_scaler: ScalerWrapper
    tda_scaler: ScalerWrapper | None
    tda_engine: TdaEngine | None
    label_names: list[str]
    window_shape: list[int]
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def is_svm(self) -> bool:
        return isinstance(self.model, SklearnSvm)

    @property
    def uses_tda(self) -> bool:
        return bool(self.pipeline.get("tda_enabled")) and not self.is_svm


def find_run_dir(artifacts_root: str | Path, run_id: str) -> Path:
    path = Path(artifacts_root) / run_id
    if not (path / "pipeline.json").exists():
        raise ArtifactError(
            f"artifacts for run_id={run_id} not found under {artifacts_root} "
            "(needed: pipeline.json). Train the run first."
        )
    return path


def load_tda_config(config_yaml: dict[str, Any]) -> TdaConfig:
    return parse_tda(config_yaml.get("tda") or {})


def load_run(artifacts_root: str | Path, run_id: str) -> LoadedRun:
    run_dir = find_run_dir(artifacts_root, run_id)
    pipeline = (run_dir / "pipeline.json").read_text(encoding="utf-8")
    import json

    pipeline = json.loads(pipeline)
    cfg_path = run_dir / "config.yaml"
    if not cfg_path.exists():
        raise ArtifactError(f"config.yaml missing in {run_dir}")
    cfg: Config = config_from_dict(yaml.safe_load(cfg_path.read_text(encoding="utf-8")))
    # artifacts' config contains absolute data paths from the machine that
    # produced it; runtime paths stay untouched by the loader
    n_classes = len(pipeline["label_names"])
    window_shape = pipeline["window_shape"]
    n_channels = int(window_shape[-1])
    tda_dim = int(pipeline.get("tda_feature_dim", 0) or 0)
    window_scaler = ScalerWrapper.load(run_dir / "model.scaler.pkl")

    if pipeline["model_type"] == "svm":
        model = SklearnSvm(cfg)
        model.load(run_dir / "model.pkl")
        tda_scaler = None
        tda_engine = None
    else:
        model_type = pipeline["model_type"]
        cfg.model.type = model_type
        tda_enabled = bool(pipeline.get("tda_enabled"))
        if model_type == "tda_deepconvlstm" and not tda_enabled:
            cfg.tda.enabled = False
        model = build_model(cfg, n_classes, n_channels, tda_dim)
        state = torch.load(
            run_dir / "model.pt", map_location="cpu", weights_only=True
        )
        model.load_state_dict(state["model_state"])
        tda_scaler = None
        tda_engine = None
        if tda_enabled:
            if tda_dim == 0:
                raise ArtifactError(
                    "pipeline says TDA enabled but tda_feature_dim==0; "
                    "artifact bundle inconsistent"
                )
            tda_engine = TdaEngine(cfg.tda)
            scaler_path = run_dir / "tda_scaler.pkl"
            if scaler_path.exists():
                tda_scaler = ScalerWrapper.load(scaler_path)

    return LoadedRun(
        run_id=run_id,
        artifacts_dir=run_dir,
        cfg=cfg,
        pipeline=pipeline,
        model=model,
        window_scaler=window_scaler,
        tda_scaler=tda_scaler,
        tda_engine=tda_engine,
        label_names=list(pipeline["label_names"]),
        window_shape=[int(x) for x in window_shape],
    )


def soft_load_eval(model) -> None:
    model.eval()
