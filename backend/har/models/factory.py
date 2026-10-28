"""Model registry: build torch / sklearn models from configuration."""

from __future__ import annotations

import numpy as np

from ..config.schema import Config, ModelConfig
from .base import ModelInputError
from .baselines import CnnModel, LstmModel
from .deepconvlstm import DeepConvLstmModel
from .fusion import TdaDeepConvLstmModel

MODEL_TYPES = {
    "tda_deepconvlstm",
    "deepconvlstm",
    "cnn",
    "lstm",
    "svm",
}


def build_model(
    cfg: Config,
    n_classes: int,
    n_channels: int,
    tda_feature_dim: int,
) -> object:
    """Instantiate a model for the configured type.

    Returns an ``nn.Module`` with signature ``forward(x, tda_features=None)``
    for torch models, or an ``SklearnSvm`` for the SVM baseline.
    """
    if n_classes < 2:
        raise ModelInputError(f"n_classes must be >= 2, got {n_classes}")
    if cfg.model.type == "svm":
        from .svm import SklearnSvm

        return SklearnSvm(cfg)
    if cfg.model.type == "tda_deepconvlstm":
        if not cfg.tda.enabled:
            # tda disabled via config -> degrade transparently to the baseline
            from .deepconvlstm import DeepConvLstmModel

            return DeepConvLstmModel(n_channels, n_classes, cfg.model)
        from .deepconvlstm import DeepConvLstmEncoder

        encoder = DeepConvLstmEncoder(n_channels, cfg.model)
        return TdaDeepConvLstmModel(
            n_channels=n_channels,
            n_classes=n_classes,
            encoder=encoder,
            tda_feature_dim=tda_feature_dim,
            head_hidden=cfg.model.head_hidden,
        )

    model_cfg: ModelConfig = cfg.model
    if model_cfg.type == "deepconvlstm":
        return DeepConvLstmModel(n_channels, n_classes, model_cfg)
    if model_cfg.type == "cnn":
        return CnnModel(n_channels, n_classes, model_cfg)
    if model_cfg.type == "lstm":
        return LstmModel(n_channels, n_classes, model_cfg)
    raise ModelInputError(f"unknown model type {model_cfg.type!r}")
