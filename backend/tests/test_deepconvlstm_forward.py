"""Model forward-pass shape tests (CPU)."""

import numpy as np
import torch

from har.config.schema import Config, ModelConfig
from har.models.deepconvlstm import DeepConvLstmEncoder, DeepConvLstmModel, build_head
from har.models.factory import build_model
from har.utils.seeds import seed_everything


def _cfg(**kwargs):
    c = ModelConfig(type="deepconvlstm", conv_channels=[16, 32, 32, 64], lstm_hidden=32)
    for k, v in kwargs.items():
        setattr(c, k, v)
    return c


def _config(model_cfg) -> Config:
    cfg = Config()
    cfg.model = model_cfg
    return cfg


def test_encoder_output_shape():
    seed_everything(0)
    enc = DeepConvLstmEncoder(n_channels=3, cfg=_cfg())
    x = torch.randn(4, 64, 3)
    feats = enc.forward_features(x)
    assert feats.shape == (4, 32)
    with torch.no_grad():
        _ = enc(x)  # x stays (B, T, C)


def test_deepconvlstm_logits_shape():
    seed_everything(1)
    model = DeepConvLstmModel(n_channels=9, n_classes=6, cfg=_cfg())
    x = torch.randn(2, 128, 9)
    logits = model(x)
    assert logits.shape == (2, 6)


def test_build_head_channels():
    seed_everything(2)
    head = build_head([256, 128], 6)
    inp = torch.randn(5, 256)
    assert head(inp).shape == (5, 6)


def test_fusion_model_requires_tda_forward():
    seed_everything(3)
    from har.models.fusion import TdaDeepConvLstmModel

    cfg = _cfg(type="tda_deepconvlstm", head_hidden=[64, 32])
    model_cfg = cfg
    from har.models.deepconvlstm import DeepConvLstmEncoder

    encoder = DeepConvLstmEncoder(n_channels=3, cfg=model_cfg)
    model = TdaDeepConvLstmModel(
        n_channels=3, n_classes=4, encoder=encoder, tda_feature_dim=400, head_hidden=[64, 32]
    )
    x = torch.randn(2, 64, 3)
    tda = torch.randn(2, 400)
    logits = model(x, tda)
    assert logits.shape == (2, 4)
    assert getattr(model, "tda_feature_dim") == 400


def test_factory_builds_expected_types():
    seed_everything(4)
    cfg = _config(_cfg(type="tda_deepconvlstm"))
    cfg.tda.enabled = True
    m = build_model(cfg, n_classes=6, n_channels=9, tda_feature_dim=400)
    from har.models.fusion import TdaDeepConvLstmModel

    assert isinstance(m, TdaDeepConvLstmModel)
    cfg2 = _config(_cfg(type="tda_deepconvlstm"))
    cfg2.tda.enabled = False  # degrade transparently to the baseline
    b = build_model(cfg2, n_classes=6, n_channels=9, tda_feature_dim=0)
    from har.models.deepconvlstm import DeepConvLstmModel

    assert isinstance(b, DeepConvLstmModel)


def test_rejects_wrong_input_rank():
    seed_everything(5)
    enc = DeepConvLstmEncoder(n_channels=3, cfg=_cfg())
    try:
        enc.forward_features(torch.randn(64, 3))
        raise AssertionError("expected ModelInputError")
    except Exception as exc:  # noqa: BLE001
        assert type(exc).__name__ == "ModelInputError"
