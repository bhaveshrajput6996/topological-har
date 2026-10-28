"""Fusion model: TDA features + DeepConvLSTM features -> Dense 256 ->
Dense 128 -> logits.  Output classes are always derived from the dataset."""

from __future__ import annotations

import torch
import torch.nn as nn

from .deepconvlstm import DeepConvLstmEncoder, build_head


class TdaDeepConvLstmModel(nn.Module):
    """The proposed two-branch model ('TDA + DeepConvLSTM')."""

    def __init__(
        self,
        n_channels: int,
        n_classes: int,
        encoder: DeepConvLstmEncoder,
        tda_feature_dim: int,
        head_hidden: list[int],
    ) -> None:
        super().__init__()
        self.encoder = encoder
        fused_dim = encoder.output_dim + tda_feature_dim
        self.head = build_head([fused_dim, *head_hidden], n_classes)
        self.tda_feature_dim = tda_feature_dim

    def forward(
        self,
        x: torch.Tensor,
        tda_features: torch.Tensor | None = None,
    ) -> torch.Tensor:
        deep = self.encoder.forward_features(x)
        if tda_features is None:
            raise ValueError(
                "TdaDeepConvLstmModel requires the TDA branch input "
                "(`tda_features`), got None"
            )
        if tda_features.shape[1] != self.tda_feature_dim:
            raise ValueError(
                f"TDA branch expects {self.tda_feature_dim} features, "
                f"got {tuple(tda_features.shape)}"
            )
        fused = torch.cat([deep, tda_features], dim=1)
        return self.head(fused)

    def forward_with_breakdown(
        self, x: torch.Tensor, tda_features: torch.Tensor
    ) -> tuple[torch.Tensor, float, float, float]:
        """Forward pass returning the branch latency breakdown (seconds)."""
        t0 = torch.cuda.Event(enable_timing=True) if torch.cuda.is_available() else None

        import time as _time

        t_start = _time.perf_counter()
        deep = self.encoder.forward_features(x)
        t_deep = _time.perf_counter() - t_start

        t_start = _time.perf_counter()
        tda_vector = tda_features
        fused = torch.cat([deep, tda_vector], dim=1)
        logits = self.head(fused)
        t_fusion = _time.perf_counter() - t_start
        t_total = t_deep + t_fusion
        return logits, t_deep, t_fusion, t_total
