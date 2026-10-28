"""DeepConvLSTM deep branch (Ordóñez & Roggen 2016, adapted to 1-D conv
over time).  The window tensor is ``(batch, T, C)``; convolutions run over
the time axis for every channel.

Architecture (report + prompt baseline):

  4 x [Conv1d -> BatchNorm -> ReLU] (C -> 64 -> 128 -> 128 -> 256, kernel 5)
  2 x LSTM(hidden=128) with dropout 0.5 after each LSTM
  -> deep feature vector (batch, 128)

Nothing is hardcoded to a specific (T, C): all sizes come from the config.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from ..config.schema import ModelConfig
from .base import ModelInputError


class DeepConvLstmEncoder(nn.Module):
    """Conv + LSTM stack producing the deep feature vector."""

    def __init__(self, n_channels: int, cfg: ModelConfig) -> None:
        super().__init__()
        if len(cfg.conv_channels) != 4:
            raise ModelInputError("model.conv_channels must contain exactly 4 sizes")
        convs: list[nn.Module] = []
        convs.append(nn.Conv1d(n_channels, cfg.conv_channels[0], cfg.conv_kernel_size, padding=cfg.conv_kernel_size // 2))
        for i in range(1, 4):
            convs.append(nn.Conv1d(cfg.conv_channels[i - 1], cfg.conv_channels[i], cfg.conv_kernel_size, padding=cfg.conv_kernel_size // 2))
        self.conv_block = nn.ModuleList(
            [nn.Sequential(c, nn.BatchNorm1d(c.out_channels), nn.ReLU()) for c in convs]
        )
        self.lstm1 = nn.LSTM(cfg.conv_channels[3], cfg.lstm_hidden, batch_first=True)
        self.lstm2 = nn.LSTM(cfg.lstm_hidden, cfg.lstm_hidden, batch_first=True)
        self.dropout = nn.Dropout(cfg.dropout)
        self.output_dim = cfg.lstm_hidden

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 3:
            raise ModelInputError(f"expected (batch, T, C), got {tuple(x.shape)}")
        h = x.transpose(1, 2)  # (B, C, T)
        for block in self.conv_block:
            h = block(h)
        h = h.transpose(1, 2)  # (B, T, F)
        out, _ = self.lstm1(h)
        out = self.dropout(out)
        out, _ = self.lstm2(out)
        out = self.dropout(out)
        return out[:, -1, :]  # last time step (B, hidden)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward_features(x)


def build_head(hidden_sizes: list[int], n_classes: int) -> nn.Module:
    layers: list[nn.Module] = []
    last = hidden_sizes[0]
    for size in hidden_sizes[1:]:
        layers += [nn.Linear(last, size), nn.ReLU(), nn.Dropout(0.3)]
        last = size
    layers.append(nn.Linear(last, n_classes))
    return nn.Sequential(*layers)


class DeepConvLstmModel(nn.Module):
    """Pure DeepConvLSTM baseline classifier (no TDA branch)."""

    def __init__(self, n_channels: int, n_classes: int, cfg: ModelConfig) -> None:
        super().__init__()
        self.encoder = DeepConvLstmEncoder(n_channels, cfg)
        self.head = nn.Linear(cfg.lstm_hidden, n_classes)

    def forward(self, x: torch.Tensor, tda_features: torch.Tensor | None = None) -> torch.Tensor:
        feats = self.encoder.forward_features(x)
        return self.head(feats)
