"""CNN and LSTM baselines."""

from __future__ import annotations

import torch
import torch.nn as nn

from ..config.schema import ModelConfig
from .base import ModelInputError


class CnnModel(nn.Module):
    """Classic 2-layer CNN + pooled FC baseline."""

    def __init__(self, n_channels: int, n_classes: int, cfg: ModelConfig) -> None:
        super().__init__()
        blocks: list[nn.Module] = []
        in_ch = n_channels
        sizes = list(cfg.conv_channels)[:2] or [64, 128]
        for out_ch in sizes:
            blocks += [
                nn.Conv1d(in_ch, out_ch, cfg.conv_kernel_size, padding=cfg.conv_kernel_size // 2),
                nn.BatchNorm1d(out_ch),
                nn.ReLU(),
                nn.MaxPool1d(2),
            ]
            in_ch = out_ch
        self.features = nn.Sequential(*blocks)
        with torch.no_grad():
            pooled = self.features(torch.zeros(1, n_channels, 8)).shape[1]
        self.flatten_dim = pooled
        self.classifier = nn.Sequential(
            nn.Linear(self.flatten_dim, cfg.head_hidden[-1] if cfg.head_hidden else 128),
            nn.ReLU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.head_hidden[-1] if cfg.head_hidden else 128, n_classes),
        )

    def forward(self, x: torch.Tensor, tda_features: torch.Tensor | None = None) -> torch.Tensor:
        h = self.features(x.transpose(1, 2))
        h = h.max(dim=2).values
        return self.classifier(h)


class LstmModel(nn.Module):
    """Stacked-LSTM baseline."""

    def __init__(self, n_channels: int, n_classes: int, cfg: ModelConfig) -> None:
        super().__init__()
        self.lstm = nn.LSTM(n_channels, cfg.lstm_hidden, num_layers=2, batch_first=True,
                            dropout=cfg.dropout)
        self.classifier = nn.Linear(cfg.lstm_hidden, n_classes)

    def forward(self, x: torch.Tensor, tda_features: torch.Tensor | None = None) -> torch.Tensor:
        out, _ = self.lstm(x)
        return self.classifier(out[:, -1, :])
