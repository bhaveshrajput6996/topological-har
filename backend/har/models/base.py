"""Common model utilities (input shape checks, parameter counting)."""

from __future__ import annotations

import numpy as np

import torch
import torch.nn as nn


class ModelInputError(ValueError):
    pass


def count_parameters(module: nn.Module) -> int:
    return sum(p.numel() for p in module.parameters() if p.requires_grad)


def as_torch_tensor(values: np.ndarray, device: torch.device | None = None) -> torch.Tensor:
    tensor = torch.as_tensor(values.astype(np.float32, copy=False))
    if device is not None:
        tensor = tensor.to(device)
    return tensor
