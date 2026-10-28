"""Train-only-fit normalisation using sklearn StandardScaler.

The scaler MUST be fitted on training subjects only.  This module simply
exposes a wrapper that works on window arrays shaped ``(n, T, C)`` or raw
series shaped ``(n, C)`` and can be persisted with the artifacts.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.preprocessing import StandardScaler


class ScalerWrapper:
    """Thin wrapper adding save/load and multi-dimensional handling."""

    def __init__(self, enabled: bool = True) -> None:
        self.enabled = enabled
        self.scaler: StandardScaler | None = StandardScaler() if enabled else None
        self.n_features: int | None = None
        self.stats: dict[str, Any] = {}

    # -- shapes ------------------------------------------------------------
    @staticmethod
    def _as_2d(values: np.ndarray) -> np.ndarray:
        if values.ndim <= 1:
            return values.reshape(-1, 1)
        return values.reshape(-1, values.shape[-1])

    # -- api ---------------------------------------------------------------
    def fit(self, values: np.ndarray) -> "ScalerWrapper":
        if not self.enabled:
            return self
        if len(values) == 0 or values[0].size == 0:
            raise ValueError("cannot fit scaler on empty data")
        self.scaler.fit(self._as_2d(values))
        self.n_features = int(values.shape[-1])
        return self

    def transform(self, values: np.ndarray) -> np.ndarray:
        if not self.enabled:
            return values
        if len(values) == 0:
            return values
        assert self.scaler is not None
        if getattr(self.scaler, "mean_", None) is None:
            raise ValueError("scaler was never fitted; fit on training data first")
        out = self.scaler.transform(self._as_2d(values))
        return out.reshape(values.shape).astype(np.float32, copy=False)

    def fit_transform(self, values: np.ndarray) -> np.ndarray:
        self.fit(values)
        return self.transform(values)

    def get_state(self) -> dict[str, Any]:
        if not self.enabled or self.scaler is None:
            return {"enabled": False}
        return {
            "enabled": True,
            "mean": self.scaler.mean_.tolist(),
            "scale": self.scaler.scale_.tolist(),
            "var": self.scaler.var_.tolist(),
            "n_features": self.n_features,
        }

    def save(self, path) -> None:
        import pickle

        with open(path, "wb") as fh:
            pickle.dump({"enabled": self.enabled, "scaler": self.scaler}, fh, protocol=4)

    @classmethod
    def load(cls, path) -> "ScalerWrapper":
        import pickle

        with open(path, "rb") as fh:
            payload = pickle.load(fh)  # noqa: S301 (artifacts are trusted files)
        wrapper = cls(enabled=bool(payload.get("enabled", True)))
        wrapper.scaler = payload.get("scaler")
        return wrapper
