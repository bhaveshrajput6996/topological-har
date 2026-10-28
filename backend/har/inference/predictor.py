"""Per-window predictor with branch-latency breakdown.

Rules enforced here:

* confidence and correctness are independent: ``confidence`` is the softmax
  probability of the predicted class; ``correct`` is only set when ground
  truth exists (never fabricated);
* TDA latency, deep-branch latency, fusion/classifier latency and total
  latency are measured separately in ONE pass (no double computation), so
  bottleneck reports are honest.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np
import torch

from ..models.base import ModelInputError
from ..tda.engine import TdaEngine, TdaEngineError


class PredictorError(RuntimeError):
    pass


@dataclass
class WindowPredictionResult:
    predicted_idx: int
    probabilities: np.ndarray
    confidence: float
    tda_ms: float
    deep_ms: float
    fusion_ms: float
    total_ms: float
    tda_info: dict = field(default_factory=dict)

    def as_dict(self) -> dict:
        return {
            "predicted_label_idx": self.predicted_idx,
            "confidence": round(self.confidence, 6),
            "probabilities": [round(float(p), 6) for p in self.probabilities],
            "inference_time_ms": round(self.total_ms, 3),
            "tda_ms": round(self.tda_ms, 3),
            "deep_ms": round(self.deep_ms, 3),
            "fusion_ms": round(self.fusion_ms, 3),
        }


class WindowPredictor:
    """Preprocessing scalers + TDA branch + deep branch + classifier head."""

    def __init__(
        self,
        model: torch.nn.Module,
        tda_engine: TdaEngine | None,
        window_scaler=None,
        tda_scaler=None,
        device: torch.device | None = None,
        warmup: int = 5,
    ) -> None:
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        self.model.eval()
        self.tda_engine = tda_engine
        self.window_scaler = window_scaler
        self.tda_scaler = tda_scaler
        self._warmup = warmup
        self._warmed = False

    # ------------------------------------------------------------------
    @property
    def uses_tda(self) -> bool:
        return getattr(self.model, "tda_feature_dim", 0) > 0

    def _prepare_window(self, window: np.ndarray) -> np.ndarray:
        window = np.asarray(window, dtype=np.float32)
        if window.ndim == 2:
            window = window[None, ...]
        if window.ndim != 3:
            raise PredictorError(f"expected window (T, C), got shape {np.shape(window)}")
        if self.window_scaler is not None and getattr(self.window_scaler, "enabled", False):
            window = self.window_scaler.transform(window)
        return window

    def compute_tda(
        self, ref_window: np.ndarray, window: np.ndarray | None = None
    ) -> tuple[torch.Tensor, float, dict]:
        """Compute TDA features for one (T, C) window.

        ``ref_window`` must be the same coordinate window we feed the deep
        branch (with the preprocessing scaler applied, matching training).
        Returns the 1xD feature tensor, elapsed ms and diagnostics.
        """
        if self.tda_engine is None:
            return torch.empty(0), 0.0, {}
        ref_window = np.asarray(ref_window, dtype=np.float32)
        if ref_window.ndim == 3:
            ref_window = ref_window[0]
        t_start = time.perf_counter()
        result = self.tda_engine.compute(ref_window)
        vector = result.feature_vector[None, :].astype(np.float32)
        if self.tda_scaler is not None:
            vector = self.tda_scaler.transform(vector)
        elapsed = (time.perf_counter() - t_start) * 1000.0
        return torch.from_numpy(vector).to(self.device), elapsed, result.feature_dict()

    # ------------------------------------------------------------------
    def warm_up(self, window: np.ndarray) -> None:
        """Run a few isolated forward passes so first-call overhead does not
        pollute the latency measurement."""
        if self._warmed:
            return
        for _ in range(max(1, self._warmup)):
            try:
                self.predict_window(window)
            except (TdaEngineError, ModelInputError):
                return
        self._warmed = True

    # ------------------------------------------------------------------
    def predict_window(self, window: np.ndarray) -> WindowPredictionResult:
        window_scaled = self._prepare_window(window)
        with torch.no_grad():
            x = torch.from_numpy(window_scaled).to(self.device)

            tda_vector, tda_ms, tda_info = (
                self.compute_tda(window_scaled[0]) if self.uses_tda else (torch.empty(0), 0.0, {})
            )

            t0 = time.perf_counter()
            deep: torch.Tensor | None = None
            if self.uses_tda:
                t_deep0 = time.perf_counter()
                deep = self.model.encoder.forward_features(x)
                t_deep = time.perf_counter() - t_deep0
                t_fus0 = time.perf_counter()
                logits = self.model.head(torch.cat([deep, tda_vector], dim=1))
                t_fusion = time.perf_counter() - t_fus0
            else:
                t_whole0 = time.perf_counter()
                logits = self.model(x)
                t_whole = time.perf_counter() - t_whole0
                t_deep, t_fusion = t_whole, 0.0
            deep_ms = t_deep * 1000.0
            fusion_ms = t_fusion * 1000.0
            wall_ms = (time.perf_counter() - t0) * 1000.0

            probs = torch.softmax(logits, dim=1)[0]

        total_ms = (tda_ms + deep_ms + fusion_ms) if (deep_ms > 0 or fusion_ms > 0) else tda_ms + wall_ms
        return WindowPredictionResult(
            predicted_idx=int(probs.argmax().item()),
            probabilities=probs.cpu().numpy(),
            confidence=float(probs.max().item()),
            tda_ms=tda_ms,
            deep_ms=deep_ms,
            fusion_ms=fusion_ms,
            total_ms=total_ms,
            tda_info=tda_info,
        )

    # -- batched evaluation path -----------------------------------------
    def evaluate_batch(
        self,
        windows: np.ndarray,
        tda_features: np.ndarray | None,
        batch_size: int = 256,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Fast batched predictions for compact metric computation."""
        probs: list[np.ndarray] = []
        with torch.no_grad():
            for start in range(0, len(windows), batch_size):
                chunk = torch.from_numpy(
                    np.ascontiguousarray(windows[start : start + batch_size], dtype=np.float32)
                ).to(self.device)
                tda = None
                if tda_features is not None:
                    tda = torch.from_numpy(
                        np.ascontiguousarray(
                            tda_features[start : start + batch_size], dtype=np.float32
                        )
                    ).to(self.device)
                logits = self.model(chunk, tda) if tda is not None else self.model(chunk)
                probs.append(torch.softmax(logits, dim=1).cpu().numpy())
        out = np.vstack(probs)
        return out.argmax(axis=1), out
