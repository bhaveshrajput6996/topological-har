"""SVM baseline on flattened windows (sklearn RBF SVC).

Inference latency, training time and per-window probabilities are measured
exactly like the torch models so all models remain comparable.  Training
set is capped by ``evaluation.svm_max_train_windows`` because SVC is
O(n^2)-ish in memory.
"""

from __future__ import annotations

import time
import types
from dataclasses import dataclass

import numpy as np

from ..config.schema import Config
from ..preprocessing.normalization import ScalerWrapper


@dataclass
class SvmPrediction:
    probabilities: np.ndarray
    predicted: np.ndarray


class SklearnSvm:
    """Wraps ``sklearn.svm.SVC`` for the (N, T*C) window representation."""

    def __init__(self, cfg: Config) -> None:
        self.cfg = cfg
        self.svc = None
        self.scaler = ScalerWrapper(enabled=cfg.preprocessing.normalization == "standard")

    @property
    def model_name(self) -> str:
        return "SVM"

    # -- training --------------------------------------------------------
    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
    ) -> dict[str, float]:
        from sklearn.svm import SVC
        from sklearn.base import clone

        limit = min(len(X), self.cfg.evaluation.svm_max_train_windows)
        if limit < len(X):
            rng = np.random.default_rng(0)
            idx = np.sort(rng.choice(len(X), size=limit, replace=False))
            X, y = X[idx], y[idx]
        X = self.scaler.fit_transform(np.asarray(X, dtype=np.float32).reshape(len(X), -1))
        y = np.asarray(y)
        svc = SVC(
            C=self.cfg.evaluation.svm_c,
            gamma=self.cfg.evaluation.svm_gamma,
            probability=True,
            random_state=self.cfg.training.seed,
            cache_size=200,
        )
        t0 = time.perf_counter()
        svc.fit(X, y)
        self.svc = svc
        return {"training_time_s": time.perf_counter() - t0, "n_train_windows": len(X)}

    # -- inference -------------------------------------------------------
    def predict_proba(self, X: np.ndarray, batch: int = 256) -> tuple[np.ndarray, np.ndarray]:
        X = np.asarray(X, dtype=np.float32).reshape(len(X), -1)
        probs: list[np.ndarray] = []
        preds: list[np.ndarray] = []
        for start in range(0, len(X), batch):
            chunk = self.scaler.transform(X[start : start + batch])
            p = self.svc.predict_proba(chunk)
            probs.append(p)
            preds.append(np.argmax(p, axis=1))
        return np.vstack(probs), np.concatenate(preds)

    def save(self, path) -> None:
        import pickle

        with open(path, "wb") as fh:
            pickle.dump(self.svc, fh, protocol=4)
        self.scaler.save(str(path) + ".scaler.pkl")

    def load(self, path) -> None:
        import pickle

        with open(path, "rb") as fh:
            self.svc = pickle.load(fh)  # noqa: S301
        try:
            self.scaler = ScalerWrapper.load(str(path) + ".scaler.pkl")
        except FileNotFoundError:
            pass
