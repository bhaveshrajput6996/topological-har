"""Torch training loop with early stopping, LR scheduling, checkpointing
and per-epoch history.  Supports both plain-window models (forward(x)) and
the fusion model (forward(x, tda)), depending on the dataset the caller
provides."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Iterator

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from ..utils.logging import get_logger, log_structured

logger = get_logger(__name__)


class WindowTorchDataset(Dataset):
    """Dataset over materialised numpy window arrays.

    ``deep_data`` shape (N, T, C); ``tda_data`` optionally (N, K) for the
    fusion branch.  TDA features are precomputed for speed; the TDA latency
    is measured separately in the profiler/predictor (it is deterministic
    per window).
    """

    def __init__(
        self,
        deep_data: np.ndarray,
        labels: np.ndarray,
        tda_data: np.ndarray | None = None,
    ) -> None:
        if len(deep_data) != len(labels):
            raise ValueError("deep_data/labels length mismatch")
        if tda_data is not None and len(tda_data) != len(deep_data):
            raise ValueError("tda_data length mismatch")
        self.deep_data = np.ascontiguousarray(deep_data.astype(np.float32, copy=False))
        self.labels = np.asarray(labels, dtype=np.int64)
        self.tda_data = (
            np.ascontiguousarray(tda_data.astype(np.float32, copy=False))
            if tda_data is not None
            else None
        )

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> tuple[Any, ...]:
        x = torch.from_numpy(self.deep_data[idx])
        y = int(self.labels[idx])
        if self.tda_data is not None:
            t = torch.from_numpy(self.tda_data[idx])
        else:
            t = torch.empty(0)
        return x, t, y


@dataclass
class EpochRecord:
    epoch: int
    train_loss: float
    val_loss: float | None
    train_accuracy: float
    val_accuracy: float | None
    epoch_time_s: float

    def as_dict(self) -> dict[str, Any]:
        return {
            "epoch": self.epoch,
            "train_loss": round(self.train_loss, 6),
            "val_loss": None if self.val_loss is None else round(self.val_loss, 6),
            "train_accuracy": round(self.train_accuracy, 6),
            "val_accuracy": None if self.val_accuracy is None else round(self.val_accuracy, 6),
            "epoch_time_s": round(self.epoch_time_s, 3),
        }


@dataclass
class TrainingResult:
    best_state: dict[str, torch.Tensor] | tuple[dict[str, torch.Tensor]] | object
    history: list[EpochRecord] = field(default_factory=list)
    best_val_loss: float = float("inf")
    best_epoch: int = 0
    training_time_s: float = 0.0
    early_stopped: bool = False
    device: str = "cpu"

    def history_dicts(self) -> list[dict[str, Any]]:
        return [e.as_dict() for e in self.history]


def _stats(loss_sum: float, n: int, correct: int) -> tuple[float, float]:
    if n == 0:
        return 0.0, 0.0
    return loss_sum / n, correct / n


class Trainer:
    """Fits a torch model (plain or fusion) with the configured schedule."""

    def __init__(self, cfg, device: torch.device | None = None) -> None:
        self.cfg = cfg.training
        self.device = device or self._resolve_device()
        self.uses_tda: bool | None = None

    @staticmethod
    def _resolve_device() -> torch.device:
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")

    # ------------------------------------------------------------------
    def fit(
        self,
        model: nn.Module,
        train: WindowTorchDataset,
        val: WindowTorchDataset | None,
        epochs: int | None = None,
        checkpoint_cb=None,
    ) -> TrainingResult:
        torch.manual_seed(self.cfg.seed)
        np.random.seed(self.cfg.seed)

        if len(train) == 0:
            raise ValueError("training dataset is empty")

        model = model.to(self.device)
        epochs = self.cfg.epochs if epochs is None else epochs
        probe = next(iter(torch.utils.data.DataLoader(train, batch_size=2, shuffle=False)))
        # probe is a collated [x, tda, y] batch; TDA data is element 1
        needs_tda = len(probe) == 3 and probe[1].numel() > 0
        self.uses_tda = needs_tda

        loader = DataLoader(
            train,
            batch_size=self.cfg.batch_size,
            shuffle=True,
            num_workers=self.cfg.num_workers,
            generator=torch.Generator().manual_seed(self.cfg.seed),
        )
        val_loader = (
            DataLoader(
                val,
                batch_size=self.cfg.batch_size,
                shuffle=False,
                num_workers=self.cfg.num_workers,
            )
            if val is not None
            else None
        )

        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=self.cfg.learning_rate,
            weight_decay=self.cfg.weight_decay,
        )
        scheduler = None
        if self.cfg.lr_scheduler == "plateau":
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode="min",
                factor=self.cfg.scheduler_factor,
                patience=self.cfg.scheduler_patience,
            )

        criterion = nn.CrossEntropyLoss()
        result = TrainingResult(best_state={}, device=str(self.device))

        t_total = time.perf_counter()
        n_no_improvement = 0
        for epoch in range(1, epochs + 1):
            t_epoch = time.perf_counter()
            train_loss, train_acc, _ = self._run_epoch(model, loader, criterion, optimizer, needs_tda)
            if val_loader is not None:
                val_loss, val_acc, _ = self._run_epoch(model, val_loader, criterion, None, needs_tda)
            else:
                val_loss, val_acc = None, None
            epoch_time = time.perf_counter() - t_epoch

            record = EpochRecord(
                epoch=epoch,
                train_loss=train_loss,
                val_loss=val_loss,
                train_accuracy=train_acc,
                val_accuracy=val_acc,
                epoch_time_s=epoch_time,
            )
            result.history.append(record)
            log_structured(
                logger,
                20,
                "epoch",
                epoch=epoch,
                train_loss=round(train_loss, 4),
                val_loss=None if val_loss is None else round(val_loss, 4),
                train_acc=round(train_acc, 4),
                val_acc=None if val_acc is None else round(val_acc, 4),
                seconds=round(epoch_time, 2),
            )

            monitor = val_loss if val_loss is not None else train_loss
            if scheduler is not None and val_loss is not None:
                scheduler.step(val_loss)
            elif scheduler is not None:
                scheduler.step(train_loss)

            is_best = monitor < result.best_val_loss - 1e-6
            if is_best:
                result.best_val_loss = monitor
                result.best_epoch = epoch
                result.best_state = {
                    k: v.detach().cpu().clone() for k, v in model.state_dict().items()
                }
                n_no_improvement = 0
                if checkpoint_cb is not None:
                    checkpoint_cb(result.best_state, epoch)
            else:
                n_no_improvement += 1
                if self.cfg.early_stopping and n_no_improvement >= self.cfg.patience:
                    result.early_stopped = True
                    log_structured(logger, 20, "early_stopping", epoch=epoch, best=result.best_epoch)
                    break

        result.training_time_s = time.perf_counter() - t_total
        if result.best_state:
            model.load_state_dict(result.best_state)
        return result

    # ------------------------------------------------------------------
    def _run_epoch(
        self,
        model: nn.Module,
        loader: DataLoader,
        criterion: nn.Module,
        optimizer: torch.optim.Optimizer | None,
        needs_tda: bool,
    ) -> tuple[float, float, int]:
        total_loss = 0.0
        n_samples = 0
        n_correct = 0
        training = optimizer is not None
        if training:
            model.train()
        else:
            model.eval()
        loader_iter: Iterator = iter(loader)
        with torch.set_grad_enabled(training):
            for x, tda, y in loader_iter:
                x = x.to(self.device, non_blocking=False)
                y = y.to(self.device, non_blocking=False)
                use_tda = needs_tda and tda.numel() > 0
                tda = tda.to(self.device) if use_tda else None
                logits = model(x, tda) if use_tda else model(x)
                loss = criterion(logits, y)
                if training:
                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    optimizer.step()
                total_loss += loss.item() * len(y)
                n_correct += int((logits.argmax(dim=1) == y).sum().item())
                n_samples += len(y)
        loss, acc = _stats(total_loss, n_samples, n_correct)
        return loss, acc, n_samples
