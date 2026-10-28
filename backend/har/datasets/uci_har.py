"""UCI HAR Dataset adapter (Anguita et al., 2013).

The official distribution contains:

* ``total_acc_*``, ``body_acc_*``, ``body_gyro_*`` inertial signals
  (triaxial, 50 Hz), **already cut into 2.56 s (128-sample) rows** with 50%
  overlap.  The rows of ``Inertial Signals/xxx_train/...`` correspond by row
  index to ``X_train.txt``/``y_train.txt``/``subject_train.txt``.
* Pre-engineered 561 feature vectors ``X_*.txt``.

Modes (``dataset.signal_mode``):

* ``inertial`` (default) -- the raw triaxial signals, 9 channels.  Windows
  are the dataset rows (identity windowing); an optional crop to a shorter
  window duration is supported for length ablations.  Re-cutting with a
  different overlap is NOT possible with this distribution (the raw,
  un-windowed stream is not distributed) and is refused loudly.
* ``engineered`` -- the 561 pre-engineered features, kept strictly separate;
  usable by the SVM baseline only.  TDA/CNN/LSTM models on this mode are
  rejected with a clear error.  No sensor values are fabricated.
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np

from ..config.schema import DatasetConfig
from ..utils.logging import get_logger
from .base import LoadedDataset, SubjectSeries, DatasetAdapter, register_adapter

logger = get_logger(__name__)

INERTIAL_CHANNEL_FILES = {
    "body_acc_x": "body_acc_x_{split}.txt",
    "body_acc_y": "body_acc_y_{split}.txt",
    "body_acc_z": "body_acc_z_{split}.txt",
    "body_gyro_x": "body_gyro_x_{split}.txt",
    "body_gyro_y": "body_gyro_y_{split}.txt",
    "body_gyro_z": "body_gyro_z_{split}.txt",
    "total_acc_x": "total_acc_x_{split}.txt",
    "total_acc_y": "total_acc_y_{split}.txt",
    "total_acc_z": "total_acc_z_{split}.txt",
}
INERTIAL_CHANNEL_ORDER = list(INERTIAL_CHANNEL_FILES.keys())

DEFAULT_ZIP_URL = (
    "https://archive.ics.uci.edu/static/public/240/"
    "human+activity+recognition+using+smartphones.zip"
)


def default_activity_labels() -> dict[int, str]:
    return {
        1: "WALKING",
        2: "WALKING_UPSTAIRS",
        3: "WALKING_DOWNSTAIRS",
        4: "SITTING",
        5: "STANDING",
        6: "LAYING",
    }


class UciHarError(RuntimeError):
    pass


@register_adapter
class UciHarAdapter(DatasetAdapter):
    name = "uci_har"
    sampling_rate = 50.0
    window_samples = 128  # 2.56 s at 50 Hz (dataset-provided rows)
    # the dataset rows were extracted with 50% overlap by the authors
    dataset_row_overlap = 0.5

    def __init__(self, cfg: DatasetConfig) -> None:
        self.cfg = cfg
        self.root = self._resolve_root(cfg.root)
        self.signal_mode = (cfg.signal_mode or "inertial").lower()
        if self.signal_mode not in ("inertial", "engineered"):
            raise UciHarError(
                f"uci_har supports signal_mode 'inertial' or 'engineered', got {cfg.signal_mode!r}"
            )
        self.requested_channels = cfg.channels
        self.subject_filter = set(cfg.subjects) if cfg.subjects else None
        self.label_map = default_activity_labels()

    def _resolve_root(self, root: str | None) -> Path:
        from ..utils.seeds import project_root

        base = Path(root) if root else project_root() / "data" / "raw" / "UCI HAR Dataset"
        if not base.exists():
            raise UciHarError(
                f"UCI HAR Dataset not found at {base}. "
                f"Run `python scripts/download_uci_har.py` first (source: {DEFAULT_ZIP_URL})."
            )
        return base

    # ------------------------------------------------------------------
    def load(self) -> LoadedDataset:
        if self.signal_mode == "engineered":
            return self._load_engineered()
        return self._load_inertial()

    # ------------------------------------------------------------------
    def _load_inertial(self) -> LoadedDataset:
        channel_selection = self._select_channels()
        labels = self._read_activity_labels()

        series: list[SubjectSeries] = []
        for split in ("train", "test"):
            data, subjects, y = self._read_split_inertial(split, channel_selection)
            for subject in np.unique(subjects):
                if self.subject_filter and str(subject) not in self.subject_filter:
                    continue
                mask = subjects == subject
                win = data[mask]
                lab = y[mask] - 1  # shipped 1-indexed; model expects 0-indexed
                series.append(
                    SubjectSeries(
                        subject_id=f"subject_{int(subject):02d}",
                        values=win.astype(np.float32),
                        labels=lab.astype(np.int64),
                        sampling_rate=self.sampling_rate,
                        channel_names=channel_selection,
                        kind="presegmented",
                        window_shape=(self.window_samples, len(channel_selection)),
                        extra={"split": split},
                    )
                )

        label_names = self._label_names(labels)
        notes = (
            "Raw triaxial inertial signals; dataset ships rows pre-cut into "
            "2.56 s / 128-sample windows with 50% overlap. Overlap is defined "
            "by the dataset and cannot be reconfigured honestly in this mode."
        )
        return LoadedDataset(
            name=self.name,
            sampling_rate=self.sampling_rate,
            channel_names=channel_selection,
            label_names=label_names,
            kind="presegmented",
            series=series,
            source_path=str(self.root),
            window_shape=(self.window_samples, len(channel_selection)),
            notes=notes,
            extra={"label_map": labels},
        )

    def _read_split_inertial(
        self, split: str, channels: list[str]
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        split_dir = self.root / split / "Inertial Signals"
        blocks = []
        for ch in channels:
            rel = INERTIAL_CHANNEL_FILES[ch].format(split=split)
            path = split_dir / rel
            arr = np.loadtxt(path, dtype=np.float32)
            if arr.ndim == 1:
                arr = arr[:, None]
            blocks.append(arr[:, :, None])
        data = np.concatenate(blocks, axis=2)  # (n_rows, T, C_file)
        y = np.loadtxt(self.root / split / f"y_{split}.txt", dtype=np.int64).ravel()
        subj = np.loadtxt(self.root / split / f"subject_{split}.txt", dtype=np.int64).ravel()
        if not (len(data) == len(y) == len(subj)):
            raise UciHarError(f"{split}: row count mismatch between files")
        if data.shape[1] != self.window_samples:
            raise UciHarError(
                f"{split}: expected {self.window_samples} samples per row, got {data.shape[1]}"
            )
        return data, subj, y

    def _select_channels(self) -> list[str]:
        available = INERTIAL_CHANNEL_ORDER
        if not self.requested_channels:
            return list(available)
        unknown = [c for c in self.requested_channels if c not in available]
        if unknown:
            raise UciHarError(
                f"unknown UCI HAR channels {unknown}; available {available}"
            )
        return [c for c in available if c in set(self.requested_channels)]

    def _label_names(self, labels: dict[int, str]) -> list[str]:
        order = [
            "WALKING",
            "WALKING_UPSTAIRS",
            "WALKING_DOWNSTAIRS",
            "SITTING",
            "STANDING",
            "LAYING",
        ]
        return order

    def _read_activity_labels(self) -> dict[int, str]:
        path = self.root / "activity_labels.txt"
        labels = default_activity_labels()
        if path.exists():
            text = path.read_text(encoding="utf-8")
            m = {}
            for line in text.splitlines():
                match = re.match(r"\s*(\d+)\s+([A-Za-z_]+)", line)
                if match:
                    m[int(match.group(1))] = match.group(2)
            if m:
                labels = m
        return labels

    # ------------------------------------------------------------------
    def _load_engineered(self) -> LoadedDataset:
        series: list[SubjectSeries] = []
        for split in ("train", "test"):
            X = np.loadtxt(self.root / split / f"X_{split}.txt", dtype=np.float32)
            y = np.loadtxt(self.root / split / f"y_{split}.txt", dtype=np.int64).ravel()
            subj = np.loadtxt(self.root / split / f"subject_{split}.txt", dtype=np.int64).ravel()
            for subject in np.unique(subj):
                if self.subject_filter and str(subject) not in self.subject_filter:
                    continue
                mask = subj == subject
                series.append(
                    SubjectSeries(
                        subject_id=f"subject_{int(subject):02d}",
                        values=X[mask].astype(np.float32),
                        labels=(y[mask] - 1).astype(np.int64),
                        sampling_rate=self.sampling_rate,
                        channel_names=[f"f{i}" for i in range(X.shape[1])],
                        kind="feature_rows",
                        extra={"split": split},
                    )
                )
        notes = (
            "Pre-engineered 561 features provided by the authors. NOT raw "
            "sensor data; TDA/window models must not be used in this mode."
        )
        return LoadedDataset(
            name=self.name,
            sampling_rate=self.sampling_rate,
            channel_names=[f"f{i}" for i in range(561)],
            label_names=list(default_activity_labels().values()),
            kind="feature_rows",
            series=series,
            source_path=str(self.root),
            notes=notes,
            extra={"label_map": self._read_activity_labels()},
        )

    # ------------------------------------------------------------------
    def get_subjects(self) -> list[str]:
        data = self.load()
        return data.subjects

    def get_sampling_rate(self) -> float:
        if self.cfg.sampling_rate_override:
            return self.cfg.sampling_rate_override
        return self.sampling_rate

    def get_sensor_columns(self) -> list[str]:
        return self._select_channels() if self.signal_mode == "inertial" else [
            f"f{i}" for i in range(561)
        ]

    def get_labels(self) -> list[str]:
        return self._label_names(self._read_activity_labels())
