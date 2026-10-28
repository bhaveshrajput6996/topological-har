"""Synthetic HAR dataset generator.

Used by the automated tests and for smoke runs without any real dataset.
Deterministic given ``seed``; can optionally be persisted to a root path so
adapter IO is also exercised.

Generated signals have deliberate topological structure so that the TDA
branch can be exercised meaningfully in tests:

* ``WALKING``  -- sensor values trace a closed loop in sensor space
                  (an H1 feature appears).
* ``SITTING``  -- a compact cluster around a fixed point (H0 persistence).
* ``STANDING`` -- a slow small loop of different radius/frequency.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from ..config.schema import DatasetConfig
from ..utils.seeds import project_root
from .base import LoadedDataset, SubjectSeries, DatasetAdapter, register_adapter

SAMPLING_RATE = 32.0

LABELS = ["SITTING", "STANDING", "WALKING"]

LABEL_NAMES = "labels.npy"
VALUES_NAME = "values.npz"


class SyntheticError(RuntimeError):
    pass


def generate_shape(kind: str, n_samples: int, channels: int, rng: np.random.Generator) -> np.ndarray:
    """Generate one continuous activity episode in sensor space."""
    t = np.arange(n_samples, dtype=np.float64)
    if kind == "WALKING":
        freq = 2.0 * np.pi / 24.0  # period = 24 samples so loops fit in windows
        base = np.empty((n_samples, channels))
        for c in range(channels):
            base[:, c] = np.sin(freq * t + c * (2.0 * np.pi / channels))
        return base + rng.normal(0.0, 0.25, base.shape)
    if kind == "STANDING":
        freq = 2.0 * np.pi / 96.0
        base = np.empty((n_samples, channels))
        for c in range(channels):
            base[:, c] = 0.5 * np.sin(freq * t + c) + 0.5 * np.cos(freq * t)
        return base + rng.normal(0.0, 0.15, base.shape)
    # SITTING: compact cluster
    center = rng.normal(0.0, 1.0, channels)
    return center[None, :] + rng.normal(0.0, 0.4, (n_samples, channels))


def make_dataset(
    subjects: int = 2,
    channels: int = 3,
    samples_per_subject: int = 640,
    sampling_rate: float = SAMPLING_RATE,
    seed: int = 0,
) -> LoadedDataset:
    if channels < 2:
        raise SyntheticError("synthetic dataset needs >= 2 channels")
    if samples_per_subject < 64:
        raise SyntheticError("synthetic dataset needs >= 64 samples per subject")

    rng = np.random.default_rng(seed)
    episode_samples = max(64, samples_per_subject // len(LABELS))
    series: list[SubjectSeries] = []
    for subject in range(subjects):
        chunks = []
        labels = []
        for idx, kind in enumerate(LABELS):
            n = episode_samples
            data = generate_shape(kind, n, channels, rng)
            chunks.append(data)
            labels.append(np.full(n, idx, dtype=np.int64))
        values = np.concatenate(chunks).astype(np.float32)
        labs = np.concatenate(labels)
        series.append(
            SubjectSeries(
                subject_id=f"synthetic_s{subject:02d}",
                values=values,
                labels=labs,
                sampling_rate=sampling_rate,
                channel_names=[f"ch_{c}" for c in range(channels)],
                kind="continuous",
                timestamps=np.arange(values.shape[0], dtype=np.float64) / sampling_rate,
                extra={"generated_seed": seed, "episode_samples": episode_samples},
            )
        )

    return LoadedDataset(
        name="synthetic",
        sampling_rate=sampling_rate,
        channel_names=[f"ch_{c}" for c in range(channels)],
        label_names=list(LABELS),
        kind="continuous",
        series=series,
        source_path="generated",
        notes="Synthetic topologically-structured data for tests and smoke runs.",
        extra={"seed": seed},
    )


def save_dataset(dataset: LoadedDataset, root: Path) -> None:
    root.mkdir(parents=True, exist_ok=True)
    arrs = {}
    label_names = np.array(dataset.label_names, dtype=object)
    for i, s in enumerate(dataset.series):
        arrs[f"values_{i}"] = s.values
        arrs[f"labels_{i}"] = s.labels
    np.savez_compressed(root / VALUES_NAME, **arrs)
    np.save(root / LABEL_NAMES, np.array(dataset.label_names, dtype=object), allow_pickle=True)
    meta = {
        "sampling_rate": dataset.sampling_rate,
        "channel_names": dataset.channel_names,
        "kind": dataset.kind,
        "n_series": len(dataset.series),
    }
    import json

    (root / "meta.json").write_text(json.dumps(meta), encoding="utf-8")


@register_adapter
class SyntheticAdapter(DatasetAdapter):
    name = "synthetic"

    def __init__(self, cfg: DatasetConfig) -> None:
        self.cfg = cfg
        root = cfg.root
        self.root: Path | None
        self.root = Path(root).expanduser() if root else None
        self.subjects_requested = cfg.subjects
        self.quality = cfg.sampling_rate_override or SAMPLING_RATE

        base = project_root() / "data" / "raw" / "synthetic"
        if self.root is None and (base / VALUES_NAME).exists():
            self.root = base

    def load(self) -> LoadedDataset:
        dataset = make_dataset(seed=self.cfg_seed())
        if self.root and self.root.exists():
            # persisted synthetic data (same content, loaded via IO path)
            return self._load_from_root()
        return dataset

    def cfg_seed(self) -> int:
        cfg = self.cfg
        seed = getattr(cfg, "seed", None) or 0
        try:
            return int(seed)
        except (TypeError, ValueError):
            return 0

    def _load_from_root(self) -> LoadedDataset:
        if self.root is None:
            raise SyntheticError("persisted synthetic dataset path unavailable")
        import json

        meta = json.loads((self.root / "meta.json").read_text(encoding="utf-8"))
        npz = np.load(self.root / VALUES_NAME)
        label_names = [str(x) for x in np.load(self.root / LABEL_NAMES, allow_pickle=True)]
        series: list[SubjectSeries] = []
        for i in range(int(meta["n_series"])):
            subject_id = f"synthetic_s{i:02d}"
            if self.subjects_requested and subject_id not in self.subjects_requested:
                continue
            series.append(
                SubjectSeries(
                    subject_id=subject_id,
                    values=npz[f"values_{i}"].astype(np.float32),
                    labels=npz[f"labels_{i}"].astype(np.int64),
                    sampling_rate=float(meta["sampling_rate"]),
                    channel_names=list(meta["channel_names"]),
                    kind="continuous",
                    timestamps=np.arange(npz[f"values_{i}"].shape[0], dtype=np.float64)
                    / float(meta["sampling_rate"]),
                    extra={"source_file": str(self.root / VALUES_NAME)},
                )
            )
        if not series:
            raise SyntheticError("no subjects left after filtering")
        return LoadedDataset(
            name=self.name,
            sampling_rate=float(meta["sampling_rate"]),
            channel_names=list(meta["channel_names"]),
            label_names=label_names,
            kind="continuous",
            series=series,
            source_path=str(self.root),
            notes="Persisted synthetic dataset",
            extra={"seed": self.cfg_seed()},
        )

    def get_subjects(self) -> list[str]:
        return [f"synthetic_s{i:02d}" for i in range(2)]

    def get_sampling_rate(self) -> float:
        return self.quality

    def get_sensor_columns(self) -> list[str]:
        return [f"ch_{c}" for c in range(3)]

    def get_labels(self) -> list[str]:
        return list(LABELS)
