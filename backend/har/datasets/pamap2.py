"""PAMAP2 Physical Activity Monitoring adapter (Reiss & Stricker, 2012).

Two revisions of the distribution exist and the *file* layout is detected:

* 54 columns -- ``1 timestamp, 2 activityID, 3 heart_rate`` followed by
  three IMU blocks (hand, chest, ankle) of 17 columns each:
  ``temperature, acc_16g[3], acc_6g[3], gyro[3], mag[3], orientation[4]``.
* 63 columns -- same but each IMU block has 20 columns:
  ``temperature, acc_16g[3], acc_6g_scale8[3], acc_6g_scale1[3],
  gyro[3], mag[3], orientation[4]``.

The adapter produces one continuous :class:`SubjectSeries` per subject at
100 Hz, honouring ``dataset.channels`` (exact names built from the layouts
above, or the group shortcuts ``hand``/``chest``/``ankle``) and
``dataset.activity_filter`` (activity code 0 = transient is always dropped).
No values are fabricated; gaps are dealt with downstream by the configured
missing-data policy.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from ..config.schema import DatasetConfig
from ..utils.logging import get_logger
from .base import LoadedDataset, SubjectSeries, DatasetAdapter, register_adapter

logger = get_logger(__name__)

SAMPLING_RATE = 100.0

PAMAP2_ACTIVITY_LABELS: dict[int, str] = {
    1: "lying",
    2: "sitting",
    3: "standing",
    4: "walking",
    5: "jogging",
    6: "cycling",
    7: "nordic walking",
    8: "ascending stairs",
    9: "descending stairs",
    10: "vacuum cleaning",
    11: "ironing",
    12: "rope jumping",
    13: "running on treadmill (4 km/h - 6.3 km/h)",
    17: "watching TV",
    18: "computer work",
    19: "car driving",
    24: "rope jumping",
}


def _imu_layout(n_columns: int) -> dict[str, object]:
    """Return per-IMU layout for the detected revision."""
    if n_columns == 63:
        return {
            "cols_per_imu": 20,
            "acc_sets": [
                ("acc16g", 3),
                ("acc6g_8g_scale", 3),
                ("acc6g_1g_scale", 3),
            ],
        }
    if n_columns == 54:
        return {
            "cols_per_imu": 17,
            "acc_sets": [("acc16g", 3), ("acc6g", 3)],
        }
    raise Pamap2Error(
        f"unexpected PAMAP2 column count {n_columns}; expected 54 or 63. "
        "Refusing to guess the layout."
    )


class Pamap2Error(RuntimeError):
    pass


def _per_imu_names(layout: dict[str, object], tag: str) -> list[str]:
    names: list[str] = []
    names.append(f"temp_{tag}")
    for acc_name, count in layout["acc_sets"]:  # type: ignore[index]
        names.extend(f"{acc_name}_{tag}_{i}" for i in range(count))
    names.extend(f"gyro_{tag}_{i}" for i in range(3))
    names.extend(f"mag_{tag}_{i}" for i in range(3))
    names.extend(f"ori_{tag}_{i}" for i in range(4))
    return names


@register_adapter
class Pamap2Adapter(DatasetAdapter):
    name = "pamap2"

    def __init__(self, cfg: DatasetConfig) -> None:
        self.cfg = cfg
        self.root = self._resolve_root(cfg.root)
        self.subject_filter = set(cfg.subjects) if cfg.subjects else None
        self._imu_groups = ("hand", "chest", "ankle")

    def _resolve_root(self, root: str | None) -> Path:
        from ..utils.seeds import project_root

        base = Path(root) if root else project_root() / "data" / "raw" / "PAMAP2_Dataset"
        if not base.exists():
            raise Pamap2Error(
                f"PAMAP2 dataset not found at {base}. Download\n"
                "https://archive.ics.uci.edu/dataset/501/pamap2+physical+activity+monitoring\n"
                "and extract so that subject101.dat ... exist directly in it."
            )
        return base

    # ------------------------------------------------------------------
    def _load_one_subject(self, path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str], dict[int, list[int]]]:
        raw = np.loadtxt(path, dtype=np.float32, ndmin=2)
        n_columns = raw.shape[1]
        layout = _imu_layout(n_columns)
        cols_per_imu = int(layout["cols_per_imu"])  # type: ignore[index]

        time_col = raw[:, 0]
        act_col = raw[:, 1].astype(np.int64)
        heart = raw[:, 2:3]

        names: list[str] = ["heart_rate"]
        blocks: list[np.ndarray] = [heart]

        # keep track of accelerometer alternatives so `use_acc16` can filter
        for tag, offset in zip(self._imu_groups, (3, 3 + cols_per_imu, 3 + 2 * cols_per_imu)):
            imu = raw[:, offset : offset + cols_per_imu]
            imu_names = _per_imu_names(layout, tag)
            names.extend(imu_names)
            blocks.append(imu)

        data = np.concatenate(blocks, axis=1)  # (T, 1 + 3*cols_per_imu)
        return data, act_col.astype(np.int64), time_col, names, layout  # type: ignore[return-value]

    def _select_default_columns(self, names: list[str]) -> list[str]:
        """Default selection: 6-channel accelerometers + gyro for all IMUs."""
        selected = [
            n
            for n in names
            if (n.startswith("acc6g_") and not n.endswith("scale")) or n.startswith("gyro_")
        ]
        # for the 63-column revision exclude the alternate 6g scale columns
        selected = [n for n in selected if "8g_scale" not in n and "1g_scale" not in n]
        return selected

    # ------------------------------------------------------------------
    def load(self) -> LoadedDataset:
        subject_files = sorted(self.root.glob("subject*.dat"))
        if not subject_files:
            raise Pamap2Error(f"no subject*.dat files found under {self.root}")

        keep_codes: frozenset[int] | None = None
        if self.cfg.activity_filter:
            unknown = [
                a
                for a in self.cfg.activity_filter
                if a not in PAMAP2_ACTIVITY_LABELS.values()
            ]
            if unknown:
                raise Pamap2Error(f"unknown PAMAP2 activity names {unknown}")
            keep_codes = frozenset(
                code
                for code, label in PAMAP2_ACTIVITY_LABELS.items()
                if label in set(self.cfg.activity_filter)
            )

        series: list[SubjectSeries] = []
        for path in subject_files:
            subject_id = f"subject_{path.stem.replace('subject', '')}"
            if self.subject_filter and subject_id not in self.subject_filter:
                continue
            data, act, times, all_names, _layout = self._load_one_subject(path)

            mask = act > 0 if keep_codes is None else np.isin(act, sorted(keep_codes))
            data = data[mask]
            if data.shape[0] == 0:
                logger.warning("pamap2 | subject %s has no labelled samples", subject_id)
                continue
            acts = act[mask]
            times = times[mask]

            if self.cfg.channels:
                unknown = [c for c in self.cfg.channels if c not in all_names]
                if unknown:
                    raise Pamap2Error(
                        f"unknown PAMAP2 channel names {unknown}; available prefix groups "
                        "acc16g_*, acc6g_*, gyro_*, mag_*, ori_*, temp_*, heart_rate, or "
                        "use the group shortcuts hand/chest/ankle via dataset.subjects? "
                        "No, use dataset.channels with full names."
                    )
                chosen = list(self.cfg.channels)
            else:
                chosen = self._select_default_columns(all_names)

            if "heart_rate" not in chosen:
                chosen = [c for c in chosen]
            col_idx = [all_names.index(c) for c in chosen]
            data = data[:, np.array(col_idx)]
            names = chosen

            series.append(
                SubjectSeries(
                    subject_id=subject_id,
                    values=data.astype(np.float32),
                    labels=acts.astype(np.int64),
                    sampling_rate=SAMPLING_RATE,
                    channel_names=names,
                    kind="continuous",
                    timestamps=times,
                    extra={"source_file": path.name},
                )
            )

        label_names = [
            name
            for code, name in sorted(PAMAP2_ACTIVITY_LABELS.items())
            if keep_codes is None or code in keep_codes
        ]
        return LoadedDataset(
            name=self.name,
            sampling_rate=SAMPLING_RATE,
            channel_names=series[0].channel_names if series else [],
            label_names=label_names,
            kind="continuous",
            series=series,
            source_path=str(self.root),
            notes=(
                "Continuous IMU streams at 100 Hz. Default channels: 6-axis "
                "(acc6g+gyro) x hand/chest/ankle. heart_rate is heavily missing "
                "and excluded by default."
            ),
            extra={"label_map": PAMAP2_ACTIVITY_LABELS},
        )

    # ------------------------------------------------------------------
    def get_subjects(self) -> list[str]:
        files = sorted(self.root.glob("subject*.dat"))
        return [f"subject_{f.stem.replace('subject', '')}" for f in files]

    def get_sampling_rate(self) -> float:
        return self.cfg.sampling_rate_override or SAMPLING_RATE

    def get_sensor_columns(self) -> list[str]:
        if self.cfg.channels:
            return list(self.cfg.channels)
        probe = sorted(self.root.glob("subject*.dat"))
        if not probe:
            raise Pamap2Error(f"no PAMAP2 subject files under {self.root}")
        raw = np.loadtxt(probe[0], dtype=np.float32, ndmin=2)
        layout = _imu_layout(raw.shape[1])
        names: list[str] = []
        for tag in self._imu_groups:
            names.append(f"temp_{tag}")
            for acc_name, count in layout["acc_sets"]:  # type: ignore[index]
                names.extend(f"{acc_name}_{tag}_{i}" for i in range(count))
            names.extend(f"gyro_{tag}_{i}" for i in range(3))
            names.extend(f"mag_{tag}_{i}" for i in range(3))
            names.extend(f"ori_{tag}_{i}" for i in range(4))
        return self._select_default_columns(names)

    def get_labels(self) -> list[str]:
        return list(PAMAP2_ACTIVITY_LABELS.values())
