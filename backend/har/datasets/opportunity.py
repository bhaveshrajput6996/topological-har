"""OPPORTUNITY Activity Recognition adapter (Roggen et al., 2010).

The OPPORTUNITY distribution ships MATLAB files per recording.  The .mat
files expose (variable names auto-detected because the dataset has several
revisions):

* an inertial data matrix with **133 columns** of body-worn sensor values
  (per Ordóñez & Roggen 2016 who used exactly these columns), read with
  ``scipy.io.loadmat`` and ``squeeze_me=True``;
* per-sample gesture labels (mid-level gestures, ids 406516..406527 plus
  null class 0) or locomotion labels (ids 0..5);
* optional timestamps.

Modes (``dataset.signal_mode``): ``gesture`` (default) or ``locomotion``.
Channel subsets are selected by column via ``dataset.channels``
(``"col_42"``) because the full column sheet depends on the MAT revision;
without a selection all 133 columns are used.  Sampling rate is 32 Hz.

The dataset is large (multiple GB); only MS-compatible .mat v7 files
handled by ``scipy.io`` are supported.
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np

from ..config.schema import DatasetConfig
from ..utils.logging import get_logger
from .base import LoadedDataset, SubjectSeries, DatasetAdapter, register_adapter

logger = get_logger(__name__)

SAMPLING_RATE = 32.0

GESTURE_LABELS: dict[int, str] = {
    406516: "open door 1",
    406517: "open door 2",
    406518: "close door 1",
    406519: "close door 2",
    406520: "open fridge",
    406521: "close fridge",
    406522: "open dishwasher",
    406523: "close dishwasher",
    406524: "open drawer 1",
    406525: "close drawer 1",
    406526: "open drawer 2",
    406527: "close drawer 2",
}

LOCOMOTION_LABELS: dict[int, str] = {
    1: "walk",
    2: "stand",
    3: "sit",
    4: "lie",
    5: "unknown",
}


class OpportunityError(RuntimeError):
    pass


def _find_variable(keys: list[str], *substrings: str) -> str | None:
    for key in keys:
        low = key.lower()
        if all(s.lower() in low for s in substrings):
            return key
    return None


@register_adapter
class OpportunityAdapter(DatasetAdapter):
    name = "opportunity"

    def __init__(self, cfg: DatasetConfig) -> None:
        self.cfg = cfg
        self.root = self._resolve_root(cfg.root)
        self.signal_mode = (cfg.signal_mode or "gesture").lower()
        if self.signal_mode not in ("gesture", "locomotion"):
            raise OpportunityError(
                "opportunity supports signal_mode 'gesture' or 'locomotion'"
            )
        self.subject_filter = set(cfg.subjects) if cfg.subjects else None
        self.include_null = False

    def _resolve_root(self, root: str | None) -> Path:
        from ..utils.seeds import project_root

        base = Path(root) if root else project_root() / "data" / "raw" / "Opportunity"
        if not base.exists():
            raise OpportunityError(
                f"OPPORTUNITY dataset not found at {base}. Download it from "
                "https://archive.ics.uci.edu/dataset/226/opportunity+activity+recognition "
                "and extract the .mat recordings (it is several GB)."
            )
        return base

    def _record_files(self) -> list[Path]:
        files = sorted(self.root.glob("*.mat")) or sorted(self.root.glob("**/*.mat"))
        if not files:
            raise OpportunityError(f"no .mat recordings found under {self.root}")
        return files

    @staticmethod
    def _subject_from_name(path: Path, fallback_index: int) -> str:
        match = re.search(r"[Ss]?(\d+)", path.stem)
        if match and int(match.group(1)) <= 4:
            return f"subject_{int(match.group(1)):02d}"
        return f"subject_unknown_{fallback_index:02d}"

    # ------------------------------------------------------------------
    def load(self) -> LoadedDataset:
        try:
            from scipy.io import loadmat
        except ImportError as exc:  # pragma: no cover
            raise OpportunityError("scipy is required for OPPORTUNITY .mat files") from exc

        label_map = GESTURE_LABELS if self.signal_mode == "gesture" else LOCOMOTION_LABELS

        series: list[SubjectSeries] = []
        for idx, path in enumerate(self._record_files()):
            subject_id = self._subject_from_name(path, idx)
            if self.subject_filter and subject_id not in self.subject_filter:
                continue
            mat = loadmat(str(path), squeeze_me=True, struct_as_record=False)
            keys = [k for k in mat.keys() if not k.startswith("__")]

            inertial_key = _find_variable(keys, "inertial")
            if inertial_key is None:
                logger.warning(
                    "opportunity | %s: no inertial data variable found; skipping "
                    "(keys=%s)",
                    path.name,
                    keys[:12],
                )
                continue
            data = np.asarray(mat[inertial_key], dtype=np.float32)
            if data.ndim == 1:
                data = data[:, None]
            n_columns = data.shape[1]

            label_key = _find_variable(
                keys, "gesture" if self.signal_mode == "gesture" else "loco"
            )
            if label_key is None:
                logger.warning(
                    "opportunity | %s: no %s label variable found; skipping",
                    path.name,
                    self.signal_mode,
                )
                continue
            labels = np.asarray(mat[label_key]).reshape(-1).astype(np.int64)

            timestamps = None
            time_key = _find_variable(keys, "time")
            if time_key is not None:
                try:
                    timestamps = np.asarray(mat[time_key]).reshape(-1).astype(np.float64)
                except (TypeError, ValueError):
                    timestamps = None

            known = set(label_map)
            known.add(0)
            mask = np.isin(labels, sorted(known))
            if not self.include_null:
                mask &= labels != 0
            n_kept = int(mask.sum())
            if n_kept == 0:
                logger.warning("opportunity | %s: no labelled samples; skipped", path.name)
                continue

            # special case: gesture labels may be stored as (T, k) matrix
            data = data[mask]
            labels_kept = labels[mask]
            times_kept = timestamps[mask] if timestamps is not None else None

            channel_names = (
                list(self.cfg.channels)
                if self.cfg.channels
                else [f"col_{i}" for i in range(n_columns)]
            )
            if self.cfg.channels:
                unknown = [c for c in channel_names if not (c.startswith("col_") and c[4:].isdigit() and int(c[4:]) < n_columns)]
                if unknown:
                    raise OpportunityError(
                        f"unknown OPPORTUNITY channel specifiers {unknown[:5]}; use 'col_<index>'"
                    )
                col_idx = np.array([int(c[4:]) for c in channel_names], dtype=np.int64)
                data = data[:, col_idx]

            series.append(
                SubjectSeries(
                    subject_id=subject_id,
                    values=data,
                    labels=labels_kept,
                    sampling_rate=SAMPLING_RATE,
                    channel_names=channel_names,
                    kind="continuous",
                    timestamps=times_kept,
                    extra={"source_file": path.name, "variable_inertial": inertial_key},
                )
            )

        label_names = [label_map[c] for c in sorted(label_map)]
        return LoadedDataset(
            name=self.name,
            sampling_rate=SAMPLING_RATE,
            channel_names=series[0].channel_names if series else [],
            label_names=label_names,
            kind="continuous",
            series=series,
            source_path=str(self.root),
            notes=(
                "Auto-detected MAT variables; channel semantics depend on the "
                "OPPORTUNITY revision (dataset provides no explicit column sheet "
                "inside the .mat). All labelled samples except null class are kept."
            ),
            extra={"label_map": label_map, "include_null": self.include_null},
        )

    # ------------------------------------------------------------------
    def get_subjects(self) -> list[str]:
        files = self._record_files()
        out = []
        for idx, path in enumerate(files):
            sid = self._subject_from_name(path, idx)
            if sid not in out:
                out.append(sid)
        return out

    def get_sampling_rate(self) -> float:
        return self.cfg.sampling_rate_override or SAMPLING_RATE

    def get_sensor_columns(self) -> list[str]:
        if self.cfg.channels:
            return list(self.cfg.channels)
        files = self._record_files()
        from scipy.io import loadmat

        mat = loadmat(str(files[0]), squeeze_me=True, struct_as_record=False)
        keys = [k for k in mat.keys() if not k.startswith("__")]
        inertial_key = _find_variable(keys, "inertial")
        if inertial_key is None:
            raise OpportunityError("no inertial variable in OPPORTUNITY recordings")
        data = np.asarray(mat[inertial_key])
        return [f"col_{i}" for i in range(data.shape[1])]

    def get_labels(self) -> list[str]:
        label_map = GESTURE_LABELS if self.signal_mode == "gesture" else LOCOMOTION_LABELS
        return [label_map[c] for c in sorted(label_map)]
