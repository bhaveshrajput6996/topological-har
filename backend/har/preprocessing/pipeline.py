"""Preprocessing orchestration (gaps, filter, per-subject pipeline)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from ..config.schema import Config
from ..datasets.base import LoadedDataset, SubjectSeries
from .filtering import apply_lowpass, design_lowpass, FilterError
from .imputation import GapReport, handle_missing, validate_timestamps
from .normalization import ScalerWrapper


@dataclass
class ProcessedSeries:
    """Preprocessed per-subject series (still pre-windowing)."""

    series: SubjectSeries
    invalid_mask: np.ndarray | None
    gap_report: GapReport | None
    filtered: bool
    stats: dict[str, Any] = field(default_factory=dict)


class Preprocessor:
    """Apply the stages common to every dataset adapter:

    raw -> timestamp validation -> missing-data analysis ->
    small-gap interpolation (policy) -> low-pass filter -> (normalisation
    happens later, fitted on training subjects only).
    """

    def __init__(self, cfg: Config, sampling_rate: float) -> None:
        self.cfg = cfg
        self.sampling_rate = sampling_rate
        self.filter_sos: np.ndarray | None = None
        if cfg.preprocessing.filter.enabled:
            self.filter_sos = design_lowpass(sampling_rate, cfg.preprocessing.filter)

    # ------------------------------------------------------------------
    def process_series(self, series: SubjectSeries) -> ProcessedSeries:
        values = np.array(series.values, dtype=np.float32, copy=True, order="C") \
            if series.kind == "continuous" else np.array(series.values, dtype=np.float32, copy=True)

        gap_report: GapReport | None = None
        timestamps_report: dict[str, float] = {}
        invalid_mask: np.ndarray | None = None

        if series.kind == "continuous":
            timestamps_report = validate_timestamps(series.timestamps, self.sampling_rate)
            values2 = values.astype(np.float32, copy=False)
            values2, invalid_mask, gap_report = handle_missing(values2, self.sampling_rate, self.cfg.preprocessing)
            values = values2
            if self.filter_sos is not None and invalid_mask is not None and invalid_mask.any():
                raise FilterError(
                    "the configured filter is incompatible with unresolved missing "
                    "samples; choose interpolation=linear with a small max gap, or "
                    "disable filtering"
                )
            if self.filter_sos is not None:
                values = apply_lowpass(values, self.filter_sos)
        else:
            # presegmented / feature rows: NaNs handled per row downstream
            if not np.isfinite(values).all():
                invalid_mask = ~np.isfinite(values).all(axis=-1).any(axis=-1) \
                    if values.ndim == 3 else None

        stats: dict[str, Any] = {}
        if timestamps_report:
            stats["timestamps"] = timestamps_report
        if gap_report is not None:
            stats["gaps"] = {
                "n_missing": gap_report.n_missing,
                "n_gap_runs": gap_report.n_gap_runs,
                "max_gap_s": gap_report.max_gap_s,
                "n_interpolated": gap_report.n_interpolated,
                "n_invalid": gap_report.n_invalid,
                "policy": gap_report.policy,
            }
        stats["filtered"] = bool(self.filter_sos is not None)

        out_series = SubjectSeries(
            subject_id=series.subject_id,
            values=values,
            labels=series.labels,
            sampling_rate=series.sampling_rate,
            channel_names=series.channel_names,
            kind=series.kind,
            window_shape=series.window_shape,
            timestamps=series.timestamps,
            extra=series.extra,
        )
        return ProcessedSeries(
            series=out_series,
            invalid_mask=invalid_mask,
            gap_report=gap_report,
            filtered=bool(self.filter_sos is not None),
            stats=stats,
        )

    # ------------------------------------------------------------------
    def process_dataset(self, dataset: LoadedDataset) -> list[ProcessedSeries]:
        return [self.process_series(s) for s in dataset.series]

    # scaler ------------------------------------------------------------
    @staticmethod
    def new_scaler(cfg: Config) -> ScalerWrapper:
        return ScalerWrapper(enabled=cfg.preprocessing.normalization == "standard")
