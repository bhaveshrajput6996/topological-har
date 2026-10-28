"""Typed experiment configuration.

The whole experiment is described by one YAML file which is parsed into the
frozen-ish dataclasses below.  Magic numbers live in the YAML, not in the
code.  Unknown keys are rejected so typos fail loudly instead of silently
using defaults.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from ..utils.seeds import project_root


class ConfigError(ValueError):
    """Raised for invalid or inconsistent configuration."""


# ---------------------------------------------------------------------------
# dataclasses
# ---------------------------------------------------------------------------
@dataclass
class ProjectConfig:
    name: str = "topological-har"
    seed: int = 42


@dataclass
class DatasetConfig:
    """Dataset selection and per-dataset channel semantics.

    ``name`` values are the keys of the adapter registry
    (uci_har / pamap2 / opportunity / synthetic).
    """

    name: str = "uci_har"
    root: str | None = None
    signal_mode: str | None = None
    subjects: list[str] | None = None
    channels: list[str] | None = None
    activity_filter: list[str] | None = None
    sampling_rate_override: float | None = None


@dataclass
class WindowConfig:
    duration_seconds: float = 2.0
    overlap: float = 0.5
    max_windows: int | None = None
    label_strategy: str = "mode"


@dataclass
class FilterConfig:
    enabled: bool = False
    type: str = "butterworth"
    order: int = 2
    cutoff_hz: float | None = None


@dataclass
class PreprocessConfig:
    interpolation: str = "linear"
    max_interp_gap_s: float = 1.0
    filter: FilterConfig = field(default_factory=FilterConfig)
    normalization: str = "standard"


@dataclass
class SubsampleConfig:
    enabled: bool = True
    max_points: int = 80
    method: str = "stride"


@dataclass
class PersistenceImageConfig:
    resolution: list[int] = field(default_factory=lambda: [20, 20])
    bandwidth: float = 0.1
    weight: str = "persistence"
    max_features: int = 0  # 0 = keep all features; >0 keep the top-K by persistence


@dataclass
class TdaConfig:
    enabled: bool = True
    subsample: SubsampleConfig = field(default_factory=SubsampleConfig)
    metric: str = "euclidean"
    homology_dimensions: list[int] = field(default_factory=lambda: [0, 1])
    persistence_image: PersistenceImageConfig = field(default_factory=PersistenceImageConfig)
    use_edge_collapse: bool = True
    min_persistence: float = 0.0
    statistical_features: bool = False


@dataclass
class ModelConfig:
    type: str = "tda_deepconvlstm"
    conv_channels: list[int] = field(default_factory=lambda: [64, 128, 128, 256])
    conv_kernel_size: int = 5
    lstm_hidden: int = 128
    dropout: float = 0.5
    head_hidden: list[int] = field(default_factory=lambda: [256, 128])
    apply_scaler_to_tda: bool = True


@dataclass
class TrainingConfig:
    batch_size: int = 64
    learning_rate: float = 0.001
    epochs: int = 50
    early_stopping: bool = True
    patience: int = 8
    lr_scheduler: str = "plateau"
    scheduler_patience: int = 3
    scheduler_factor: float = 0.5
    weight_decay: float = 0.0
    seed: int = 42
    val_splits: int = 5
    num_workers: int = 0


@dataclass
class EvalConfig:
    cv_folds: int = 5
    folds: list[int] | None = None
    inference_batch: int = 64
    latency_runs: int = 0
    svm_c: float = 1.0
    svm_gamma: str | float = "scale"
    svm_max_train_windows: int = 20000


@dataclass
class PathsConfig:
    data_dir: str = "data"
    artifacts_dir: str = "artifacts"
    db_path: str = "artifacts/har_experiments.db"


@dataclass
class Config:
    project: ProjectConfig = field(default_factory=ProjectConfig)
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    window: dict[str, WindowConfig] = field(
        default_factory=lambda: {"default": WindowConfig()}
    )
    preprocessing: PreprocessConfig = field(default_factory=PreprocessConfig)
    tda: TdaConfig = field(default_factory=TdaConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    evaluation: EvalConfig = field(default_factory=EvalConfig)
    paths: PathsConfig = field(default_factory=PathsConfig)


# ---------------------------------------------------------------------------
# parsing
# ---------------------------------------------------------------------------
def _require_keys(d: dict[str, Any], cls_name: str) -> None:
    if not isinstance(d, dict):
        raise ConfigError(f"{cls_name} must be a mapping, got {type(d).__name__}")


def parse_filter(d: dict[str, Any]) -> FilterConfig:
    _require_keys(d, "preprocessing.filter")
    allowed = {"enabled", "type", "order", "cutoff_hz"}
    unknown = set(d) - allowed
    if unknown:
        raise ConfigError(f"preprocessing.filter has unknown keys {sorted(unknown)}")
    cfg = FilterConfig()
    if "enabled" in d:
        cfg.enabled = bool(d["enabled"])
    if "type" in d:
        if d["type"] != "butterworth":
            raise ConfigError(f"unsupported filter type {d['type']!r}")
        cfg.type = str(d["type"])
    if "order" in d:
        cfg.order = int(d["order"])
        if cfg.order < 1:
            raise ConfigError("filter.order must be >= 1")
    if "cutoff_hz" in d and d["cutoff_hz"] is not None:
        cfg.cutoff_hz = float(d["cutoff_hz"])
        if cfg.cutoff_hz <= 0:
            raise ConfigError("filter.cutoff_hz must be > 0")
    return cfg


def parse_preprocess(d: dict[str, Any]) -> PreprocessConfig:
    _require_keys(d, "preprocessing")
    allowed = {"interpolation", "max_interp_gap_s", "filter", "normalization"}
    unknown = set(d) - allowed
    if unknown:
        raise ConfigError(f"preprocessing has unknown keys {sorted(unknown)}")
    cfg = PreprocessConfig()
    if "interpolation" in d:
        if d["interpolation"] not in ("linear", "drop_window", "mark_invalid"):
            raise ConfigError(
                "preprocessing.interpolation must be one of linear|drop_window|mark_invalid"
            )
        cfg.interpolation = str(d["interpolation"])
    if "max_interp_gap_s" in d:
        cfg.max_interp_gap_s = float(d["max_interp_gap_s"])
        if cfg.max_interp_gap_s < 0:
            raise ConfigError("max_interp_gap_s must be >= 0")
    if "filter" in d and d["filter"] is not None:
        cfg.filter = parse_filter(d["filter"])
    if "normalization" in d:
        if d["normalization"] not in ("standard", "none"):
            raise ConfigError("preprocessing.normalization must be standard|none")
        cfg.normalization = str(d["normalization"])
    return cfg


def parse_tda(d: dict[str, Any]) -> TdaConfig:
    _require_keys(d, "tda")
    allowed = {
        "enabled",
        "subsample",
        "metric",
        "homology_dimensions",
        "persistence_image",
        "use_edge_collapse",
        "min_persistence",
        "statistical_features",
    }
    unknown = set(d) - allowed
    if unknown:
        raise ConfigError(f"tda has unknown keys {sorted(unknown)}")
    cfg = TdaConfig()
    if "enabled" in d:
        cfg.enabled = bool(d["enabled"])
    if "subsample" in d and d["subsample"] is not None:
        s = d["subsample"]
        _require_keys(s, "tda.subsample")
        cfg.subsample = SubsampleConfig(
            enabled=bool(s.get("enabled", True)),
            max_points=int(s.get("max_points", 80)),
            method=str(s.get("method", "stride")),
        )
        if cfg.subsample.max_points < 2:
            raise ConfigError("tda.subsample.max_points must be >= 2")
        if cfg.subsample.method not in ("stride", "random"):
            raise ConfigError("tda.subsample.method must be stride|random")
    if "metric" in d and d["metric"] != "euclidean":
        raise ConfigError("tda.metric must be euclidean (gudhi Vietoris-Rips)")
    if "homology_dimensions" in d:
        dims = d["homology_dimensions"]
        if not isinstance(dims, list) or any(int(x) != x or x < 0 for x in dims):
            raise ConfigError("tda.homology_dimensions must be a list of non-negative ints")
        if max(dims) > 1:
            raise ConfigError("only H0 and H1 are computed (gudhi VR expansion max_dim=2)")
        cfg.homology_dimensions = [int(x) for x in dims]
    if "persistence_image" in d and d["persistence_image"] is not None:
        p = d["persistence_image"]
        _require_keys(p, "tda.persistence_image")
        res = p.get("resolution", [20, 20])
        if not isinstance(res, list) or len(res) != 2 or any(int(x) != x or x < 1 for x in res):
            raise ConfigError("tda.persistence_image.resolution must be [rows, cols] ints >= 1")
        bw = float(p.get("bandwidth", 0.1))
        if bw <= 0:
            raise ConfigError("tda.persistence_image.bandwidth must be > 0")
        weight = str(p.get("weight", "persistence"))
        if weight not in ("persistence", "uniform"):
            raise ConfigError("tda.persistence_image.weight must be persistence|uniform")
        max_features = int(p.get("max_features", 0))
        if max_features < 0:
            raise ConfigError("tda.persistence_image.max_features must be >= 0")
        cfg.persistence_image = PersistenceImageConfig(
            resolution=[int(res[0]), int(res[1])],
            bandwidth=bw,
            weight=weight,
            max_features=max_features,
        )
    if "use_edge_collapse" in d:
        cfg.use_edge_collapse = bool(d["use_edge_collapse"])
    if "min_persistence" in d:
        cfg.min_persistence = float(d["min_persistence"])
        if cfg.min_persistence < 0:
            raise ConfigError("tda.min_persistence must be >= 0")
    if "statistical_features" in d:
        cfg.statistical_features = bool(d["statistical_features"])
    return cfg


def parse_model(d: dict[str, Any]) -> ModelConfig:
    _require_keys(d, "model")
    allowed = {
        "type",
        "conv_channels",
        "conv_kernel_size",
        "lstm_hidden",
        "dropout",
        "head_hidden",
        "apply_scaler_to_tda",
    }
    unknown = set(d) - allowed
    if unknown:
        raise ConfigError(f"model has unknown keys {sorted(unknown)}")
    cfg = ModelConfig()
    known_types = {
        "tda_deepconvlstm",
        "deepconvlstm",
        "cnn",
        "lstm",
        "svm",
    }
    if "type" in d:
        if d["type"] not in known_types:
            raise ConfigError(f"model.type must be one of {sorted(known_types)}")
        cfg.type = str(d["type"])
    if d.get("type") == "svm":
        # SVM is a torch-free sklearn baseline; window arrays only
        pass
    if "conv_channels" in d:
        cfg.conv_channels = [int(x) for x in d["conv_channels"]]
    if "conv_kernel_size" in d:
        cfg.conv_kernel_size = int(d["conv_kernel_size"])
    if "lstm_hidden" in d:
        cfg.lstm_hidden = int(d["lstm_hidden"])
    if "dropout" in d:
        cfg.dropout = float(d["dropout"])
        if not 0 <= cfg.dropout < 1:
            raise ConfigError("model.dropout must be in [0, 1)")
    if "head_hidden" in d:
        cfg.head_hidden = [int(x) for x in d["head_hidden"]]
    if "apply_scaler_to_tda" in d:
        cfg.apply_scaler_to_tda = bool(d["apply_scaler_to_tda"])
    return cfg


def parse_training(d: dict[str, Any]) -> TrainingConfig:
    _require_keys(d, "training")
    allowed = {
        "batch_size",
        "learning_rate",
        "epochs",
        "early_stopping",
        "patience",
        "lr_scheduler",
        "scheduler_patience",
        "scheduler_factor",
        "weight_decay",
        "seed",
        "val_splits",
        "num_workers",
    }
    unknown = set(d) - allowed
    if unknown:
        raise ConfigError(f"training has unknown keys {sorted(unknown)}")
    cfg = TrainingConfig()
    if "batch_size" in d:
        cfg.batch_size = int(d["batch_size"])
    if "learning_rate" in d:
        cfg.learning_rate = float(d["learning_rate"])
    if "epochs" in d:
        cfg.epochs = int(d["epochs"])
    if "early_stopping" in d:
        cfg.early_stopping = bool(d["early_stopping"])
    if "patience" in d:
        cfg.patience = int(d["patience"])
    if "lr_scheduler" in d:
        if d["lr_scheduler"] not in ("none", "plateau"):
            raise ConfigError("training.lr_scheduler must be none|plateau")
        cfg.lr_scheduler = str(d["lr_scheduler"])
    if "scheduler_patience" in d:
        cfg.scheduler_patience = int(d["scheduler_patience"])
    if "scheduler_factor" in d:
        cfg.scheduler_factor = float(d["scheduler_factor"])
    if "weight_decay" in d:
        cfg.weight_decay = float(d["weight_decay"])
    if "seed" in d:
        cfg.seed = int(d["seed"])
    if "val_splits" in d:
        cfg.val_splits = int(d["val_splits"])
    if "num_workers" in d:
        cfg.num_workers = int(d["num_workers"])
    if cfg.epochs < 1:
        raise ConfigError("training.epochs must be >= 1")
    if cfg.batch_size < 1:
        raise ConfigError("training.batch_size must be >= 1")
    return cfg


def parse_eval(d: dict[str, Any]) -> EvalConfig:
    _require_keys(d, "evaluation")
    allowed = {
        "cv_folds",
        "folds",
        "inference_batch",
        "latency_runs",
        "svm_c",
        "svm_gamma",
        "svm_max_train_windows",
    }
    unknown = set(d) - allowed
    if unknown:
        raise ConfigError(f"evaluation has unknown keys {sorted(unknown)}")
    cfg = EvalConfig()
    if "cv_folds" in d:
        cfg.cv_folds = int(d["cv_folds"])
    if "folds" in d:
        if d["folds"] is not None:
            cfg.folds = [int(x) for x in d["folds"]]
    if "inference_batch" in d:
        cfg.inference_batch = int(d["inference_batch"])
    if "latency_runs" in d:
        cfg.latency_runs = int(d["latency_runs"])
    if "svm_c" in d:
        cfg.svm_c = float(d["svm_c"])
    if "svm_gamma" in d:
        cfg.svm_gamma = "scale" if d["svm_gamma"] == "scale" else float(d["svm_gamma"])
    if "svm_max_train_windows" in d:
        cfg.svm_max_train_windows = int(d["svm_max_train_windows"])
    if cfg.cv_folds < 2:
        raise ConfigError("evaluation.cv_folds must be >= 2")
    return cfg


WINDOW_FIELDS = {"duration_seconds", "overlap", "max_windows", "label_strategy"}


def _window_from_fields(raw: dict[str, Any]) -> WindowConfig:
    unknown = set(raw) - WINDOW_FIELDS
    if unknown:
        raise ConfigError(f"window has unknown keys {sorted(unknown)}")
    cfg = WindowConfig()
    if "duration_seconds" in raw:
        cfg.duration_seconds = float(raw["duration_seconds"])
        if cfg.duration_seconds <= 0:
            raise ConfigError("window.duration_seconds must be > 0")
    if "overlap" in raw:
        if raw["overlap"] is None:
            # dataset-defined (presegmented mode); validated downstream
            cfg.overlap = None
        else:
            cfg.overlap = float(raw["overlap"])
            if not 0 <= cfg.overlap < 1:
                raise ConfigError("window.overlap must be in [0, 1)")
    if "max_windows" in raw:
        cfg.max_windows = None if raw["max_windows"] is None else int(raw["max_windows"])
        if cfg.max_windows is not None and cfg.max_windows < 1:
            raise ConfigError("window.max_windows must be >= 1")
    if "label_strategy" in raw:
        if raw["label_strategy"] not in ("mode", "center"):
            raise ConfigError("window.label_strategy must be mode|center")
        cfg.label_strategy = str(raw["label_strategy"])
    return cfg


def parse_window(d: dict[str, WindowConfig], raw: Any) -> dict[str, WindowConfig]:
    if raw is None:
        raise ConfigError("window section is required")
    _require_keys(raw, "window")
    if not raw:
        raise ConfigError("window section must define at least one window profile")
    flat = set(raw) <= WINDOW_FIELDS
    if flat:
        return {"default": _window_from_fields(raw)}
    named: dict[str, WindowConfig] = {}
    for name, sub in raw.items():
        _require_keys(sub, f"window.{name}")
        unknown = set(sub) - WINDOW_FIELDS
        if unknown:
            raise ConfigError(f"window.{name} has unknown keys {sorted(unknown)}")
        named[str(name)] = _window_from_fields(sub)
    return named


def resolve_paths(cfg: Config) -> Config:
    base = project_root()
    cfg.paths.data_dir = str(Path(cfg.paths.data_dir).expanduser())
    cfg.paths.artifacts_dir = str(Path(cfg.paths.artifacts_dir).expanduser())
    cfg.paths.db_path = str(Path(cfg.paths.db_path).expanduser())
    for path_attr in ("data_dir", "artifacts_dir", "db_path"):
        value = Path(getattr(cfg.paths, path_attr))
        if not value.is_absolute():
            setattr(cfg.paths, path_attr, str(base / value))
    return cfg


def validate(cfg: Config, sampling_rate: float | None) -> None:
    """Cross-field validation executed after the dataset sampling rate is known."""
    if sampling_rate is not None:
        filt = cfg.preprocessing.filter
        if filt.enabled:
            cutoff = filt.cutoff_hz
            if cutoff is None:
                # conservative default: 20% of sampling rate
                cutoff = round(sampling_rate * 0.2, 3)
                filt.cutoff_hz = cutoff
            if cutoff * 2 >= sampling_rate:
                raise ConfigError(
                    f"filter.cutoff_hz={cutoff} violates Nyquist "
                    f"(sampling_rate={sampling_rate})"
                )
        if cfg.window["default"].duration_seconds * sampling_rate < 8:
            raise ConfigError("window covers less than 8 samples")


def load_config(path: str | Path) -> Config:
    """Load and validate a YAML configuration file."""
    path = Path(path)
    if not path.exists():
        raise ConfigError(f"config file not found: {path}")
    with path.open("r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh)
    return config_from_dict(raw)


def config_from_dict(raw: dict[str, Any] | None) -> Config:
    raw = raw or {}
    _require_keys(raw, "config")
    allowed = {
        "project",
        "dataset",
        "window",
        "preprocessing",
        "tda",
        "model",
        "training",
        "evaluation",
        "paths",
    }
    unknown = set(raw) - allowed
    if unknown:
        raise ConfigError(f"config has unknown top-level keys {sorted(unknown)}")

    cfg = Config()

    project = raw.get("project") or {}
    _require_keys(project, "project")
    cfg.project = ProjectConfig(
        name=str(project.get("name", cfg.project.name)),
        seed=int(project.get("seed", cfg.project.seed)),
    )

    dataset = raw.get("dataset") or {}
    _require_keys(dataset, "dataset")
    cfg.dataset = DatasetConfig(
        name=str(dataset.get("name", cfg.dataset.name)),
        root=dataset.get("root"),
        signal_mode=dataset.get("signal_mode"),
        subjects=dataset.get("subjects"),
        channels=dataset.get("channels"),
        activity_filter=dataset.get("activity_filter"),
        sampling_rate_override=dataset.get("sampling_rate_override"),
    )

    cfg.window = parse_window(cfg.window, raw.get("window"))
    cfg.preprocessing = parse_preprocess(raw.get("preprocessing") or {})
    cfg.tda = parse_tda(raw.get("tda") or {})
    cfg.model = parse_model(raw.get("model") or {})
    cfg.training = parse_training(raw.get("training") or {})
    cfg.evaluation = parse_eval(raw.get("evaluation") or {})

    paths_raw = raw.get("paths") or {}
    _require_keys(paths_raw, "paths")
    allowed_paths = {"data_dir", "artifacts_dir", "db_path"}
    unknown_paths = set(paths_raw) - allowed_paths
    if unknown_paths:
        raise ConfigError(f"paths has unknown keys {sorted(unknown_paths)}")
    for key in allowed_paths:
        if key in paths_raw:
            setattr(cfg.paths, key, str(paths_raw[key]))

    if cfg.model.type != "tda_deepconvlstm" and cfg.tda.enabled and cfg.model.type in {
        "deepconvlstm",
        "cnn",
        "lstm",
        "svm",
    }:
        cfg.tda.enabled = False

    resolve_paths(cfg)
    return cfg


def config_to_dict(cfg: Config) -> dict[str, Any]:
    """Serialise back to a plain dict (for artifact dumps)."""

    def asdict_recursive(obj: Any) -> Any:
        if isinstance(obj, dict):
            return {k: asdict_recursive(v) for k, v in obj.items()}
        if hasattr(obj, "__dataclass_fields__"):
            return {k: asdict_recursive(getattr(obj, k)) for k in obj.__dataclass_fields__}
        if isinstance(obj, list):
            return [asdict_recursive(x) for x in obj]
        return obj

    return asdict_recursive(cfg)
