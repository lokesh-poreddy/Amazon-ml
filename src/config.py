"""
src/config.py
Amazon ML Challenge 2026 — Configuration loader and TaskSpec.

Loads configs/config.yaml and exposes typed dataclasses.
Supports runtime overrides for notebook and CLI usage.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import yaml


# ─────────────────────────────────────────────────────────────────
# DEFAULT PROJECT ROOT
# ─────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent


# ─────────────────────────────────────────────────────────────────
# TASK SPECIFICATION
# ─────────────────────────────────────────────────────────────────
@dataclass
class TaskSpec:
    """
    Captures everything about the competition task.

    Fill this AFTER reading the official problem statement.
    Do NOT guess these values before the dataset is released.

    Parameters
    ----------
    task_type : str
        One of: regression | binary | multiclass | multilabel |
                extraction | ranking | text_classification | image_classification
    target_column : str
        Name of the target column in train data.
    id_column : str
        Name of the row identifier column.
    metric : str
        Primary evaluation metric (e.g. smape, mae, f1, auc).
    direction : str
        "minimize" or "maximize" for the metric.
    prediction_clip_min : float | None
        Minimum allowed prediction value (None = no clip).
    prediction_clip_max : float | None
        Maximum allowed prediction value (None = no clip).
    prediction_column : str
        Name of the column in the submission file.
    num_classes : int | None
        For classification tasks, total number of classes.
    class_names : list[str] | None
        Ordered class names for multiclass/multilabel tasks.
    """

    task_type: str = "UNKNOWN"
    target_column: str = "UNKNOWN"
    id_column: str = "UNKNOWN"
    metric: str = "UNKNOWN"
    direction: str = "minimize"
    prediction_clip_min: Optional[float] = None
    prediction_clip_max: Optional[float] = None
    prediction_column: str = "prediction"
    num_classes: Optional[int] = None
    class_names: Optional[List[str]] = None

    def validate(self) -> None:
        """Raise ValueError if any critical field is still UNKNOWN."""
        unknowns = [
            f for f in ("task_type", "target_column", "id_column", "metric", "direction")
            if getattr(self, f) == "UNKNOWN"
        ]
        if unknowns:
            raise ValueError(
                f"TaskSpec has unresolved fields: {unknowns}. "
                "Please fill them after reading the competition problem statement."
            )

    def is_regression(self) -> bool:
        return self.task_type == "regression"

    def is_classification(self) -> bool:
        return self.task_type in ("binary", "multiclass", "multilabel")

    def is_binary(self) -> bool:
        return self.task_type == "binary"

    def is_multiclass(self) -> bool:
        return self.task_type == "multiclass"

    def is_extraction(self) -> bool:
        return self.task_type == "extraction"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# ─────────────────────────────────────────────────────────────────
# FLAT CONFIG (mirrors config.yaml structure)
# ─────────────────────────────────────────────────────────────────
@dataclass
class ProjectConfig:
    name: str = "amazon-ml-2026"
    seed: int = 42
    debug: bool = False
    final_run: bool = False
    version: str = "v0.1.0"


@dataclass
class DataConfig:
    root_dir: str = "data/raw"
    train_path: Optional[str] = None
    test_path: Optional[str] = None
    target_column: Optional[str] = None
    id_column: Optional[str] = None
    image_dir: Optional[str] = None
    cache_dir: str = "embedding_cache"
    debug_sample: int = 2000
    random_sample_seed: int = 42


@dataclass
class ValidationConfig:
    strategy: str = "kfold"       # kfold | stratified | group | timeseries | holdout
    folds: int = 5
    shuffle: bool = True
    seed: int = 42
    group_column: Optional[str] = None
    time_column: Optional[str] = None
    holdout_fraction: float = 0.2


@dataclass
class PreprocessingConfig:
    numeric_imputer: str = "median"
    categorical_imputer: str = "constant"
    categorical_fill_value: str = "MISSING"
    numeric_fill_value: float = -1.0
    scale_numeric: bool = False
    clip_quantile_outliers: bool = False
    clip_lower: float = 0.001
    clip_upper: float = 0.999


@dataclass
class FeatureConfig:
    version: str = "v0"
    text_columns: List[str] = field(default_factory=list)
    categorical_columns: List[str] = field(default_factory=list)
    numeric_columns: List[str] = field(default_factory=list)
    datetime_columns: List[str] = field(default_factory=list)
    image_url_column: Optional[str] = None
    image_path_column: Optional[str] = None
    use_text_tfidf: bool = False
    use_text_embeddings: bool = False
    use_image_embeddings: bool = False
    tfidf_max_features: int = 50_000
    tfidf_ngram_range: List[int] = field(default_factory=lambda: [1, 2])
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    image_encoder: str = "openai/clip-vit-base-patch32"


@dataclass
class EnsembleConfig:
    enabled: bool = False
    method: str = "weighted_avg"
    correlation_threshold: float = 0.98


@dataclass
class SubmissionConfig:
    output_dir: str = "submission"
    filename: str = "final_submission.csv"
    id_column: Optional[str] = None
    prediction_column: Optional[str] = None
    round_digits: Optional[int] = None


@dataclass
class LoggingConfig:
    level: str = "INFO"
    log_dir: str = "logs"
    experiments_csv: str = "logs/experiments.csv"


@dataclass
class ArtifactConfig:
    models_dir: str = "artifacts/models"
    features_dir: str = "artifacts/features"
    predictions_dir: str = "artifacts/predictions"
    encoders_dir: str = "artifacts/encoders"
    reports_dir: str = "artifacts/reports"


@dataclass
class Config:
    """
    Root configuration object.

    Usage
    -----
    cfg = load_config()                          # from configs/config.yaml
    cfg = load_config("configs/custom.yaml")     # from custom path
    cfg.project.seed                             # 42
    cfg.task.target_column                       # from yaml
    cfg.override({"project.debug": True})        # runtime override
    """

    project: ProjectConfig = field(default_factory=ProjectConfig)
    data: DataConfig = field(default_factory=DataConfig)
    task: TaskSpec = field(default_factory=TaskSpec)
    validation: ValidationConfig = field(default_factory=ValidationConfig)
    preprocessing: PreprocessingConfig = field(default_factory=PreprocessingConfig)
    features: FeatureConfig = field(default_factory=FeatureConfig)
    ensemble: EnsembleConfig = field(default_factory=EnsembleConfig)
    submission: SubmissionConfig = field(default_factory=SubmissionConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    artifacts: ArtifactConfig = field(default_factory=ArtifactConfig)

    def override(self, overrides: Dict[str, Any]) -> None:
        """
        Apply dot-notation overrides at runtime.

        Example
        -------
        cfg.override({"project.debug": True, "validation.folds": 3})
        """
        for key, value in overrides.items():
            parts = key.split(".", maxsplit=1)
            if len(parts) == 1:
                if hasattr(self, parts[0]):
                    setattr(self, parts[0], value)
                else:
                    raise KeyError(f"Config has no top-level key '{parts[0]}'")
            else:
                top, rest = parts
                sub = getattr(self, top, None)
                if sub is None:
                    raise KeyError(f"Config has no section '{top}'")
                sub_parts = rest.split(".", maxsplit=1)
                if len(sub_parts) == 1:
                    if hasattr(sub, sub_parts[0]):
                        setattr(sub, sub_parts[0], value)
                    else:
                        raise KeyError(f"Config section '{top}' has no key '{sub_parts[0]}'")
                else:
                    raise NotImplementedError("Deep overrides (>2 levels) are not supported.")

    def make_dirs(self) -> None:
        """Create all artifact and log directories relative to PROJECT_ROOT."""
        dirs = [
            self.data.root_dir,
            self.data.cache_dir,
            self.logging.log_dir,
            self.submission.output_dir,
            self.artifacts.models_dir,
            self.artifacts.features_dir,
            self.artifacts.predictions_dir,
            self.artifacts.encoders_dir,
            self.artifacts.reports_dir,
            "data/raw",
            "data/interim",
            "data/processed",
            "logs/runs",
        ]
        for d in dirs:
            Path(d).mkdir(parents=True, exist_ok=True)


# ─────────────────────────────────────────────────────────────────
# YAML → CONFIG LOADER
# ─────────────────────────────────────────────────────────────────
def _deep_update(base: dict, update: dict) -> dict:
    """Recursively merge update into base (in-place)."""
    for k, v in update.items():
        if isinstance(v, dict) and k in base and isinstance(base[k], dict):
            _deep_update(base[k], v)
        else:
            base[k] = v
    return base


def _yaml_to_config(raw: dict) -> Config:
    """Convert a raw YAML dict into a typed Config object."""

    def _get(section: str, default: dict | None = None) -> dict:
        return raw.get(section, default or {})

    task_raw = _get("task")
    data_raw = _get("data")
    task_spec = TaskSpec(
        task_type=task_raw.get("type") or "UNKNOWN",
        target_column=data_raw.get("target_column") or "UNKNOWN",
        id_column=data_raw.get("id_column") or "UNKNOWN",
        metric=task_raw.get("metric") or "UNKNOWN",
        direction=task_raw.get("direction") or "minimize",
        prediction_clip_min=task_raw.get("prediction_clip_min"),
        prediction_clip_max=task_raw.get("prediction_clip_max"),
    )

    val_raw = _get("validation")
    pre_raw = _get("preprocessing")
    feat_raw = _get("features")
    ens_raw = _get("ensemble")
    sub_raw = _get("submission")
    log_raw = _get("logging")
    art_raw = _get("artifacts")
    proj_raw = _get("project")

    return Config(
        project=ProjectConfig(
            name=proj_raw.get("name", "amazon-ml-2026"),
            seed=proj_raw.get("seed", 42),
            debug=proj_raw.get("debug", False),
            final_run=proj_raw.get("final_run", False),
            version=proj_raw.get("version", "v0.1.0"),
        ),
        data=DataConfig(
            root_dir=data_raw.get("root_dir", "data/raw"),
            train_path=data_raw.get("train_path"),
            test_path=data_raw.get("test_path"),
            target_column=data_raw.get("target_column"),
            id_column=data_raw.get("id_column"),
            image_dir=data_raw.get("image_dir"),
            cache_dir=data_raw.get("cache_dir", "embedding_cache"),
            debug_sample=data_raw.get("debug_sample", 2000),
            random_sample_seed=data_raw.get("random_sample_seed", 42),
        ),
        task=task_spec,
        validation=ValidationConfig(
            strategy=val_raw.get("strategy", "kfold"),
            folds=val_raw.get("folds", 5),
            shuffle=val_raw.get("shuffle", True),
            seed=val_raw.get("seed", 42),
            group_column=val_raw.get("group_column"),
            time_column=val_raw.get("time_column"),
            holdout_fraction=val_raw.get("holdout_fraction", 0.2),
        ),
        preprocessing=PreprocessingConfig(
            numeric_imputer=pre_raw.get("numeric_imputer", "median"),
            categorical_imputer=pre_raw.get("categorical_imputer", "constant"),
            categorical_fill_value=pre_raw.get("categorical_fill_value", "MISSING"),
            numeric_fill_value=float(pre_raw.get("numeric_fill_value", -1)),
            scale_numeric=pre_raw.get("scale_numeric", False),
            clip_quantile_outliers=pre_raw.get("clip_quantile_outliers", False),
            clip_lower=pre_raw.get("clip_lower", 0.001),
            clip_upper=pre_raw.get("clip_upper", 0.999),
        ),
        features=FeatureConfig(
            version=feat_raw.get("version", "v0"),
            text_columns=feat_raw.get("text_columns") or [],
            categorical_columns=feat_raw.get("categorical_columns") or [],
            numeric_columns=feat_raw.get("numeric_columns") or [],
            datetime_columns=feat_raw.get("datetime_columns") or [],
            image_url_column=feat_raw.get("image_url_column"),
            image_path_column=feat_raw.get("image_path_column"),
            use_text_tfidf=feat_raw.get("use_text_tfidf", False),
            use_text_embeddings=feat_raw.get("use_text_embeddings", False),
            use_image_embeddings=feat_raw.get("use_image_embeddings", False),
            tfidf_max_features=feat_raw.get("tfidf_max_features", 50_000),
            tfidf_ngram_range=feat_raw.get("tfidf_ngram_range", [1, 2]),
            embedding_model=feat_raw.get(
                "embedding_model", "sentence-transformers/all-MiniLM-L6-v2"
            ),
            image_encoder=feat_raw.get(
                "image_encoder", "openai/clip-vit-base-patch32"
            ),
        ),
        ensemble=EnsembleConfig(
            enabled=ens_raw.get("enabled", False),
            method=ens_raw.get("method", "weighted_avg"),
            correlation_threshold=ens_raw.get("correlation_threshold", 0.98),
        ),
        submission=SubmissionConfig(
            output_dir=sub_raw.get("output_dir", "submission"),
            filename=sub_raw.get("filename", "final_submission.csv"),
            id_column=sub_raw.get("id_column"),
            prediction_column=sub_raw.get("prediction_column"),
            round_digits=sub_raw.get("round_digits"),
        ),
        logging=LoggingConfig(
            level=log_raw.get("level", "INFO"),
            log_dir=log_raw.get("log_dir", "logs"),
            experiments_csv=log_raw.get("experiments_csv", "logs/experiments.csv"),
        ),
        artifacts=ArtifactConfig(
            models_dir=art_raw.get("models_dir", "artifacts/models"),
            features_dir=art_raw.get("features_dir", "artifacts/features"),
            predictions_dir=art_raw.get("predictions_dir", "artifacts/predictions"),
            encoders_dir=art_raw.get("encoders_dir", "artifacts/encoders"),
            reports_dir=art_raw.get("reports_dir", "artifacts/reports"),
        ),
    )


def load_config(
    path: Optional[Union[str, Path]] = None,
    overrides: Optional[Dict[str, Any]] = None,
) -> Config:
    """
    Load configuration from a YAML file.

    Parameters
    ----------
    path : str | Path | None
        Path to the YAML config file.
        Defaults to ``configs/config.yaml`` relative to PROJECT_ROOT.
    overrides : dict | None
        Runtime overrides using dot-notation, e.g.
        {"project.debug": True, "validation.folds": 3}.

    Returns
    -------
    Config
        Fully typed configuration object.
    """
    if path is None:
        path = PROJECT_ROOT / "configs" / "config.yaml"

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Config file not found: {path}\n"
            f"Expected location: {PROJECT_ROOT / 'configs' / 'config.yaml'}"
        )

    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}

    cfg = _yaml_to_config(raw)

    if overrides:
        cfg.override(overrides)

    return cfg


# ─────────────────────────────────────────────────────────────────
# CONVENIENCE SINGLETON
# ─────────────────────────────────────────────────────────────────
_global_config: Optional[Config] = None


def get_config() -> Config:
    """Return the global config singleton, loading defaults if needed."""
    global _global_config
    if _global_config is None:
        try:
            _global_config = load_config()
        except FileNotFoundError:
            _global_config = Config()
    return _global_config


def set_config(cfg: Config) -> None:
    """Replace the global config singleton (useful in notebooks)."""
    global _global_config
    _global_config = cfg
