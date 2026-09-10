"""
src/config.py
Amazon ML Challenge 2026 — Typed configuration system.

Design principles:
  1. TaskSpec is the single source of truth for target_column, id_column,
     prediction_column, task_type, metric, direction, output_format.
  2. DataConfig owns only paths and cache locations — NO task semantics.
  3. direction=null raises ConfigError at the point of use, never silently
     defaults to "minimize". Forgetting to fill this in is a disqualifying
     mistake if the competition uses accuracy/AUC (maximize).
  4. All relative paths are resolved through PROJECT_ROOT so the framework
     works regardless of the caller's working directory.

Usage:
    cfg = load_config()                    # loads configs/config.yaml
    cfg = load_config("my_config.yaml")    # custom path
    cfg = load_config(overrides={"task.metric": "mae"})

    # Access:
    cfg.task.target_column   # str, e.g. "entity_value"
    cfg.data.train_path      # Path (resolved)
    cfg.task.direction_safe  # raises ConfigError if null
"""

from __future__ import annotations

import copy
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

from src.logging_utils import get_logger

logger = get_logger(__name__)

# ─────────────────────────────────────────────────────────────────
# PROJECT ROOT
# ─────────────────────────────────────────────────────────────────
_THIS_FILE   = Path(__file__).resolve()
PROJECT_ROOT = _THIS_FILE.parent.parent   # src/ is one level below root


def resolve_path(p: Optional[str], must_exist: bool = False) -> Optional[Path]:
    """
    Resolve a path relative to PROJECT_ROOT.

    Parameters
    ----------
    p : str | None
    must_exist : bool
        If True, raises FileNotFoundError when the resolved path doesn't exist.

    Returns
    -------
    Path (absolute) or None if p is None.
    """
    if p is None:
        return None
    path = Path(p)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    path = path.resolve()
    if must_exist and not path.exists():
        raise FileNotFoundError(
            f"Configured path does not exist: {path}\n"
            f"  (original value: '{p}', PROJECT_ROOT: {PROJECT_ROOT})"
        )
    return path


# ─────────────────────────────────────────────────────────────────
# EXCEPTIONS
# ─────────────────────────────────────────────────────────────────
class ConfigError(ValueError):
    """Raised when a safety-critical configuration field is unresolved."""


# ─────────────────────────────────────────────────────────────────
# DATA CONFIG — paths only
# ─────────────────────────────────────────────────────────────────
@dataclass
class DataConfig:
    """
    Owns dataset file paths and cache locations only.
    No task semantics (target, id) live here.
    """
    train_path:      Optional[Path] = None
    test_path:       Optional[Path] = None
    image_dir:       Optional[Path] = None
    cache_dir:       Path           = field(default_factory=lambda: PROJECT_ROOT / "embedding_cache")
    artifacts_dir:   Path           = field(default_factory=lambda: PROJECT_ROOT / "artifacts")
    submission_dir:  Path           = field(default_factory=lambda: PROJECT_ROOT / "submission")
    logs_dir:        Path           = field(default_factory=lambda: PROJECT_ROOT / "logs")

    def ensure_dirs(self) -> None:
        """Create all configured directories."""
        for d in [self.cache_dir, self.artifacts_dir, self.submission_dir, self.logs_dir]:
            if d is not None:
                d.mkdir(parents=True, exist_ok=True)


# ─────────────────────────────────────────────────────────────────
# TASK SPEC — single source of truth for the task contract
# ─────────────────────────────────────────────────────────────────
@dataclass
class TaskSpec:
    """
    Defines everything about what we are optimising.

    This is the single authoritative contract for the task.
    Nothing else should carry target_column or id_column.

    direction is deliberately not defaulted. Calling direction_safe
    when direction is None raises ConfigError immediately.
    Forgetting to set this is a disqualifying error.
    """
    task_type:          Optional[str] = None   # regression | binary | multiclass
    target_column:      Optional[str] = None
    id_column:          Optional[str] = None
    prediction_column:  Optional[str] = "prediction"
    metric:             Optional[str] = None   # smape | mae | auc | f1 | ...
    direction:          Optional[str] = None   # minimize | maximize — NO DEFAULT
    output_format:      str           = "csv"
    is_multilabel:      bool          = False

    # ── Safe accessors (raise on unresolved) ─────────────────────
    @property
    def direction_safe(self) -> str:
        if self.direction not in ("minimize", "maximize"):
            raise ConfigError(
                f"task.direction is unresolved (got: {self.direction!r}).\n"
                f"Set it in configs/config.yaml or pass an override:\n"
                f"  direction: minimize   # or maximize\n"
                f"This is safety-critical. A wrong direction can silently "
                f"optimise the wrong way for 72 hours."
            )
        return self.direction

    @property
    def target_column_safe(self) -> str:
        if not self.target_column:
            raise ConfigError(
                "task.target_column is not set. Fill it in configs/config.yaml "
                "after reading the problem statement."
            )
        return self.target_column

    @property
    def id_column_safe(self) -> str:
        if not self.id_column:
            raise ConfigError(
                "task.id_column is not set. Fill it in configs/config.yaml "
                "after reading the problem statement."
            )
        return self.id_column

    @property
    def is_resolved(self) -> bool:
        """True when the task spec is complete enough to train."""
        return all([
            self.task_type,
            self.target_column,
            self.id_column,
            self.metric,
            self.direction in ("minimize", "maximize"),
        ])

    def assert_resolved(self) -> None:
        """Raise ConfigError listing all missing fields."""
        missing = []
        if not self.task_type:         missing.append("task.type")
        if not self.target_column:     missing.append("task.target_column")
        if not self.id_column:         missing.append("task.id_column")
        if not self.metric:            missing.append("task.metric")
        if self.direction not in ("minimize", "maximize"):
            missing.append("task.direction")
        if missing:
            raise ConfigError(
                "Task spec is not fully resolved. Missing fields:\n"
                + "\n".join(f"  - {f}" for f in missing)
                + "\n\nFill these in configs/config.yaml after reading the "
                  "competition problem statement."
            )


# ─────────────────────────────────────────────────────────────────
# SPLIT CONFIG
# ─────────────────────────────────────────────────────────────────
@dataclass
class SplitConfig:
    strategy:    str           = "kfold"
    n_splits:    int           = 5
    shuffle:     bool          = True
    seed:        int           = 42
    group_col:   Optional[str] = None
    time_col:    Optional[str] = None
    holdout_frac: float        = 0.2


# ─────────────────────────────────────────────────────────────────
# MODEL CONFIG
# ─────────────────────────────────────────────────────────────────
@dataclass
class ModelConfig:
    primary:                str            = "lgbm"
    candidates:             List[str]      = field(default_factory=lambda: ["lgbm"])
    early_stopping_rounds:  int            = 100
    lgbm_params:            Dict[str, Any] = field(default_factory=dict)
    xgb_params:             Dict[str, Any] = field(default_factory=dict)
    catboost_params:        Dict[str, Any] = field(default_factory=dict)


# ─────────────────────────────────────────────────────────────────
# FEATURE CONFIG
# ─────────────────────────────────────────────────────────────────
@dataclass
class FeatureConfig:
    numeric_cols:           List[str] = field(default_factory=list)
    categorical_cols:       List[str] = field(default_factory=list)
    text_cols:              List[str] = field(default_factory=list)
    image_url_column:       Optional[str] = None
    image_path_column:      Optional[str] = None
    use_tfidf:              bool = True
    tfidf_max_features:     int  = 50_000
    tfidf_ngram_range:      List[int] = field(default_factory=lambda: [1, 2])
    use_text_structured:    bool = True
    use_image_embeddings:   bool = False
    image_encoder:          str  = "openai/clip-vit-base-patch32"
    use_text_embeddings:    bool = False
    text_encoder:           str  = "sentence-transformers/all-MiniLM-L6-v2"


# ─────────────────────────────────────────────────────────────────
# MASTER CONFIG
# ─────────────────────────────────────────────────────────────────
@dataclass
class Config:
    data:     DataConfig    = field(default_factory=DataConfig)
    task:     TaskSpec      = field(default_factory=TaskSpec)
    split:    SplitConfig   = field(default_factory=SplitConfig)
    model:    ModelConfig   = field(default_factory=ModelConfig)
    features: FeatureConfig = field(default_factory=FeatureConfig)
    seed:     int           = 42
    debug:    bool          = False
    final_run: bool         = False

    @property
    def experiment_id_prefix(self) -> str:
        return f"{self.task.task_type or 'UNKNOWN'}_{self.task.metric or 'UNKNOWN'}"


# ─────────────────────────────────────────────────────────────────
# YAML LOADER
# ─────────────────────────────────────────────────────────────────
_DEFAULT_CONFIG_PATH = PROJECT_ROOT / "configs" / "config.yaml"


def _apply_overrides(raw: Dict, overrides: Dict[str, Any]) -> Dict:
    """Apply dot-notation overrides to the raw dict. E.g. 'task.metric' → 'mae'."""
    for dotkey, value in overrides.items():
        parts = dotkey.split(".")
        d = raw
        for part in parts[:-1]:
            d = d.setdefault(part, {})
        d[parts[-1]] = value
    return raw


def load_config(
    path: Optional[str] = None,
    overrides: Optional[Dict[str, Any]] = None,
) -> Config:
    """
    Load and parse config.yaml into a typed Config object.

    Parameters
    ----------
    path : str | None
        Path to YAML file. Defaults to configs/config.yaml relative to PROJECT_ROOT.
    overrides : dict | None
        Dot-notation runtime overrides, e.g. {"task.metric": "mae"}.

    Returns
    -------
    Config
    """
    config_path = resolve_path(path) if path else _DEFAULT_CONFIG_PATH

    if not config_path.exists():
        logger.warning("Config file not found at %s. Using defaults.", config_path)
        raw: Dict[str, Any] = {}
    else:
        with open(config_path, "r", encoding="utf-8") as f:
            raw = yaml.safe_load(f) or {}
        logger.info("Loaded config from %s", config_path)

    if overrides:
        raw = _apply_overrides(copy.deepcopy(raw), overrides)

    # ── Parse data section ────────────────────────────────────────
    data_raw = raw.get("data", {}) or {}
    data_cfg = DataConfig(
        train_path     = resolve_path(data_raw.get("train_path")),
        test_path      = resolve_path(data_raw.get("test_path")),
        image_dir      = resolve_path(data_raw.get("image_dir")),
        cache_dir      = resolve_path(data_raw.get("cache_dir")) or (PROJECT_ROOT / "embedding_cache"),
        artifacts_dir  = resolve_path(data_raw.get("artifacts_dir")) or (PROJECT_ROOT / "artifacts"),
        submission_dir = resolve_path(data_raw.get("submission_dir")) or (PROJECT_ROOT / "submission"),
        logs_dir       = resolve_path(data_raw.get("logs_dir")) or (PROJECT_ROOT / "logs"),
    )

    # ── Parse task section ────────────────────────────────────────
    task_raw = raw.get("task", {}) or {}
    task_cfg = TaskSpec(
        task_type         = task_raw.get("type") or None,
        target_column     = task_raw.get("target_column") or None,
        id_column         = task_raw.get("id_column") or None,
        prediction_column = task_raw.get("prediction_column") or "prediction",
        metric            = task_raw.get("metric") or None,
        direction         = task_raw.get("direction") or None,  # NO SILENT DEFAULT
        output_format     = task_raw.get("output_format") or "csv",
        is_multilabel     = bool(task_raw.get("is_multilabel", False)),
    )

    # ── Parse split section ───────────────────────────────────────
    split_raw = raw.get("split", {}) or {}
    split_cfg = SplitConfig(
        strategy     = split_raw.get("strategy", "kfold"),
        n_splits     = int(split_raw.get("n_splits", 5)),
        shuffle      = bool(split_raw.get("shuffle", True)),
        seed         = int(split_raw.get("seed", 42)),
        group_col    = split_raw.get("group_col"),
        time_col     = split_raw.get("time_col"),
        holdout_frac = float(split_raw.get("holdout_frac", 0.2)),
    )

    # ── Parse model section ───────────────────────────────────────
    model_raw = raw.get("model", {}) or {}
    model_cfg = ModelConfig(
        primary                = model_raw.get("primary", "lgbm"),
        candidates             = model_raw.get("candidates", ["lgbm"]),
        early_stopping_rounds  = int(model_raw.get("early_stopping_rounds", 100)),
        lgbm_params            = dict(model_raw.get("lgbm_params", {})),
        xgb_params             = dict(model_raw.get("xgb_params", {})),
        catboost_params        = dict(model_raw.get("catboost_params", {})),
    )

    # ── Parse feature section ─────────────────────────────────────
    feat_raw = raw.get("features", {}) or {}
    feat_cfg = FeatureConfig(
        numeric_cols         = list(feat_raw.get("numeric_cols", [])),
        categorical_cols     = list(feat_raw.get("categorical_cols", [])),
        text_cols            = list(feat_raw.get("text_cols", [])),
        image_url_column     = feat_raw.get("image_url_column"),
        image_path_column    = feat_raw.get("image_path_column"),
        use_tfidf            = bool(feat_raw.get("use_tfidf", True)),
        tfidf_max_features   = int(feat_raw.get("tfidf_max_features", 50_000)),
        tfidf_ngram_range    = list(feat_raw.get("tfidf_ngram_range", [1, 2])),
        use_text_structured  = bool(feat_raw.get("use_text_structured", True)),
        use_image_embeddings = bool(feat_raw.get("use_image_embeddings", False)),
        image_encoder        = feat_raw.get("image_encoder", "openai/clip-vit-base-patch32"),
        use_text_embeddings  = bool(feat_raw.get("use_text_embeddings", False)),
        text_encoder         = feat_raw.get("text_encoder", "sentence-transformers/all-MiniLM-L6-v2"),
    )

    # ── Assemble master config ────────────────────────────────────
    project_raw = raw.get("project", {}) or {}
    cfg = Config(
        data      = data_cfg,
        task      = task_cfg,
        split     = split_cfg,
        model     = model_cfg,
        features  = feat_cfg,
        seed      = int(project_raw.get("seed", 42)),
        debug     = bool(project_raw.get("debug", False)),
        final_run = bool(project_raw.get("final_run", False)),
    )

    # ── Log resolved state ────────────────────────────────────────
    status = "RESOLVED ✅" if cfg.task.is_resolved else "⚠️  UNRESOLVED (fill config.yaml)"
    logger.info("TaskSpec status: %s", status)
    if not cfg.task.is_resolved:
        logger.warning(
            "Task spec is not fully resolved. This is expected on Day 1 before "
            "reading the problem statement. Fill in configs/config.yaml before training."
        )

    return cfg
