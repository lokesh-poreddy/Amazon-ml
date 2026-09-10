"""
src/train.py
Amazon ML Challenge 2026 — Cross-validation training loop with OOF predictions.

Core workflow:
  1. Accept splits from split.py
  2. Build features INSIDE each fold (fold-aware — no leakage)
  3. Fit model on train fold
  4. Evaluate on validation fold
  5. Collect out-of-fold (OOF) predictions
  6. Save fold models and OOF predictions
  7. Return a TrainingResult summary

The function does NOT make assumptions about the task type —
it delegates metric computation to the MetricSpec passed in.
"""

from __future__ import annotations

import gc
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

from src.logging_utils import get_logger
from src.metrics import MetricSpec, evaluate
from src.models import BaseModel, build_model
from src.split import SplitInfo
from src.utils import Timer, ensure_dir, now_str, save_json

logger = get_logger(__name__)


# ─────────────────────────────────────────────────────────────────
# TRAINING RESULT
# ─────────────────────────────────────────────────────────────────
@dataclass
class FoldResult:
    fold: int
    val_score: float
    n_train: int
    n_val: int
    elapsed_seconds: float
    model_path: Optional[str] = None
    best_iteration: Optional[int] = None


@dataclass
class TrainingResult:
    """Summary of a completed CV training run."""

    experiment_id: str
    model_name: str
    feature_version: str
    split_version: str
    metric_name: str
    fold_results: List[FoldResult]
    oof_predictions: np.ndarray
    oof_indices: np.ndarray
    cv_mean: float
    cv_std: float
    total_elapsed: float
    suggested_final_n_estimators: Optional[int] = None
    config_snapshot: Dict[str, Any] = field(default_factory=dict)

    def summary(self) -> str:
        lines = [
            f"Experiment   : {self.experiment_id}",
            f"Model        : {self.model_name}",
            f"Metric       : {self.metric_name}",
            f"CV Mean      : {self.cv_mean:.6f}",
            f"CV Std       : {self.cv_std:.6f}",
            f"CV ±         : {self.cv_mean:.6f} ± {self.cv_std:.6f}",
            f"Time         : {self.total_elapsed:.1f}s",
        ]
        for fr in self.fold_results:
            lines.append(f"  Fold {fr.fold}: {fr.val_score:.6f} ({fr.elapsed_seconds:.1f}s)")
        return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────
# FEATURE BUILDER PROTOCOL
# ─────────────────────────────────────────────────────────────────
FeatureBuilderFn = Callable[[pd.DataFrame, Optional[pd.DataFrame]], pd.DataFrame]
"""
Callable(df, train_df) → feature DataFrame.

The caller is responsible for passing train_df = training fold only.
See src/features.py for reference implementations.
"""


# ─────────────────────────────────────────────────────────────────
# CROSS-VALIDATION LOOP
# ─────────────────────────────────────────────────────────────────
def cross_validate(
    df: pd.DataFrame,
    target_col: str,
    splits: List[SplitInfo],
    model_name: str,
    metric_spec: MetricSpec,
    feature_builder: Optional[FeatureBuilderFn] = None,
    model_params: Optional[Dict[str, Any]] = None,
    task_type: str = "regression",
    model_kwargs: Optional[Dict[str, Any]] = None,
    experiment_id: Optional[str] = None,
    feature_version: str = "v0",
    split_version: str = "v0",
    artifacts_dir: Optional[str] = "artifacts",
    save_models: bool = True,
    predict_proba: bool = False,
    postprocess_fn: Optional[Callable[[np.ndarray], np.ndarray]] = None,
) -> TrainingResult:
    """
    Run k-fold cross-validation with out-of-fold prediction collection.

    Parameters
    ----------
    df : pd.DataFrame
        Full training data (with target).
    target_col : str
        Name of the target column.
    splits : list[SplitInfo]
        From split.make_splits().
    model_name : str
        Registered model name ("lgbm" | "xgb" | "catboost" | "linear" | "dummy").
    metric_spec : MetricSpec
        From metrics.get_metric().
    feature_builder : callable | None
        fn(df, train_df) → pd.DataFrame of features.
        If None, df (minus target) is used as-is.
    model_params : dict | None
        Hyperparameters passed to the model.
    task_type : str
        "regression" | "binary" | "multiclass"
    model_kwargs : dict | None
        Additional kwargs forwarded to build_model().
    experiment_id : str | None
        Unique experiment identifier. Auto-generated if None.
    feature_version, split_version : str
        For bookkeeping in the result.
    artifacts_dir : str | None
        Directory to save fold models.
    save_models : bool
        If False, fold models are not serialized (faster iteration).
    predict_proba : bool
        Collect probabilities instead of class predictions (classification only).
    postprocess_fn : callable | None
        Applied to predictions before metric computation (e.g., clip, expm1).

    Returns
    -------
    TrainingResult
    """
    if experiment_id is None:
        experiment_id = f"E_{now_str()}"

    if target_col not in df.columns:
        raise ValueError(
            f"Target column '{target_col}' not found. "
            f"Available columns: {list(df.columns)}"
        )

    y_all = df[target_col].values
    n = len(df)
    oof_preds = None  # Lazy init to handle (N, K) multiclass shape
    oof_idx   = np.zeros(n, dtype=np.int64)
    validation_count = np.zeros(n, dtype=np.int32)
    fold_results: List[FoldResult] = []

    t_start = time.perf_counter()
    models_dir = Path(artifacts_dir) / "models" / experiment_id if artifacts_dir else None

    for si in splits:
        fold = si.fold
        t_fold = time.perf_counter()

        train_df = df.iloc[si.train_idx].reset_index(drop=True)
        val_df   = df.iloc[si.val_idx].reset_index(drop=True)

        y_train = train_df[target_col].values
        y_val   = val_df[target_col].values

        # ── Build features (fold-aware) ───────────────────────────
        if feature_builder is not None:
            X_train = feature_builder(train_df, train_df)
            X_val   = feature_builder(val_df, train_df)  # reference = train fold
        else:
            drop_cols = [c for c in [target_col] if c in train_df.columns]
            X_train = train_df.drop(columns=drop_cols)
            X_val   = val_df.drop(columns=drop_cols)

        # ── Fit model ─────────────────────────────────────────────
        model = build_model(
            model_name,
            params=model_params,
            task_type=task_type,
            **(model_kwargs or {}),
        )
        model.fit(X_train, y_train, X_val=X_val, y_val=y_val)

        # ── Predict ───────────────────────────────────────────────
        if predict_proba and task_type != "regression":
            preds = model.predict_proba(X_val)
            if task_type == "binary" and preds is not None and preds.ndim == 2:
                preds = preds[:, 1]  # binary: take positive class
        else:
            preds = model.predict(X_val)

        if postprocess_fn is not None:
            preds = postprocess_fn(preds)

        # ── Score ─────────────────────────────────────────────────
        score = metric_spec(y_val, preds)
        elapsed = time.perf_counter() - t_fold

        logger.info(
            "Fold %d | %s=%.6f | train=%d | val=%d | %.1fs",
            fold, metric_spec.name, score, len(train_df), len(val_df), elapsed,
        )

        # ── Store OOF ─────────────────────────────────────────────
        if oof_preds is None:
            if preds.ndim == 1:
                oof_preds = np.zeros(n, dtype=preds.dtype)
            else:
                oof_preds = np.zeros((n, preds.shape[1]), dtype=preds.dtype)
                
        oof_preds[si.val_idx] = preds
        oof_idx[si.val_idx]   = si.val_idx
        validation_count[si.val_idx] += 1

        # ── Save model ────────────────────────────────────────────
        model_path = None
        if save_models and models_dir is not None:
            ensure_dir(models_dir)
            model_path = str(models_dir / f"fold_{fold}.pkl")
            model.save(model_path)

        # ── Extract best iteration ────────────────────────────────
        best_iter = None
        if hasattr(model, "_model") and model._model is not None:
            if hasattr(model._model, "best_iteration_"):
                best_iter = model._model.best_iteration_
            elif hasattr(model._model, "best_iteration"):
                best_iter = model._model.best_iteration
            elif hasattr(model._model, "get_best_iteration"):
                best_iter = model._model.get_best_iteration()

        fold_results.append(FoldResult(
            fold=fold,
            val_score=score,
            n_train=len(train_df),
            n_val=len(val_df),
            elapsed_seconds=elapsed,
            model_path=model_path,
            best_iteration=best_iter,
        ))

        del model, X_train, X_val, train_df, val_df
        gc.collect()

    # ── Aggregate ─────────────────────────────────────────────────
    if np.any(validation_count != 1):
        missing = np.sum(validation_count == 0)
        duplicates = np.sum(validation_count > 1)
        # Note: TimeSeriesSplit intentionally leaves out early rows from validation
        if not any(s.strategy == "timeseries" for s in splits):
            raise RuntimeError(f"OOF coverage violation: {missing} rows missed, {duplicates} rows predicted multiple times. Every row must be in exactly one validation fold.")
        else:
            logger.warning("OOF coverage: %d rows missed (expected for TimeSeriesSplit).", missing)

    scores = [fr.val_score for fr in fold_results]
    cv_mean = float(np.mean(scores))
    cv_std  = float(np.std(scores))
    total_elapsed = time.perf_counter() - t_start
    
    iters = [fr.best_iteration for fr in fold_results if fr.best_iteration is not None]
    suggested_n_estimators = int(round(np.median(iters) * 1.05)) if iters else None

    result = TrainingResult(
        experiment_id=experiment_id,
        model_name=model_name,
        feature_version=feature_version,
        split_version=split_version,
        metric_name=metric_spec.name,
        fold_results=fold_results,
        oof_predictions=oof_preds,
        oof_indices=oof_idx,
        cv_mean=cv_mean,
        cv_std=cv_std,
        total_elapsed=total_elapsed,
        suggested_final_n_estimators=suggested_n_estimators,
    )

    logger.info("\n%s\n%s", "=" * 55, result.summary())
    return result


# ─────────────────────────────────────────────────────────────────
# FULL TRAIN (all data, no CV — for final model only)
# ─────────────────────────────────────────────────────────────────
def train_final_model(
    df: pd.DataFrame,
    target_col: str,
    model_name: str,
    feature_builder: Optional[FeatureBuilderFn] = None,
    model_params: Optional[Dict[str, Any]] = None,
    task_type: str = "regression",
    model_kwargs: Optional[Dict[str, Any]] = None,
    save_path: Optional[str] = None,
    n_estimators: Optional[int] = None,
) -> BaseModel:
    """
    Train a single model on the FULL training dataset.

    Use ONLY after validating via cross_validate().
    This model is used for final test-set inference.

    Parameters
    ----------
    df : pd.DataFrame
        Full training data.
    target_col : str
    model_name : str
    feature_builder : callable | None
    model_params : dict | None
    task_type : str
    model_kwargs : dict | None
    save_path : str | None
        If provided, save the model here.

    Returns
    -------
    BaseModel
    """
    y = df[target_col].values
    if feature_builder is not None:
        X = feature_builder(df, df)
    else:
        drop_cols = [c for c in [target_col] if c in df.columns]
        X = df.drop(columns=drop_cols)

    if n_estimators is None:
        logger.warning("train_final_model called without n_estimators! Model may overfit or underfit. Use result.suggested_final_n_estimators.")
    else:
        model_params = model_params or {}
        if model_name.lower() in ("catboost",):
            model_params["iterations"] = n_estimators
        else:
            model_params["n_estimators"] = n_estimators

    model = build_model(
        model_name,
        params=model_params,
        task_type=task_type,
        **(model_kwargs or {}),
    )
    # No validation set → no early stopping
    model.fit(X, y)

    if save_path:
        model.save(save_path)
        logger.info("Final model saved → %s", save_path)

    return model
