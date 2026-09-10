"""
src/predict.py
Amazon ML Challenge 2026 — Test-set inference engine.

Supports:
  1. Single final model prediction
  2. Fold-averaged prediction (ensemble of saved fold models)
  3. Batch inference for large datasets
  4. Optional post-processing before returning
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable, List, Optional, Union

import numpy as np
import pandas as pd

from src.logging_utils import get_logger
from src.models import BaseModel

logger = get_logger(__name__)


# ─────────────────────────────────────────────────────────────────
# SINGLE MODEL INFERENCE
# ─────────────────────────────────────────────────────────────────
def predict(
    model: BaseModel,
    X_test: Union[pd.DataFrame, np.ndarray],
    postprocess_fn: Optional[Callable[[np.ndarray], np.ndarray]] = None,
    batch_size: Optional[int] = None,
) -> np.ndarray:
    """
    Run inference with a single trained model.

    Parameters
    ----------
    model : BaseModel
    X_test : pd.DataFrame | np.ndarray
    postprocess_fn : callable | None
        Applied to predictions after inference (e.g. clip, expm1).
    batch_size : int | None
        If set, run inference in batches (reduces peak memory for large test sets).

    Returns
    -------
    np.ndarray of shape (n_test,)
    """
    logger.info("Running inference on %d rows with %s.", len(X_test), model.name)

    if batch_size is not None:
        preds = _batch_predict(model, X_test, batch_size)
    else:
        preds = model.predict(X_test)

    if postprocess_fn is not None:
        preds = postprocess_fn(preds)

    logger.info("Prediction stats: mean=%.4f | std=%.4f | min=%.4f | max=%.4f",
                np.mean(preds), np.std(preds), np.min(preds), np.max(preds))
    return preds


# ─────────────────────────────────────────────────────────────────
# FOLD ENSEMBLE INFERENCE
# ─────────────────────────────────────────────────────────────────
def predict_from_fold_models(
    fold_model_paths: List[Union[str, Path]],
    X_test: Union[pd.DataFrame, np.ndarray],
    feature_builder: Optional[Callable] = None,
    train_df: Optional[pd.DataFrame] = None,
    postprocess_fn: Optional[Callable[[np.ndarray], np.ndarray]] = None,
    aggregation: str = "mean",
) -> np.ndarray:
    """
    Average predictions from multiple saved fold models.

    This gives a more stable estimate than a single model, at the cost
    of n_folds× inference time.

    Parameters
    ----------
    fold_model_paths : list[str | Path]
        Paths to serialized fold models (.pkl).
    X_test : pd.DataFrame | np.ndarray
        Test features.
    feature_builder : callable | None
        If features need to be rebuilt from raw test data.
        Signature: fn(test_df, train_df) → feature_df
    train_df : pd.DataFrame | None
        Required if feature_builder needs a reference.
    postprocess_fn : callable | None
    aggregation : str
        "mean" | "median"

    Returns
    -------
    np.ndarray
    """
    all_preds: List[np.ndarray] = []

    for path in fold_model_paths:
        path = Path(path)
        if not path.exists():
            logger.warning("Fold model not found: %s — skipping.", path)
            continue

        model = BaseModel.load(path)  # type: ignore

        if feature_builder is not None and train_df is not None:
            X = feature_builder(X_test, train_df)
        else:
            X = X_test

        preds = model.predict(X)
        if postprocess_fn is not None:
            preds = postprocess_fn(preds)

        all_preds.append(preds)
        logger.info("Fold model %s: pred mean=%.4f", path.name, np.mean(preds))

    if not all_preds:
        raise RuntimeError("No valid fold models found. Cannot generate predictions.")

    arr = np.stack(all_preds, axis=0)   # shape (n_folds, n_test)

    if aggregation == "mean":
        result = arr.mean(axis=0)
    elif aggregation == "median":
        result = np.median(arr, axis=0)
    else:
        raise ValueError(f"Unknown aggregation: '{aggregation}'")

    logger.info(
        "Fold ensemble (%d models, %s): mean=%.4f | std=%.4f",
        len(all_preds), aggregation, result.mean(), result.std(),
    )
    return result


# ─────────────────────────────────────────────────────────────────
# BATCH INFERENCE HELPER
# ─────────────────────────────────────────────────────────────────
def _batch_predict(
    model: BaseModel,
    X: Union[pd.DataFrame, np.ndarray],
    batch_size: int,
) -> np.ndarray:
    """Run prediction in batches to limit peak memory."""
    n = len(X)
    results: List[np.ndarray] = []
    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        batch = X.iloc[start:end] if isinstance(X, pd.DataFrame) else X[start:end]
        results.append(model.predict(batch))
    return np.concatenate(results)
