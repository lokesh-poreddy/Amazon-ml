"""
src/metrics.py
Amazon ML Challenge 2026 — Unified metric framework.

Design:
  - MetricSpec ties together name, direction, and implementation
  - Compute functions are plain numpy — no sklearn required
  - get_metric() returns a MetricSpec by name string
  - evaluate() computes a metric and returns a named result

For Amazon 2025 the primary metric was SMAPE.
Do NOT assume the 2026 metric — fill task.metric after reading
the official problem statement.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional

import numpy as np
import pandas as pd
from sklearn import metrics as sk_metrics

from src.logging_utils import get_logger

logger = get_logger(__name__)


# ─────────────────────────────────────────────────────────────────
# METRIC SPECIFICATION
# ─────────────────────────────────────────────────────────────────
@dataclass
class MetricSpec:
    """
    Complete description of a competition metric.

    Parameters
    ----------
    name : str
        Short identifier (e.g. "smape").
    direction : str
        "minimize" or "maximize".
    fn : Callable
        Function f(y_true, y_pred) → float.
    description : str
        Human-readable description.
    """

    name: str
    direction: str
    fn: Callable[[Any, Any], float]
    description: str = ""

    def __call__(self, y_true: Any, y_pred: Any) -> float:
        return self.fn(y_true, y_pred)

    def is_better(self, score_a: float, score_b: float) -> bool:
        """Return True if score_a is better than score_b."""
        if self.direction == "minimize":
            return score_a < score_b
        return score_a > score_b

    def best_of(self, scores: list) -> float:
        if self.direction == "minimize":
            return float(min(scores))
        return float(max(scores))

    def worst_value(self) -> float:
        return float("inf") if self.direction == "minimize" else float("-inf")


# ─────────────────────────────────────────────────────────────────
# METRIC RESULT
# ─────────────────────────────────────────────────────────────────
@dataclass
class MetricResult:
    metric_name: str
    score: float
    direction: str
    n_samples: int
    extra: Dict[str, float] = None  # type: ignore

    def __post_init__(self):
        if self.extra is None:
            self.extra = {}

    def __str__(self) -> str:
        return f"{self.metric_name}: {self.score:.6f} (n={self.n_samples})"


# ─────────────────────────────────────────────────────────────────
# INDIVIDUAL METRIC IMPLEMENTATIONS
# ─────────────────────────────────────────────────────────────────
def _to_arrays(y_true: Any, y_pred: Any):
    """Convert inputs to clean numpy float64 arrays."""
    y_true = np.asarray(y_true, dtype=np.float64).ravel()
    y_pred = np.asarray(y_pred, dtype=np.float64).ravel()
    if len(y_true) != len(y_pred):
        raise ValueError(
            f"Length mismatch: y_true has {len(y_true)} rows, "
            f"y_pred has {len(y_pred)} rows."
        )
    return y_true, y_pred


# Regression ──────────────────────────────────────────────────────
def smape(y_true: Any, y_pred: Any) -> float:
    """
    Symmetric Mean Absolute Percentage Error.

    SMAPE = mean(200 * |y_true - y_pred| / (|y_true| + |y_pred| + ε))

    Range: [0, 200]. Lower is better.
    """
    y_true, y_pred = _to_arrays(y_true, y_pred)
    eps = 1e-8
    return float(
        np.mean(200.0 * np.abs(y_true - y_pred) / (np.abs(y_true) + np.abs(y_pred) + eps))
    )


def mape(y_true: Any, y_pred: Any) -> float:
    """Mean Absolute Percentage Error. Lower is better."""
    y_true, y_pred = _to_arrays(y_true, y_pred)
    eps = 1e-8
    return float(np.mean(np.abs((y_true - y_pred) / (np.abs(y_true) + eps))) * 100)


def mae(y_true: Any, y_pred: Any) -> float:
    y_true, y_pred = _to_arrays(y_true, y_pred)
    return float(np.mean(np.abs(y_true - y_pred)))


def mse(y_true: Any, y_pred: Any) -> float:
    y_true, y_pred = _to_arrays(y_true, y_pred)
    return float(np.mean((y_true - y_pred) ** 2))


def rmse(y_true: Any, y_pred: Any) -> float:
    return float(np.sqrt(mse(y_true, y_pred)))


def rmsle(y_true: Any, y_pred: Any) -> float:
    """Root Mean Squared Log Error. Requires non-negative inputs."""
    y_true, y_pred = _to_arrays(y_true, y_pred)
    y_true = np.maximum(y_true, 0)
    y_pred = np.maximum(y_pred, 0)
    return float(np.sqrt(np.mean((np.log1p(y_true) - np.log1p(y_pred)) ** 2)))


def r2(y_true: Any, y_pred: Any) -> float:
    """Coefficient of determination. Higher is better."""
    y_true, y_pred = _to_arrays(y_true, y_pred)
    return float(sk_metrics.r2_score(y_true, y_pred))


# Classification ──────────────────────────────────────────────────
def accuracy(y_true: Any, y_pred: Any) -> float:
    return float(sk_metrics.accuracy_score(y_true, y_pred))


def f1_binary(y_true: Any, y_pred: Any) -> float:
    return float(sk_metrics.f1_score(y_true, y_pred, average="binary"))


def f1_macro(y_true: Any, y_pred: Any) -> float:
    return float(sk_metrics.f1_score(y_true, y_pred, average="macro"))


def f1_weighted(y_true: Any, y_pred: Any) -> float:
    return float(sk_metrics.f1_score(y_true, y_pred, average="weighted"))


def roc_auc(y_true: Any, y_pred: Any) -> float:
    """ROC-AUC. y_pred should be probability scores, not binary labels."""
    return float(sk_metrics.roc_auc_score(y_true, y_pred))


def pr_auc(y_true: Any, y_pred: Any) -> float:
    precision, recall, _ = sk_metrics.precision_recall_curve(y_true, y_pred)
    return float(sk_metrics.auc(recall, precision))


def log_loss_metric(y_true: Any, y_pred: Any) -> float:
    return float(sk_metrics.log_loss(y_true, y_pred))


def cohen_kappa(y_true: Any, y_pred: Any) -> float:
    return float(sk_metrics.cohen_kappa_score(y_true, y_pred))


# ─────────────────────────────────────────────────────────────────
# METRIC REGISTRY
# ─────────────────────────────────────────────────────────────────
_REGISTRY: Dict[str, MetricSpec] = {
    # Regression
    "smape":        MetricSpec("smape",        "minimize", smape,         "Symmetric MAPE [0,200]"),
    "mape":         MetricSpec("mape",          "minimize", mape,          "Mean Absolute Percentage Error"),
    "mae":          MetricSpec("mae",           "minimize", mae,           "Mean Absolute Error"),
    "mse":          MetricSpec("mse",           "minimize", mse,           "Mean Squared Error"),
    "rmse":         MetricSpec("rmse",          "minimize", rmse,          "Root Mean Squared Error"),
    "rmsle":        MetricSpec("rmsle",         "minimize", rmsle,         "Root Mean Squared Log Error"),
    "r2":           MetricSpec("r2",            "maximize", r2,            "R² (coefficient of determination)"),
    # Classification
    "accuracy":     MetricSpec("accuracy",      "maximize", accuracy,      "Classification accuracy"),
    "f1_binary":    MetricSpec("f1_binary",     "maximize", f1_binary,     "Binary F1"),
    "f1_macro":     MetricSpec("f1_macro",      "maximize", f1_macro,      "Macro-averaged F1"),
    "f1_weighted":  MetricSpec("f1_weighted",   "maximize", f1_weighted,   "Weighted F1"),
    "roc_auc":      MetricSpec("roc_auc",       "maximize", roc_auc,       "ROC-AUC"),
    "pr_auc":       MetricSpec("pr_auc",        "maximize", pr_auc,        "Precision-Recall AUC"),
    "log_loss":     MetricSpec("log_loss",      "minimize", log_loss_metric, "Log loss"),
    "kappa":        MetricSpec("kappa",         "maximize", cohen_kappa,   "Cohen's Kappa"),
}


def get_metric(name: str) -> MetricSpec:
    """
    Retrieve a MetricSpec by name.

    Parameters
    ----------
    name : str
        One of: smape, mape, mae, mse, rmse, rmsle, r2,
                accuracy, f1_binary, f1_macro, f1_weighted,
                roc_auc, pr_auc, log_loss, kappa

    Returns
    -------
    MetricSpec
    """
    name = name.lower().strip()
    if name not in _REGISTRY:
        raise ValueError(
            f"Unknown metric: '{name}'. "
            f"Available: {sorted(_REGISTRY.keys())}"
        )
    return _REGISTRY[name]


def list_metrics() -> list:
    """Return all registered metric names."""
    return sorted(_REGISTRY.keys())


# ─────────────────────────────────────────────────────────────────
# EVALUATE
# ─────────────────────────────────────────────────────────────────
def evaluate(
    y_true: Any,
    y_pred: Any,
    metric_name: str,
    extra_metrics: Optional[list] = None,
) -> MetricResult:
    """
    Compute a primary metric and optional secondary metrics.

    Parameters
    ----------
    y_true, y_pred : array-like
    metric_name : str
        Primary metric name from the registry.
    extra_metrics : list[str] | None
        Additional metric names to compute and include in result.extra.

    Returns
    -------
    MetricResult
    """
    spec = get_metric(metric_name)
    score = spec(y_true, y_pred)

    extra: Dict[str, float] = {}
    for em in (extra_metrics or []):
        try:
            extra[em] = get_metric(em)(y_true, y_pred)
        except Exception as exc:
            logger.warning("Could not compute extra metric '%s': %s", em, exc)

    result = MetricResult(
        metric_name=metric_name,
        score=score,
        direction=spec.direction,
        n_samples=len(np.asarray(y_true)),
        extra=extra,
    )
    logger.info("Metric: %s = %.6f (n=%d)", metric_name, score, result.n_samples)
    return result


# ─────────────────────────────────────────────────────────────────
# ERROR ANALYSIS UTILITIES
# ─────────────────────────────────────────────────────────────────
def error_analysis_regression(
    y_true: Any,
    y_pred: Any,
    df_context: Optional[pd.DataFrame] = None,
    group_cols: Optional[list] = None,
    n_worst: int = 20,
) -> pd.DataFrame:
    """
    Build a per-sample error table for regression problems.

    Returns a DataFrame with columns:
      y_true | y_pred | abs_error | rel_error | smape_contrib
    Optionally joined with context columns and grouped.

    Parameters
    ----------
    y_true, y_pred : array-like
    df_context : pd.DataFrame | None
        Original rows for joining context (e.g. category, brand).
    group_cols : list | None
        Columns to group by for aggregate error analysis.
    n_worst : int
        Number of worst predictions to highlight.

    Returns
    -------
    pd.DataFrame
    """
    y_true_arr, y_pred_arr = _to_arrays(y_true, y_pred)
    eps = 1e-8

    err_df = pd.DataFrame({
        "y_true": y_true_arr,
        "y_pred": y_pred_arr,
        "abs_error": np.abs(y_true_arr - y_pred_arr),
        "rel_error": np.abs(y_true_arr - y_pred_arr) / (np.abs(y_true_arr) + eps),
        "smape_contrib": 200.0 * np.abs(y_true_arr - y_pred_arr) / (np.abs(y_true_arr) + np.abs(y_pred_arr) + eps),
        "pred_ratio": y_pred_arr / (y_true_arr + eps),
    })

    if df_context is not None:
        for col in (group_cols or []):
            if col in df_context.columns:
                err_df[col] = df_context[col].values

    logger.info(
        "Error analysis: abs_err mean=%.4f | SMAPE mean=%.4f",
        err_df["abs_error"].mean(), err_df["smape_contrib"].mean(),
    )

    if group_cols:
        for col in group_cols:
            if col in err_df.columns:
                group_stats = err_df.groupby(col)[["abs_error", "smape_contrib"]].mean().round(4)
                logger.info("\nGroup error by '%s':\n%s", col, group_stats.to_string())

    return err_df
