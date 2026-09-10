"""
src/ensemble.py
Amazon ML Challenge 2026 — OOF-based ensemble engine.

A model should be added to an ensemble only when its errors are
complementary to the existing ensemble — not just because it exists.

Supported methods:
  - weighted_avg:  weights learned from OOF metric
  - rank_avg:      rank-normalize then average (robust for different scales)
  - simple_avg:    equal weights
  - stacking:      train a meta-learner on OOF predictions
  - blending:      fixed linear combination set manually

Core workflow:
  1. Generate OOF predictions for each candidate model (see train.py)
  2. Compute OOF metric for each model
  3. Inspect pairwise OOF correlations
  4. Drop models that are too correlated (add no diversity)
  5. Optimise or assign weights
  6. Apply to test predictions

CRITICAL PRINCIPLE:
  OOF predictions must come from the same CV split for all models.
  Mixing different splits invalidates the ensemble analysis.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import rankdata

from src.logging_utils import get_logger
from src.metrics import MetricSpec

logger = get_logger(__name__)


# ─────────────────────────────────────────────────────────────────
# OOF ANALYSIS
# ─────────────────────────────────────────────────────────────────
@dataclass
class OOFCandidate:
    """One model's OOF predictions and associated metadata."""
    name: str
    oof_predictions: np.ndarray
    oof_score: float
    test_predictions: Optional[np.ndarray] = None
    feature_version: str = "v0"
    notes: str = ""


def oof_correlation_matrix(candidates: List[OOFCandidate]) -> pd.DataFrame:
    """
    Compute pairwise Pearson correlation between OOF predictions.

    Low correlation → ensemble will likely help.
    Correlation > 0.98 → models are nearly identical, probably no benefit.

    Returns
    -------
    pd.DataFrame
        Symmetric correlation matrix.
    """
    names = [c.name for c in candidates]
    preds = np.stack([c.oof_predictions for c in candidates], axis=1)
    corr = np.corrcoef(preds.T)
    return pd.DataFrame(corr, index=names, columns=names)


def score_weighted_selection(
    candidates: List[OOFCandidate],
    metric_spec: MetricSpec,
    y_true: np.ndarray,
    correlation_threshold: float = 0.98,
) -> List[OOFCandidate]:
    """
    Greedily select candidates that:
      1. Improve the actual simple blend score over the already-selected candidates
      2. Fall below a pairwise correlation threshold with already-selected candidates

    Parameters
    ----------
    candidates : list[OOFCandidate]
    metric_spec : MetricSpec
    y_true : np.ndarray
        True labels corresponding to OOF indices.
    correlation_threshold : float
        Reject a candidate if it correlates > this with any selected candidate.

    Returns
    -------
    list[OOFCandidate]
        Filtered diverse subset.
    """
    # Sort by quality (best first)
    scored = sorted(
        candidates,
        key=lambda c: c.oof_score,
        reverse=(metric_spec.direction == "maximize"),
    )

    selected: List[OOFCandidate] = []
    current_blend_score = None

    for candidate in scored:
        if not selected:
            selected.append(candidate)
            current_blend_score = candidate.oof_score
            logger.info("Ensemble: SELECT '%s' (score=%.6f) [first]", candidate.name, candidate.oof_score)
            continue
            
        max_corr = max(
            abs(np.corrcoef(candidate.oof_predictions, s.oof_predictions)[0, 1])
            for s in selected
        )
        if max_corr > correlation_threshold:
            logger.info(
                "Ensemble: REJECT '%s' (score=%.6f, max_corr=%.4f > threshold=%.4f)",
                candidate.name, candidate.oof_score, max_corr, correlation_threshold,
            )
            continue

        # Evaluate marginal blend improvement
        new_blend_preds = np.mean([s.oof_predictions for s in selected] + [candidate.oof_predictions], axis=0)
        new_blend_score = metric_spec(y_true, new_blend_preds)
        
        improved = (
            (metric_spec.direction == "minimize" and new_blend_score < current_blend_score) or
            (metric_spec.direction == "maximize" and new_blend_score > current_blend_score)
        )
        
        if improved:
            logger.info(
                "Ensemble: SELECT '%s' (score=%.6f, max_corr=%.4f, blend %.6f -> %.6f)",
                candidate.name, candidate.oof_score, max_corr, current_blend_score, new_blend_score
            )
            selected.append(candidate)
            current_blend_score = new_blend_score
        else:
            logger.info(
                "Ensemble: REJECT '%s' (score=%.6f, max_corr=%.4f, NO BLEND IMPROVEMENT %.6f -> %.6f)",
                candidate.name, candidate.oof_score, max_corr, current_blend_score, new_blend_score
            )

    return selected


# ─────────────────────────────────────────────────────────────────
# ENSEMBLE METHODS
# ─────────────────────────────────────────────────────────────────
def simple_avg(test_preds_list: List[np.ndarray]) -> np.ndarray:
    """Equal-weight average."""
    arr = np.stack(test_preds_list, axis=0)
    return arr.mean(axis=0)


def weighted_avg(test_preds_list: List[np.ndarray], weights: List[float]) -> np.ndarray:
    """Weighted average. Weights are normalised internally."""
    w = np.array(weights, dtype=np.float64)
    w = w / w.sum()
    arr = np.stack(test_preds_list, axis=0)   # (n_models, n_test)
    return (arr * w[:, None]).sum(axis=0)


def rank_avg(test_preds_list: List[np.ndarray], y_ref: Optional[np.ndarray] = None) -> np.ndarray:
    """
    Rank-average (Boruta-style):
      1. Rank predictions within each model
      2. Average ranks
      3. Optionally re-scale back to original range

    Robust to scale differences between models.
    
    WARNING: Unsuitable for raw regression if y_ref is None, as it returns [0, 1] fractions.
    Provide y_ref (e.g., training targets) to rescale to [min(y_ref), max(y_ref)].
    """
    ranked = [rankdata(p) / len(p) for p in test_preds_list]
    avg_rank = np.stack(ranked, axis=0).mean(axis=0)
    
    if y_ref is not None:
        y_min, y_max = np.min(y_ref), np.max(y_ref)
        avg_rank = y_min + avg_rank * (y_max - y_min)
        
    return avg_rank


def optimise_weights(
    candidates: List[OOFCandidate],
    y_true: np.ndarray,
    metric_spec: MetricSpec,
    n_trials: int = 500,
    seed: int = 42,
) -> Tuple[List[float], float]:
    """
    Find the blend weights that minimise / maximise the OOF metric.

    Uses scipy's Nelder-Mead optimizer with random restarts.

    Parameters
    ----------
    candidates : list[OOFCandidate]
    y_true : np.ndarray
    metric_spec : MetricSpec
    n_trials : int
        Number of random restarts for the optimizer.
    seed : int

    Returns
    -------
    (weights, best_score) where weights sum to 1.
    """
    n = len(candidates)
    rng = np.random.RandomState(seed)
    oof_matrix = np.stack([c.oof_predictions for c in candidates], axis=0)  # (n_models, n_rows)

    def objective(w: np.ndarray) -> float:
        w_pos = np.exp(w)
        w_norm = w_pos / w_pos.sum()
        blended = (oof_matrix * w_norm[:, None]).sum(axis=0)
        score = metric_spec(y_true, blended)
        return score if metric_spec.direction == "minimize" else -score

    best_result = None
    best_score = float("inf")

    for _ in range(n_trials):
        x0 = rng.dirichlet(np.ones(n))
        x0 = np.log(x0 + 1e-9)
        result = minimize(objective, x0, method="Nelder-Mead",
                          options={"maxiter": 1000, "xatol": 1e-6, "fatol": 1e-6})
        if result.fun < best_score:
            best_score = result.fun
            best_result = result

    w_raw = np.exp(best_result.x)
    weights = (w_raw / w_raw.sum()).tolist()
    final_score = metric_spec.direction == "minimize" and best_score or -best_score

    logger.info(
        "Optimised weights: %s  →  OOF %s = %.6f",
        [f"{w:.4f}" for w in weights], metric_spec.name, final_score,
    )
    return weights, final_score


# ─────────────────────────────────────────────────────────────────
# STACKING META-LEARNER
# ─────────────────────────────────────────────────────────────────
def stack_models(
    candidates: List[OOFCandidate],
    y_true: np.ndarray,
    task_type: str = "regression",
    meta_model_name: str = "linear",
) -> "BaseModel":
    """
    Train a meta-learner on OOF predictions (stacking).

    Parameters
    ----------
    candidates : list[OOFCandidate]
    y_true : np.ndarray
    task_type : str
    meta_model_name : str
        Model for the meta-layer. Keep it simple ("linear" recommended).

    Returns
    -------
    BaseModel
        Fitted meta-learner.
    """
    from src.models import build_model

    meta_X = pd.DataFrame(
        {c.name: c.oof_predictions for c in candidates}
    )
    meta_y = pd.Series(y_true, name="target")

    meta_model = build_model(meta_model_name, task_type=task_type)
    meta_model.fit(meta_X, meta_y)

    logger.info("Stacking meta-learner (%s) trained on %d OOF rows, %d base models.",
                meta_model_name, len(meta_X), len(candidates))
    return meta_model


def predict_stack(
    meta_model: "BaseModel",
    candidates: List[OOFCandidate],
) -> np.ndarray:
    """Apply the stack meta-learner to test predictions."""
    if any(c.test_predictions is None for c in candidates):
        raise ValueError("All candidates must have test_predictions set.")
    meta_X = pd.DataFrame({c.name: c.test_predictions for c in candidates})
    return meta_model.predict(meta_X)
