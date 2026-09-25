"""
models/matchers.py — Pair scoring models.

Architecture:
    Level 0: Rule-based deterministic scorer
    Level 1: Feature-based ML scorer (LightGBM/CatBoost/XGBoost)

All models implement the same interface:
    fit(X, y)
    predict_proba(X) -> np.ndarray of match probabilities
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd


class PairScorer(ABC):
    """Abstract pair scoring model."""

    @property
    @abstractmethod
    def name(self) -> str:
        ...

    @abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray, **kwargs) -> None:
        ...

    @abstractmethod
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return P(match) for each pair."""
        ...

    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        return (self.predict_proba(X) >= threshold).astype(int)


class RuleBasedScorer(PairScorer):
    """
    Deterministic rule-based scorer.

    Uses a weighted combination of key similarity features:
        name_edit_sim * 0.45
        name_token_jaccard * 0.20
        addr_edit_sim * 0.15
        addr_token_jaccard * 0.10
        country_norm_exact * 0.10

    This is the baseline scorer. It requires no training.
    """

    WEIGHTS = {
        "name_edit_sim": 0.35,
        "name_jaro_winkler": 0.10,
        "name_token_jaccard": 0.15,
        "name_compact_exact": 0.05,
        "addr_edit_sim": 0.10,
        "addr_token_jaccard": 0.10,
        "addr_postal_match": 0.05,
        "country_norm_exact": 0.05,
        "evidence_strength": 0.05,
    }

    @property
    def name(self) -> str:
        return "rule_based"

    def fit(self, X: np.ndarray, y: np.ndarray, **kwargs) -> None:
        pass  # No training needed

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Cannot use raw numpy — needs feature names."""
        raise NotImplementedError(
            "RuleBasedScorer.predict_proba requires a DataFrame. "
            "Use predict_proba_df instead."
        )

    def predict_proba_df(self, df: pd.DataFrame) -> np.ndarray:
        """Score pairs from a DataFrame with named feature columns."""
        scores = np.zeros(len(df), dtype=np.float64)
        for feat, weight in self.WEIGHTS.items():
            if feat in df.columns:
                scores += df[feat].fillna(0).values.astype(np.float64) * weight
        return np.clip(scores, 0.0, 1.0)


class LightGBMScorer(PairScorer):
    """LightGBM-based pair scorer."""

    def __init__(self, params: Optional[Dict] = None):
        self.params = params or {
            "n_estimators": 1000,
            "learning_rate": 0.05,
            "num_leaves": 63,
            "min_child_samples": 20,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "reg_alpha": 0.1,
            "reg_lambda": 1.0,
            "verbose": -1,
            "n_jobs": -1,
            "random_state": 42,
            "is_unbalance": True,
        }
        self._model = None

    @property
    def name(self) -> str:
        return "lightgbm"

    def fit(self, X: np.ndarray, y: np.ndarray, **kwargs) -> None:
        import lightgbm as lgb

        eval_set = kwargs.get("eval_set", None)
        early_stopping = kwargs.get("early_stopping_rounds", 100)

        self._model = lgb.LGBMClassifier(**self.params)

        fit_params = {}
        if eval_set is not None:
            callbacks = [
                lgb.early_stopping(stopping_rounds=early_stopping, verbose=False),
                lgb.log_evaluation(period=0),
            ]
            fit_params["eval_set"] = [eval_set]
            fit_params["callbacks"] = callbacks

        self._model.fit(X, y, **fit_params)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if self._model is None:
            raise RuntimeError("Model not fitted")
        return self._model.predict_proba(X)[:, 1]

    def feature_importance(self, feature_names: Optional[List[str]] = None) -> pd.DataFrame:
        if self._model is None:
            raise RuntimeError("Model not fitted")
        importances = self._model.feature_importances_
        if feature_names is None:
            feature_names = [f"f_{i}" for i in range(len(importances))]
        return pd.DataFrame({
            "feature": feature_names,
            "importance": importances,
        }).sort_values("importance", ascending=False)

    def save(self, path: str) -> None:
        import joblib
        joblib.dump(self._model, path)

    def load(self, path: str) -> None:
        import joblib
        self._model = joblib.load(path)


class CatBoostScorer(PairScorer):
    """CatBoost-based pair scorer."""

    def __init__(self, params: Optional[Dict] = None):
        self.params = params or {
            "iterations": 1000,
            "learning_rate": 0.05,
            "depth": 6,
            "l2_leaf_reg": 3.0,
            "verbose": 0,
            "random_state": 42,
            "auto_class_weights": "Balanced",
        }
        self._model = None

    @property
    def name(self) -> str:
        return "catboost"

    def fit(self, X: np.ndarray, y: np.ndarray, **kwargs) -> None:
        from catboost import CatBoostClassifier

        eval_set = kwargs.get("eval_set", None)
        early_stopping = kwargs.get("early_stopping_rounds", 100)

        params = {**self.params}
        if early_stopping:
            params["early_stopping_rounds"] = early_stopping

        self._model = CatBoostClassifier(**params)

        fit_params = {}
        if eval_set is not None:
            fit_params["eval_set"] = eval_set

        self._model.fit(X, y, **fit_params)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if self._model is None:
            raise RuntimeError("Model not fitted")
        return self._model.predict_proba(X)[:, 1]

    def save(self, path: str) -> None:
        if self._model is None:
            raise RuntimeError("Model not fitted")
        self._model.save_model(path)

    def load(self, path: str) -> None:
        from catboost import CatBoostClassifier
        self._model = CatBoostClassifier()
        self._model.load_model(path)


class XGBoostScorer(PairScorer):
    """XGBoost-based pair scorer."""

    def __init__(self, params: Optional[Dict] = None):
        self.params = params or {
            "n_estimators": 1000,
            "learning_rate": 0.05,
            "max_depth": 6,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "reg_alpha": 0.1,
            "reg_lambda": 1.0,
            "verbosity": 0,
            "random_state": 42,
            "scale_pos_weight": 1,
            "eval_metric": "logloss",
            "use_label_encoder": False,
        }
        self._model = None

    @property
    def name(self) -> str:
        return "xgboost"

    def fit(self, X: np.ndarray, y: np.ndarray, **kwargs) -> None:
        import xgboost as xgb

        eval_set = kwargs.get("eval_set", None)
        early_stopping = kwargs.get("early_stopping_rounds", 100)

        self._model = xgb.XGBClassifier(**self.params)

        fit_params = {}
        if eval_set is not None:
            fit_params["eval_set"] = [eval_set]
            fit_params["early_stopping_rounds"] = early_stopping
            fit_params["verbose"] = False

        self._model.fit(X, y, **fit_params)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if self._model is None:
            raise RuntimeError("Model not fitted")
        return self._model.predict_proba(X)[:, 1]

    def save(self, path: str) -> None:
        import joblib
        joblib.dump(self._model, path)

    def load(self, path: str) -> None:
        import joblib
        self._model = joblib.load(path)


# ──────────────────────────────────────────────────────────────
# MODEL REGISTRY
# ──────────────────────────────────────────────────────────────

MODEL_REGISTRY = {
    "rule_based": RuleBasedScorer,
    "lightgbm": LightGBMScorer,
    "catboost": CatBoostScorer,
    "xgboost": XGBoostScorer,
}


def get_scorer(name: str, params: Optional[Dict] = None) -> PairScorer:
    """Get a scorer by name."""
    if name not in MODEL_REGISTRY:
        raise ValueError(
            f"Unknown scorer: '{name}'. Available: {list(MODEL_REGISTRY.keys())}"
        )
    cls = MODEL_REGISTRY[name]
    if params and name != "rule_based":
        return cls(params=params)
    return cls()
