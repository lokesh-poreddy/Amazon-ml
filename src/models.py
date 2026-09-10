"""
src/models.py
Amazon ML Challenge 2026 — Model abstraction layer.

Provides a unified interface:
    model.fit(X_train, y_train, X_val, y_val)
    model.predict(X)
    model.feature_importance()
    model.save() / model.load()

Supported backends:
  - LightGBMModel   (primary recommendation for tabular)
  - XGBoostModel
  - CatBoostModel
  - LinearModel     (Ridge / Logistic)
  - DummyModel      (mean / most-frequent baseline)

Adding a new model: subclass BaseModel and implement the four core methods.

Model wrappers do NOT implement CV logic — that lives in train.py.
"""

from __future__ import annotations

import pickle
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

from src.logging_utils import get_logger

logger = get_logger(__name__)


# ─────────────────────────────────────────────────────────────────
# BASE MODEL
# ─────────────────────────────────────────────────────────────────
class BaseModel(ABC):
    """
    Abstract base class for all competition models.

    Subclasses must implement:
      - _fit()
      - _predict()
      - feature_importance() [optional]
    """

    def __init__(self, params: Optional[Dict[str, Any]] = None) -> None:
        self.params = params or {}
        self._model: Any = None
        self._feature_names: List[str] = []
        self.name: str = "base"

    # ── Public interface ──────────────────────────────────────────
    def fit(
        self,
        X_train: Union[pd.DataFrame, np.ndarray],
        y_train: Union[pd.Series, np.ndarray],
        X_val: Optional[Union[pd.DataFrame, np.ndarray]] = None,
        y_val: Optional[Union[pd.Series, np.ndarray]] = None,
        sample_weight: Optional[np.ndarray] = None,
    ) -> "BaseModel":
        """
        Fit the model on training data.

        Parameters
        ----------
        X_train, y_train : training features and target
        X_val, y_val : optional validation set for early stopping
        sample_weight : optional per-sample weights
        """
        if isinstance(X_train, pd.DataFrame):
            self._feature_names = list(X_train.columns)
        logger.info("Fitting %s on %d rows, %d features.", self.name, len(X_train), X_train.shape[1] if hasattr(X_train, "shape") else "?")
        return self._fit(X_train, y_train, X_val, y_val, sample_weight)

    def predict(
        self,
        X: Union[pd.DataFrame, np.ndarray],
    ) -> np.ndarray:
        """Return predictions (continuous for regression, class label for classification)."""
        if self._model is None:
            raise RuntimeError(f"{self.name} has not been fitted yet. Call .fit() first.")
        return self._predict(X)

    def predict_proba(
        self,
        X: Union[pd.DataFrame, np.ndarray],
    ) -> Optional[np.ndarray]:
        """Return class probabilities. None if not applicable."""
        return None

    def feature_importance(self) -> Optional[pd.DataFrame]:
        """Return a sorted DataFrame of feature names and importances."""
        return None

    def save(self, path: Union[str, Path]) -> None:
        """Serialize model to disk."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self, f)
        logger.info("Saved %s → %s", self.name, path)

    @classmethod
    def load(cls, path: Union[str, Path]) -> "BaseModel":
        """Deserialize model from disk."""
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Model file not found: {path}")
        with open(path, "rb") as f:
            obj = pickle.load(f)
        logger.info("Loaded model from %s", path)
        return obj

    # ── Abstract ──────────────────────────────────────────────────
    @abstractmethod
    def _fit(self, X_train, y_train, X_val, y_val, sample_weight) -> "BaseModel":
        ...

    @abstractmethod
    def _predict(self, X) -> np.ndarray:
        ...


# ─────────────────────────────────────────────────────────────────
# DUMMY BASELINE
# ─────────────────────────────────────────────────────────────────
class DummyModel(BaseModel):
    """
    Predict the training mean (regression) or majority class
    (classification). Establishes the absolute floor for performance.
    """

    def __init__(self, strategy: str = "mean") -> None:
        super().__init__()
        self.name = "dummy"
        self.strategy = strategy
        self._value: float = 0.0

    def _fit(self, X_train, y_train, X_val, y_val, sample_weight) -> "DummyModel":
        y = np.asarray(y_train)
        if self.strategy == "mean":
            self._value = float(np.mean(y))
        elif self.strategy == "median":
            self._value = float(np.median(y))
        elif self.strategy == "mode":
            vals, counts = np.unique(y, return_counts=True)
            self._value = float(vals[np.argmax(counts)])
        logger.info("DummyModel (%s): value = %.4f", self.strategy, self._value)
        return self

    def _predict(self, X) -> np.ndarray:
        n = len(X) if hasattr(X, "__len__") else X.shape[0]
        return np.full(n, self._value, dtype=np.float64)


# ─────────────────────────────────────────────────────────────────
# LIGHTGBM
# ─────────────────────────────────────────────────────────────────
class LightGBMModel(BaseModel):
    """
    LightGBM wrapper with early stopping support.

    Default params are tuned for an unknown tabular regression task.
    Override via params dict or configs/config.yaml.
    """

    _DEFAULT_PARAMS = {
        "n_estimators": 1000,
        "learning_rate": 0.05,
        "num_leaves": 127,
        "max_depth": -1,
        "min_child_samples": 20,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "reg_alpha": 0.1,
        "reg_lambda": 1.0,
        "verbose": -1,
        "n_jobs": -1,
    }

    def __init__(
        self,
        params: Optional[Dict[str, Any]] = None,
        task_type: str = "regression",
        early_stopping_rounds: int = 100,
    ) -> None:
        merged = {**self._DEFAULT_PARAMS, **(params or {})}
        super().__init__(merged)
        self.name = "lgbm"
        self.task_type = task_type
        self.early_stopping_rounds = early_stopping_rounds

    def _fit(self, X_train, y_train, X_val, y_val, sample_weight) -> "LightGBMModel":
        try:
            import lightgbm as lgb
        except ImportError:
            raise ImportError(
                "LightGBM is not installed. Run: pip install lightgbm"
            )

        objective = "regression" if self.task_type == "regression" else "binary"
        params = {**self.params, "objective": objective}

        callbacks = [lgb.log_evaluation(period=100), lgb.early_stopping(self.early_stopping_rounds)]

        eval_set = [(X_val, y_val)] if X_val is not None else None

        self._model = lgb.LGBMRegressor(**params) if self.task_type == "regression" \
            else lgb.LGBMClassifier(**params)

        fit_params: Dict = {}
        if eval_set is not None:
            fit_params["eval_set"] = eval_set
            fit_params["callbacks"] = callbacks
        if sample_weight is not None:
            fit_params["sample_weight"] = sample_weight

        self._model.fit(X_train, y_train, **fit_params)
        logger.info("LGBM best iteration: %s", getattr(self._model, "best_iteration_", "n/a"))
        return self

    def _predict(self, X) -> np.ndarray:
        return self._model.predict(X)

    def predict_proba(self, X) -> Optional[np.ndarray]:
        if self.task_type != "regression" and hasattr(self._model, "predict_proba"):
            return self._model.predict_proba(X)
        return None

    def feature_importance(self) -> Optional[pd.DataFrame]:
        if self._model is None:
            return None
        fi = pd.DataFrame({
            "feature": self._feature_names or list(range(self._model.n_features_in_)),
            "importance": self._model.feature_importances_,
        }).sort_values("importance", ascending=False).reset_index(drop=True)
        return fi


# ─────────────────────────────────────────────────────────────────
# XGBOOST
# ─────────────────────────────────────────────────────────────────
class XGBoostModel(BaseModel):
    """XGBoost wrapper."""

    _DEFAULT_PARAMS = {
        "n_estimators": 1000,
        "learning_rate": 0.05,
        "max_depth": 6,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "reg_alpha": 0.1,
        "reg_lambda": 1.0,
        "verbosity": 0,
    }

    def __init__(
        self,
        params: Optional[Dict[str, Any]] = None,
        task_type: str = "regression",
        early_stopping_rounds: int = 100,
    ) -> None:
        merged = {**self._DEFAULT_PARAMS, **(params or {})}
        super().__init__(merged)
        self.name = "xgb"
        self.task_type = task_type
        self.early_stopping_rounds = early_stopping_rounds

    def _fit(self, X_train, y_train, X_val, y_val, sample_weight) -> "XGBoostModel":
        try:
            import xgboost as xgb
        except ImportError:
            raise ImportError("XGBoost is not installed. Run: pip install xgboost")

        params = {**self.params}
        if self.task_type == "binary":
            params["objective"] = "binary:logistic"
            params["eval_metric"] = "logloss"
        elif self.task_type == "multiclass":
            params["objective"] = "multi:softmax"
            params["eval_metric"] = "mlogloss"
        else:
            params["objective"] = "reg:squarederror"
            params["eval_metric"] = "rmse"

        Cls = xgb.XGBRegressor if self.task_type == "regression" else xgb.XGBClassifier
        self._model = Cls(**params, early_stopping_rounds=self.early_stopping_rounds)

        fit_params: Dict = {}
        if X_val is not None:
            fit_params["eval_set"] = [(X_val, y_val)]
        if sample_weight is not None:
            fit_params["sample_weight"] = sample_weight

        self._model.fit(X_train, y_train, verbose=False, **fit_params)
        return self

    def _predict(self, X) -> np.ndarray:
        return self._model.predict(X)

    def predict_proba(self, X) -> Optional[np.ndarray]:
        if self.task_type != "regression" and hasattr(self._model, "predict_proba"):
            return self._model.predict_proba(X)
        return None

    def feature_importance(self) -> Optional[pd.DataFrame]:
        if self._model is None:
            return None
        return pd.DataFrame({
            "feature": self._feature_names or list(range(self._model.n_features_in_)),
            "importance": self._model.feature_importances_,
        }).sort_values("importance", ascending=False).reset_index(drop=True)


# ─────────────────────────────────────────────────────────────────
# CATBOOST
# ─────────────────────────────────────────────────────────────────
class CatBoostModel(BaseModel):
    """CatBoost wrapper with native categorical support."""

    _DEFAULT_PARAMS = {
        "iterations": 1000,
        "learning_rate": 0.05,
        "depth": 6,
        "l2_leaf_reg": 3.0,
        "verbose": 0,
    }

    def __init__(
        self,
        params: Optional[Dict[str, Any]] = None,
        task_type: str = "regression",
        cat_features: Optional[List[str]] = None,
        early_stopping_rounds: int = 100,
    ) -> None:
        merged = {**self._DEFAULT_PARAMS, **(params or {})}
        super().__init__(merged)
        self.name = "catboost"
        self.task_type = task_type
        self.cat_features = cat_features or []
        self.early_stopping_rounds = early_stopping_rounds

    def _fit(self, X_train, y_train, X_val, y_val, sample_weight) -> "CatBoostModel":
        try:
            from catboost import CatBoostClassifier, CatBoostRegressor, Pool
        except ImportError:
            raise ImportError("CatBoost is not installed. Run: pip install catboost")

        cat_idx = None
        if self.cat_features and isinstance(X_train, pd.DataFrame):
            cat_idx = [i for i, c in enumerate(X_train.columns) if c in self.cat_features]

        train_pool = Pool(X_train, y_train, cat_features=cat_idx, weight=sample_weight)
        eval_pool  = Pool(X_val, y_val, cat_features=cat_idx) if X_val is not None else None

        Cls = CatBoostRegressor if self.task_type == "regression" else CatBoostClassifier
        self._model = Cls(**self.params, early_stopping_rounds=self.early_stopping_rounds)
        self._model.fit(train_pool, eval_set=eval_pool, use_best_model=eval_pool is not None)
        return self

    def _predict(self, X) -> np.ndarray:
        return self._model.predict(X)

    def predict_proba(self, X) -> Optional[np.ndarray]:
        if self.task_type != "regression" and hasattr(self._model, "predict_proba"):
            return self._model.predict_proba(X)
        return None

    def feature_importance(self) -> Optional[pd.DataFrame]:
        if self._model is None:
            return None
        return pd.DataFrame({
            "feature": self._feature_names,
            "importance": self._model.get_feature_importance(),
        }).sort_values("importance", ascending=False).reset_index(drop=True)


# ─────────────────────────────────────────────────────────────────
# SKLEARN LINEAR MODEL
# ─────────────────────────────────────────────────────────────────
class LinearModel(BaseModel):
    """Ridge Regression or Logistic Regression via sklearn."""

    def __init__(
        self,
        params: Optional[Dict[str, Any]] = None,
        task_type: str = "regression",
    ) -> None:
        super().__init__(params or {"alpha": 1.0})
        self.name = "linear"
        self.task_type = task_type

    def _fit(self, X_train, y_train, X_val, y_val, sample_weight) -> "LinearModel":
        from sklearn.linear_model import LogisticRegression, Ridge

        if self.task_type == "regression":
            self._model = Ridge(**self.params)
        else:
            self._model = LogisticRegression(**self.params, max_iter=1000)
        self._model.fit(X_train, y_train, sample_weight=sample_weight)
        return self

    def _predict(self, X) -> np.ndarray:
        return self._model.predict(X)

    def predict_proba(self, X) -> Optional[np.ndarray]:
        if hasattr(self._model, "predict_proba"):
            return self._model.predict_proba(X)
        return None


# ─────────────────────────────────────────────────────────────────
# MODEL FACTORY
# ─────────────────────────────────────────────────────────────────
_MODEL_REGISTRY: Dict[str, type] = {
    "lgbm":     LightGBMModel,
    "lightgbm": LightGBMModel,
    "xgb":      XGBoostModel,
    "xgboost":  XGBoostModel,
    "catboost": CatBoostModel,
    "linear":   LinearModel,
    "dummy":    DummyModel,
}


def build_model(
    name: str,
    params: Optional[Dict[str, Any]] = None,
    task_type: str = "regression",
    **kwargs: Any,
) -> BaseModel:
    """
    Instantiate a model by name.

    Parameters
    ----------
    name : str
        One of: lgbm | xgb | catboost | linear | dummy
    params : dict | None
        Hyperparameter overrides.
    task_type : str
        "regression" | "binary" | "multiclass"
    **kwargs
        Extra arguments forwarded to the model constructor.

    Returns
    -------
    BaseModel
    """
    name = name.lower().strip()
    if name not in _MODEL_REGISTRY:
        raise ValueError(
            f"Unknown model: '{name}'. "
            f"Available: {sorted(_MODEL_REGISTRY.keys())}"
        )
    cls = _MODEL_REGISTRY[name]
    if name in ("lgbm", "lightgbm", "xgb", "xgboost", "catboost", "linear"):
        return cls(params=params, task_type=task_type, **kwargs)
    return cls(**kwargs)
