"""
src/preprocessing.py
Amazon ML Challenge 2026 — Fold-aware preprocessing pipeline.

Design principle: ALL transformations (imputers, encoders, scalers,
target-encoding statistics) must be FIT on training data only within
each CV fold. Never fit on the full dataset before splitting.

Key classes:
  - FoldAwarePreprocessor  : sklearn Pipeline wrapped to enforce this rule
  - TargetEncoder          : fold-safe mean-target encoding with smoothing
  - build_preprocessor()   : factory that assembles column-specific transformers

Leakage checklist:
  ✓ Imputers fit on train fold only
  ✓ Scalers fit on train fold only
  ✓ Categorical encoders fit on train fold only
  ✓ TargetEncoder uses out-of-fold statistics
  ✗ NEVER call .fit() on validation or test data
"""

from __future__ import annotations

import warnings
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    LabelEncoder,
    MinMaxScaler,
    OrdinalEncoder,
    StandardScaler,
)

from src.logging_utils import get_logger

logger = get_logger(__name__)


# ─────────────────────────────────────────────────────────────────
# SAFE TARGET ENCODER (fold-aware, no leakage)
# ─────────────────────────────────────────────────────────────────
class TargetEncoder(BaseEstimator, TransformerMixin):
    """
    Mean-target encoding with additive smoothing.

    LEAKAGE-SAFE DESIGN:
      - fit()          → compute group statistics from training data (for val/test)
      - transform()    → apply fit statistics (safe for val and test)
      - fit_transform() → Leave-One-Out (LOO) encoding for train rows.
                          Each row's own target is EXCLUDED from its group
                          statistic, so there is no self-information leakage.

    The LOO design means that even when called on training data, a row's
    own target value does not contribute to the encoding of that row.

    Parameters
    ----------
    smoothing : float
        Higher → stronger regularisation toward global mean.
    min_samples_leaf : int
        Minimum group size to trust the group mean.
    handle_unknown : str
        "global_mean" → unknown categories get the global mean.
    """

    def __init__(
        self,
        smoothing: float = 10.0,
        min_samples_leaf: int = 1,
        handle_unknown: str = "global_mean",
    ) -> None:
        self.smoothing = smoothing
        self.min_samples_leaf = min_samples_leaf
        self.handle_unknown = handle_unknown
        self._global_mean: float = 0.0
        self._mapping: Dict[str, Dict] = {}
        self._count_mapping: Dict[str, Dict] = {}
        self._sum_mapping: Dict[str, Dict] = {}

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "TargetEncoder":
        """
        Fit on training data only.

        Computes group means used for encoding validation and test rows.
        NOT used for training row encoding (see fit_transform).

        Parameters
        ----------
        X : pd.DataFrame
            Categorical columns to encode.
        y : pd.Series
            Continuous target (for regression) or binary label.
        """
        self._global_mean = float(y.mean())
        self._mapping = {}
        self._count_mapping = {}
        self._sum_mapping = {}

        for col in X.columns:
            grp = (
                pd.concat([X[[col]], y.rename("__target__")], axis=1)
                .groupby(col)["__target__"]
                .agg(["count", "mean", "sum"])
            )
            smoother = 1 / (
                1 + np.exp(-(grp["count"] - self.min_samples_leaf) / self.smoothing)
            )
            grp["encoded"] = smoother * grp["mean"] + (1 - smoother) * self._global_mean
            self._mapping[col]       = grp["encoded"].to_dict()
            self._count_mapping[col] = grp["count"].to_dict()
            self._sum_mapping[col]   = grp["sum"].to_dict()

        return self

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        """
        Apply encoding using fit statistics.

        Use for VALIDATION and TEST rows only.
        For training rows, use fit_transform() which applies LOO.
        """
        result = np.zeros((len(X), len(X.columns)), dtype=np.float32)
        for i, col in enumerate(X.columns):
            result[:, i] = (
                X[col]
                .map(self._mapping.get(col, {}))
                .fillna(self._global_mean)
                .values
            )
        return result

    def fit_transform(self, X: pd.DataFrame, y: pd.Series = None) -> np.ndarray:  # type: ignore
        """
        Leave-One-Out (LOO) target encoding for TRAINING rows.

        For each row i in group g:
          - group sum   = sum(y for all rows in g)
          - group count = count(rows in g)
          - LOO mean    = (group_sum - y_i) / (group_count - 1)
          - Smoothed    = blend(LOO_mean, global_mean)

        This eliminates the self-leakage that occurs when a row's own
        target contributes to its group encoding statistic.
        """
        self.fit(X, y)
        y_arr = np.asarray(y, dtype=np.float64)
        result = np.zeros((len(X), len(X.columns)), dtype=np.float32)

        for i, col in enumerate(X.columns):
            col_vals = X[col].values
            encoded  = np.full(len(X), self._global_mean, dtype=np.float64)

            for j, cat in enumerate(col_vals):
                total_sum   = self._sum_mapping[col].get(cat, 0.0)
                total_count = self._count_mapping[col].get(cat, 0)

                if total_count > 1:
                    loo_sum   = total_sum - y_arr[j]
                    loo_count = total_count - 1
                    loo_mean  = loo_sum / loo_count
                else:
                    # Singleton: fall back to global mean (safest option)
                    loo_mean = self._global_mean

                smoother = 1 / (
                    1 + np.exp(-(total_count - self.min_samples_leaf) / self.smoothing)
                )
                encoded[j] = smoother * loo_mean + (1 - smoother) * self._global_mean

            result[:, i] = encoded.astype(np.float32)

        return result



# ─────────────────────────────────────────────────────────────────
# FREQUENCY ENCODER
# ─────────────────────────────────────────────────────────────────
class FrequencyEncoder(BaseEstimator, TransformerMixin):
    """
    Replace each category with its frequency in the training set.

    Fit only on training data. Unknown categories in test get 0.
    """

    def __init__(self) -> None:
        self._freq_maps: Dict[str, Dict] = {}

    def fit(self, X: pd.DataFrame, y=None) -> "FrequencyEncoder":
        self._freq_maps = {}
        for col in X.columns:
            freq = X[col].value_counts(normalize=True).to_dict()
            self._freq_maps[col] = freq
        return self

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        result = np.zeros((len(X), len(X.columns)), dtype=np.float32)
        for i, col in enumerate(X.columns):
            result[:, i] = X[col].map(self._freq_maps.get(col, {})).fillna(0.0).values
        return result


# ─────────────────────────────────────────────────────────────────
# SKLEARN-COMPATIBLE PREPROCESSOR BUILDER
# ─────────────────────────────────────────────────────────────────
def build_preprocessor(
    numeric_cols: List[str],
    categorical_cols: List[str],
    text_cols: Optional[List[str]] = None,
    numeric_imputer: str = "median",
    categorical_imputer: str = "constant",
    categorical_fill_value: str = "MISSING",
    numeric_fill_value: float = -1.0,
    scale_numeric: bool = False,
    categorical_strategy: str = "ordinal",  # "ordinal" | "frequency" | "onehot"
) -> ColumnTransformer:
    """
    Build a fold-aware ColumnTransformer for standard preprocessing.

    This is designed for use inside a CV loop:
        - Call .fit() only on train folds
        - Call .transform() on train fold AND validation/test

    Parameters
    ----------
    numeric_cols : list[str]
        Column names to treat as numeric.
    categorical_cols : list[str]
        Column names to treat as categorical.
    text_cols : list[str] | None
        Text columns — usually passed to TF-IDF separately, so
        they are dropped here by default.
    numeric_imputer : str
        "median" | "mean" | "constant" | "none"
    categorical_imputer : str
        "constant" | "most_frequent" | "none"
    categorical_fill_value : str
        Fill value for categorical NaN.
    numeric_fill_value : float
        Fill value for numeric NaN (only used when imputer="constant").
    scale_numeric : bool
        If True, apply StandardScaler after imputation.
        Not generally recommended for tree models.
    categorical_strategy : str
        "ordinal" → OrdinalEncoder (recommended for tree models)
        "frequency" → FrequencyEncoder
        "onehot" → one-hot (only for low-cardinality, linear models)

    Returns
    -------
    sklearn ColumnTransformer
    """
    text_cols = text_cols or []

    # ── Numeric pipeline ─────────────────────────────────────────
    num_steps: List = []
    if numeric_imputer == "median":
        num_steps.append(("imputer", SimpleImputer(strategy="median")))
    elif numeric_imputer == "mean":
        num_steps.append(("imputer", SimpleImputer(strategy="mean")))
    elif numeric_imputer == "constant":
        num_steps.append(("imputer", SimpleImputer(strategy="constant", fill_value=numeric_fill_value)))
    # else "none": no imputation step

    if scale_numeric:
        num_steps.append(("scaler", StandardScaler()))

    if not num_steps:
        # fallback identity
        num_steps.append(("passthrough", "passthrough"))

    numeric_pipe = Pipeline(steps=num_steps) if len(num_steps) > 1 else num_steps[0][1]

    # ── Categorical pipeline ──────────────────────────────────────
    cat_steps: List = []
    if categorical_imputer == "constant":
        cat_steps.append(("imputer", SimpleImputer(strategy="constant", fill_value=categorical_fill_value)))
    elif categorical_imputer == "most_frequent":
        cat_steps.append(("imputer", SimpleImputer(strategy="most_frequent")))

    if categorical_strategy == "ordinal":
        cat_steps.append((
            "encoder",
            OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1),
        ))
    elif categorical_strategy == "frequency":
        cat_steps.append(("encoder", FrequencyEncoder()))
    elif categorical_strategy == "onehot":
        from sklearn.preprocessing import OneHotEncoder
        cat_steps.append((
            "encoder",
            OneHotEncoder(handle_unknown="ignore", sparse_output=False),
        ))
    else:
        raise ValueError(f"Unknown categorical_strategy: {categorical_strategy}")

    cat_pipe = Pipeline(steps=cat_steps)

    # ── Assemble ──────────────────────────────────────────────────
    transformers = []
    if numeric_cols:
        transformers.append(("numeric", numeric_pipe, numeric_cols))
    if categorical_cols:
        transformers.append(("categorical", cat_pipe, categorical_cols))

    if not transformers:
        logger.warning("build_preprocessor: no numeric or categorical columns specified. "
                       "Returning passthrough transformer.")
        return ColumnTransformer(transformers=[("passthrough", "passthrough", slice(None))])

    return ColumnTransformer(
        transformers=transformers,
        remainder="drop",   # drop text and unrecognized columns
        verbose_feature_names_out=True,
    )


# ─────────────────────────────────────────────────────────────────
# CLIP OUTLIERS (fold-aware)
# ─────────────────────────────────────────────────────────────────
class QuantileClipper(BaseEstimator, TransformerMixin):
    """
    Clip numeric values to [lower_q, upper_q] quantiles.

    Quantiles are computed on training data only.
    """

    def __init__(self, lower_q: float = 0.001, upper_q: float = 0.999) -> None:
        self.lower_q = lower_q
        self.upper_q = upper_q
        self._lower: Optional[pd.Series] = None
        self._upper: Optional[pd.Series] = None

    def fit(self, X: pd.DataFrame, y=None) -> "QuantileClipper":
        self._lower = X.quantile(self.lower_q)
        self._upper = X.quantile(self.upper_q)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        return X.clip(lower=self._lower, upper=self._upper, axis=1)


# ─────────────────────────────────────────────────────────────────
# LEAKAGE GUARD
# ─────────────────────────────────────────────────────────────────
def check_for_leakage(
    train_df: pd.DataFrame,
    target_col: str,
    feature_cols: List[str],
) -> List[str]:
    """
    Run basic leakage checks on a feature set.

    Returns a list of warning strings for any suspected leakage.

    Checks:
      1. Target column present in features
      2. Columns with perfect or near-perfect correlation with target
      3. Columns whose name contains target-related keywords
    """
    warnings_list: List[str] = []

    # 1. Direct target inclusion
    if target_col in feature_cols:
        warnings_list.append(
            f"CRITICAL: target column '{target_col}' is in feature_cols. "
            "Remove it immediately."
        )

    # 2. High correlation with target (only for numeric features)
    if target_col in train_df.columns:
        target = train_df[target_col]
        for col in feature_cols:
            if col == target_col:
                continue
            if pd.api.types.is_numeric_dtype(train_df[col]):
                try:
                    corr = train_df[col].corr(target)
                    if abs(corr) > 0.98:
                        warnings_list.append(
                            f"HIGH CORRELATION: '{col}' has |corr|={abs(corr):.3f} "
                            f"with '{target_col}'. Possible leakage."
                        )
                except Exception:
                    pass

    # 3. Name-based heuristic
    leakage_keywords = {
        "target", "label", "score", "rank", "price", "revenue",
        "sale", "profit", "outcome", "result", "answer", "ground",
        "future", "leaked",
    }
    for col in feature_cols:
        if col == target_col:
            continue
        if any(kw in col.lower() for kw in leakage_keywords):
            warnings_list.append(
                f"NAME WARNING: '{col}' contains a leakage-risk keyword."
            )

    if warnings_list:
        for w in warnings_list:
            logger.warning("⚠️  LEAKAGE GUARD: %s", w)
    else:
        logger.info("Leakage guard: no issues detected for %d features.", len(feature_cols))

    return warnings_list
