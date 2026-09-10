"""
src/features.py
Amazon ML Challenge 2026 — Feature engineering system.

Principles:
  1. Features are computed INSIDE cross-validation folds where applicable
     (e.g., target-encoding statistics, aggregate stats).
  2. Stateless transforms (character counts, regex patterns, log) are safe
     to apply globally before the fold loop.
  3. Text features preserve numbers, units, and quantities — do NOT
     aggressively strip information from product-like text.
  4. Every feature group is independently testable.

Module structure:
  - numeric_features()      → log/ratio/interaction transforms
  - categorical_features()  → frequency, count, rank
  - text_structured_features() → length, digit, pattern, quantity features
  - tfidf_features()        → sparse TF-IDF matrix
  - create_features()       → master pipeline (calls the above)
"""

from __future__ import annotations

import re
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import scipy.sparse as sp

from src.logging_utils import get_logger

logger = get_logger(__name__)


# ─────────────────────────────────────────────────────────────────
# REGEX PATTERNS (product data)
# ─────────────────────────────────────────────────────────────────
_RE_DIGITS       = re.compile(r"\d")
_RE_NUMBERS      = re.compile(r"\b\d+\.?\d*\b")
_RE_UNITS        = re.compile(
    r"\b(?:mg|g|kg|ml|l|litre|liter|cm|mm|m|inch|inches|in|ft|oz|lb|lbs|"
    r"watt|watts|w|volt|volts|v|amp|amps|a|hz|mhz|ghz|rpm|psi|"
    r"gb|mb|tb|kb|pixel|px|dp|dpi)\b",
    re.IGNORECASE,
)
_RE_DIMENSIONS   = re.compile(r"\d+\s*[xX×]\s*\d+(?:\s*[xX×]\s*\d+)?")
_RE_PACK_SIZE    = re.compile(r"\b(?:pack\s*of|set\s*of|pack|pieces|pcs|count|ct|qty)\s*\d+|\b\d+\s*(?:pack|pcs|pieces|count)\b", re.IGNORECASE)
_RE_SKU          = re.compile(r"\b[A-Z0-9]{4,}-[A-Z0-9]{2,}\b")
_RE_MODEL_NUM    = re.compile(r"\b[A-Z]{1,4}\d{3,}\b")
_RE_BRAND_CAPS   = re.compile(r"\b[A-Z][A-Z0-9]{2,}\b")   # heuristic: all-caps brand token
_RE_URL          = re.compile(r"https?://\S+")
_RE_SPECIAL_CHAR = re.compile(r"[^a-zA-Z0-9\s]")


# ─────────────────────────────────────────────────────────────────
# TEXT STRUCTURED FEATURES
# ─────────────────────────────────────────────────────────────────
def text_structured_features(
    df: pd.DataFrame,
    text_cols: List[str],
    prefix: str = "txt",
) -> pd.DataFrame:
    """
    Extract structured numeric signals from text columns.

    These transforms are stateless — safe to apply before the CV loop.

    Features extracted per text column:
      char_count, word_count, unique_word_count, avg_word_len,
      digit_count, number_count, uppercase_ratio, special_char_count,
      has_unit, unit_count, has_dimension, has_pack_size,
      has_sku_pattern, has_model_number, has_brand_caps

    Parameters
    ----------
    df : pd.DataFrame
    text_cols : list[str]
        Column names to process.
    prefix : str
        Feature name prefix.

    Returns
    -------
    pd.DataFrame
        New DataFrame with only the generated features (same index as df).
    """
    feature_frames: List[pd.DataFrame] = []

    for col in text_cols:
        if col not in df.columns:
            logger.warning("text_structured_features: column '%s' not found — skipping.", col)
            continue

        s = df[col].fillna("").astype(str)
        col_slug = col.replace(" ", "_").replace("-", "_")
        p = f"{prefix}_{col_slug}"

        feats: Dict[str, pd.Series] = {}

        feats[f"{p}__char_count"]         = s.str.len()
        feats[f"{p}__word_count"]         = s.str.split().str.len().fillna(0)
        feats[f"{p}__unique_word_count"]  = s.str.lower().str.split().apply(
            lambda ws: len(set(ws)) if isinstance(ws, list) else 0
        )
        feats[f"{p}__avg_word_len"]       = s.str.split().apply(
            lambda ws: np.mean([len(w) for w in ws]) if ws else 0
        ).fillna(0)

        feats[f"{p}__digit_count"]        = s.str.count(_RE_DIGITS)
        feats[f"{p}__number_count"]       = s.apply(lambda x: len(_RE_NUMBERS.findall(x)))
        feats[f"{p}__uppercase_ratio"]    = s.apply(
            lambda x: sum(1 for c in x if c.isupper()) / max(len(x), 1)
        )
        feats[f"{p}__special_char_count"] = s.str.count(_RE_SPECIAL_CHAR)
        feats[f"{p}__has_unit"]           = s.str.contains(_RE_UNITS, na=False).astype(int)
        feats[f"{p}__unit_count"]         = s.apply(lambda x: len(_RE_UNITS.findall(x)))
        feats[f"{p}__has_dimension"]      = s.str.contains(_RE_DIMENSIONS, na=False).astype(int)
        feats[f"{p}__has_pack_size"]      = s.str.contains(_RE_PACK_SIZE, na=False).astype(int)
        feats[f"{p}__has_sku_pattern"]    = s.str.contains(_RE_SKU, na=False).astype(int)
        feats[f"{p}__has_model_number"]   = s.str.contains(_RE_MODEL_NUM, na=False).astype(int)
        feats[f"{p}__has_brand_caps"]     = s.str.contains(_RE_BRAND_CAPS, na=False).astype(int)
        feats[f"{p}__has_url"]            = s.str.contains(_RE_URL, na=False).astype(int)

        feature_frames.append(pd.DataFrame(feats, index=df.index))

    if not feature_frames:
        return pd.DataFrame(index=df.index)

    result = pd.concat(feature_frames, axis=1)
    logger.info("text_structured_features: generated %d features from %d text cols.", len(result.columns), len(text_cols))
    return result


# ─────────────────────────────────────────────────────────────────
# NUMERIC FEATURES
# ─────────────────────────────────────────────────────────────────
def numeric_features(
    df: pd.DataFrame,
    numeric_cols: List[str],
    log_cols: Optional[List[str]] = None,
    ratio_pairs: Optional[List[Tuple[str, str]]] = None,
    interaction_pairs: Optional[List[Tuple[str, str]]] = None,
    prefix: str = "num",
) -> pd.DataFrame:
    """
    Derive numeric features: log transforms, ratios, interactions.

    Parameters
    ----------
    df : pd.DataFrame
    numeric_cols : list[str]
        Base numeric columns to include as-is.
    log_cols : list[str] | None
        Columns to log1p-transform.
    ratio_pairs : list[(str, str)] | None
        Pairs (a, b) → feature a / (b + eps).
    interaction_pairs : list[(str, str)] | None
        Pairs (a, b) → feature a * b.
    prefix : str

    Returns
    -------
    pd.DataFrame
    """
    feats: Dict[str, pd.Series] = {}
    eps = 1e-8

    # Base passthrough
    for col in numeric_cols:
        if col in df.columns:
            feats[f"{prefix}__{col}"] = df[col]

    # Log transforms
    for col in (log_cols or []):
        if col in df.columns:
            feats[f"{prefix}__log1p_{col}"] = np.log1p(df[col].clip(lower=0))

    # Ratios
    for a, b in (ratio_pairs or []):
        if a in df.columns and b in df.columns:
            feats[f"{prefix}__ratio_{a}__{b}"] = df[a] / (df[b] + eps)

    # Interactions
    for a, b in (interaction_pairs or []):
        if a in df.columns and b in df.columns:
            feats[f"{prefix}__interact_{a}__{b}"] = df[a] * df[b]

    result = pd.DataFrame(feats, index=df.index)
    logger.info("numeric_features: generated %d features.", len(result.columns))
    return result


# ─────────────────────────────────────────────────────────────────
# CATEGORICAL FEATURES
# ─────────────────────────────────────────────────────────────────
def categorical_features(
    df: pd.DataFrame,
    categorical_cols: List[str],
    train_df: Optional[pd.DataFrame] = None,
    prefix: str = "cat",
) -> pd.DataFrame:
    """
    Frequency-based categorical features.

    IMPORTANT: Frequency statistics are computed from train_df only
    (or from df itself if train_df is None — safe when called inside
    a fold with only training data).

    Parameters
    ----------
    df : pd.DataFrame
        Data to transform.
    categorical_cols : list[str]
    train_df : pd.DataFrame | None
        Reference DataFrame for frequency computation.
        MUST be the training fold, not the full dataset.
    prefix : str

    Returns
    -------
    pd.DataFrame
    """
    ref = train_df if train_df is not None else df
    feats: Dict[str, pd.Series] = {}

    for col in categorical_cols:
        if col not in df.columns:
            logger.warning("categorical_features: '%s' not found — skipping.", col)
            continue

        # Frequency encoding from training data
        freq_map = ref[col].value_counts(normalize=True).to_dict()
        feats[f"{prefix}__freq_{col}"] = df[col].map(freq_map).fillna(0.0)

        # Count encoding (absolute counts from training)
        cnt_map = ref[col].value_counts().to_dict()
        feats[f"{prefix}__cnt_{col}"] = df[col].map(cnt_map).fillna(0)

    result = pd.DataFrame(feats, index=df.index)
    logger.info("categorical_features: generated %d features.", len(result.columns))
    return result


# ─────────────────────────────────────────────────────────────────
# TF-IDF FEATURES
# ─────────────────────────────────────────────────────────────────
class TfidfFeatureBuilder:
    """
    Thin wrapper around sklearn TfidfVectorizer for competition use.

    Must be fit on training data only (inside each fold).
    Supports combining multiple text columns.
    """

    def __init__(
        self,
        max_features: int = 50_000,
        ngram_range: Tuple[int, int] = (1, 2),
        sublinear_tf: bool = True,
        analyzer: str = "word",
    ) -> None:
        self.max_features = max_features
        self.ngram_range = ngram_range
        self.sublinear_tf = sublinear_tf
        self.analyzer = analyzer
        self._vectorizers: Dict[str, TfidfVectorizer] = {}

    def fit(self, df: pd.DataFrame, text_cols: List[str]) -> "TfidfFeatureBuilder":
        """Fit vectorizers on training data only."""
        self._vectorizers = {}
        for col in text_cols:
            if col not in df.columns:
                continue
            texts = df[col].fillna("").astype(str).tolist()
            vec = TfidfVectorizer(
                max_features=self.max_features,
                ngram_range=self.ngram_range,
                sublinear_tf=self.sublinear_tf,
                analyzer=self.analyzer,
                dtype=np.float32,
            )
            vec.fit(texts)
            self._vectorizers[col] = vec
        logger.info("TfidfFeatureBuilder: fit on %d text cols.", len(self._vectorizers))
        return self

    def transform(self, df: pd.DataFrame) -> sp.csr_matrix:
        """Transform — returns a sparse matrix of shape (n_rows, total_features)."""
        matrices = []
        for col, vec in self._vectorizers.items():
            texts = df[col].fillna("").astype(str).tolist()
            matrices.append(vec.transform(texts))
        if not matrices:
            return sp.csr_matrix((len(df), 0))
        return sp.hstack(matrices, format="csr")

    def fit_transform(self, df: pd.DataFrame, text_cols: List[str]) -> sp.csr_matrix:
        return self.fit(df, text_cols).transform(df)


# ─────────────────────────────────────────────────────────────────
# MASTER FEATURE CREATOR
# ─────────────────────────────────────────────────────────────────
def create_features(
    df: pd.DataFrame,
    numeric_cols: Optional[List[str]] = None,
    categorical_cols: Optional[List[str]] = None,
    text_cols: Optional[List[str]] = None,
    train_df: Optional[pd.DataFrame] = None,
    log_cols: Optional[List[str]] = None,
    ratio_pairs: Optional[List[Tuple[str, str]]] = None,
    interaction_pairs: Optional[List[Tuple[str, str]]] = None,
    include_text_structured: bool = True,
) -> pd.DataFrame:
    """
    Master feature creation pipeline.

    Call this inside each CV fold:
        - train_df = train fold only (for frequency/count statistics)
        - df = whichever split you're transforming

    Parameters
    ----------
    df : pd.DataFrame
        Data to transform.
    numeric_cols, categorical_cols, text_cols : list | None
    train_df : pd.DataFrame | None
        MUST be the training fold (to avoid leakage in statistics).
    log_cols : list | None
        Numeric columns to log-transform.
    ratio_pairs, interaction_pairs : list | None
    include_text_structured : bool
        Whether to include text structural features.

    Returns
    -------
    pd.DataFrame
        Dense feature matrix (sparse TF-IDF is handled separately).
    """
    frames: List[pd.DataFrame] = []

    # Numeric
    if numeric_cols:
        frames.append(numeric_features(
            df,
            numeric_cols,
            log_cols=log_cols,
            ratio_pairs=ratio_pairs,
            interaction_pairs=interaction_pairs,
        ))

    # Categorical
    if categorical_cols:
        frames.append(categorical_features(df, categorical_cols, train_df=train_df))

    # Text structured
    if text_cols and include_text_structured:
        frames.append(text_structured_features(df, text_cols))

    if not frames:
        logger.warning("create_features: no feature groups specified. Returning empty DataFrame.")
        return pd.DataFrame(index=df.index)

    result = pd.concat(frames, axis=1)

    # Sanity check: no NaN/Inf should appear in engineered features
    n_nan = result.isna().sum().sum()
    n_inf = np.isinf(result.select_dtypes(include=np.number).values).sum()
    if n_nan > 0 or n_inf > 0:
        logger.warning(
            "create_features: %d NaN and %d Inf values in output — "
            "check imputation or clipping.",
            n_nan, n_inf,
        )

    logger.info("create_features: output shape = %s", result.shape)
    return result
