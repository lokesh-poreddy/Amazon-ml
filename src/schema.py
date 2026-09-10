"""
src/schema.py
Amazon ML Challenge 2026 — Automatic schema audit and column classification.

This module inspects an unknown dataset and produces a structured report:
  - Column types (numeric / categorical / text / datetime / identifier / URL / image)
  - Missingness
  - Cardinality
  - Train/test differences
  - Suspicious columns (leakage indicators, constants, near-constants, duplicates)
  - Probable target and ID columns

Critical: this module never modifies data — it only audits and reports.
"""

from __future__ import annotations

import re
import warnings
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd

from src.logging_utils import get_logger

logger = get_logger(__name__)


# ─────────────────────────────────────────────────────────────────
# COLUMN TYPE LABELS
# ─────────────────────────────────────────────────────────────────
NUMERIC       = "numeric"
CATEGORICAL   = "categorical"
HIGH_CARD_CAT = "high_cardinality_categorical"
TEXT          = "text"
BOOLEAN       = "boolean"
DATETIME      = "datetime"
IDENTIFIER    = "identifier"
URL           = "url"
IMAGE_PATH    = "image_path"
CONSTANT      = "constant"
UNKNOWN_TYPE  = "unknown"

# Heuristic thresholds
_HIGH_CARD_RATIO    = 0.5   # unique/total > this → high cardinality
_TEXT_MEAN_WORDS    = 4.0   # mean word count > this → text
_ID_UNIQUE_RATIO    = 0.95  # unique/total > this → likely identifier
_NEAR_CONST_RATIO   = 0.995 # one value covers > this fraction → near-constant

# URL / image path patterns
_URL_RE        = re.compile(r"https?://", re.IGNORECASE)
_IMG_EXT_RE    = re.compile(r"\.(?:jpg|jpeg|png|webp|gif|bmp|tiff)$", re.IGNORECASE)


# ─────────────────────────────────────────────────────────────────
# PER-COLUMN REPORT
# ─────────────────────────────────────────────────────────────────
@dataclass
class ColumnReport:
    name: str
    dtype: str
    inferred_type: str
    n_missing: int
    pct_missing: float
    n_unique: int
    unique_ratio: float
    sample_values: List[Any]
    warnings: List[str] = field(default_factory=list)


# ─────────────────────────────────────────────────────────────────
# DATASET AUDIT REPORT
# ─────────────────────────────────────────────────────────────────
@dataclass
class AuditReport:
    """Full audit report for a single dataset."""

    source: str         # "train" | "test" | "other"
    n_rows: int
    n_cols: int
    columns: List[ColumnReport]
    duplicate_rows: int
    duplicate_id_count: int
    probable_target: Optional[str]
    probable_id: Optional[str]
    suspicious: List[str]           # column names flagged as suspicious
    constant_cols: List[str]
    near_constant_cols: List[str]
    leakage_candidates: List[str]

    def col_by_type(self, type_label: str) -> List[str]:
        return [c.name for c in self.columns if c.inferred_type == type_label]

    @property
    def numeric_cols(self) -> List[str]:
        return self.col_by_type(NUMERIC)

    @property
    def categorical_cols(self) -> List[str]:
        return self.col_by_type(CATEGORICAL) + self.col_by_type(HIGH_CARD_CAT)

    @property
    def text_cols(self) -> List[str]:
        return self.col_by_type(TEXT)

    @property
    def datetime_cols(self) -> List[str]:
        return self.col_by_type(DATETIME)

    @property
    def identifier_cols(self) -> List[str]:
        return self.col_by_type(IDENTIFIER)

    @property
    def url_cols(self) -> List[str]:
        return self.col_by_type(URL)

    @property
    def image_cols(self) -> List[str]:
        return self.col_by_type(IMAGE_PATH)


# ─────────────────────────────────────────────────────────────────
# COLUMN TYPE INFERENCE
# ─────────────────────────────────────────────────────────────────
def _infer_column_type(series: pd.Series) -> Tuple[str, List[str]]:
    """
    Infer the semantic type of a column.

    Returns
    -------
    (type_label, warnings)
    """
    warns: List[str] = []
    n = len(series)
    n_unique = series.nunique(dropna=True)
    unique_ratio = n_unique / n if n > 0 else 0

    non_null = series.dropna()
    if len(non_null) == 0:
        return UNKNOWN_TYPE, ["All values are null"]

    # ── Constant ─────────────────────────────────────────────────
    if n_unique <= 1:
        return CONSTANT, ["Column is constant — provides no signal"]

    # ── Boolean ──────────────────────────────────────────────────
    if set(non_null.unique()).issubset({True, False, 0, 1, "True", "False", "true", "false"}):
        return BOOLEAN, warns

    # ── Numeric ──────────────────────────────────────────────────
    if pd.api.types.is_numeric_dtype(series):
        if n_unique == 2:
            return BOOLEAN, warns
        return NUMERIC, warns

    # ── Datetime ─────────────────────────────────────────────────
    if pd.api.types.is_datetime64_any_dtype(series):
        return DATETIME, warns

    # ── Object / string columns need deeper inspection ───────────
    sample_str = non_null.astype(str).head(200)

    # Try numeric parsing
    numeric_parsed = pd.to_numeric(non_null.head(500), errors="coerce")
    numeric_frac = numeric_parsed.notna().mean()
    if numeric_frac > 0.8:
        return NUMERIC, warns

    # Try datetime parsing
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            dt_parsed = pd.to_datetime(non_null.head(200), errors="coerce", infer_datetime_format=True)
        if dt_parsed.notna().mean() > 0.8:
            return DATETIME, warns
    except Exception:
        pass

    # URL
    url_frac = sample_str.str.contains(_URL_RE, na=False).mean()
    if url_frac > 0.7:
        img_frac = sample_str.str.contains(_IMG_EXT_RE, na=False).mean()
        if img_frac > 0.5:
            return IMAGE_PATH, warns
        return URL, warns

    # Image path (local)
    img_frac = sample_str.str.contains(_IMG_EXT_RE, na=False).mean()
    if img_frac > 0.5:
        return IMAGE_PATH, warns

    # Text (long strings) — CHECK BEFORE identifier to catch product descriptions
    mean_words = sample_str.str.split().str.len().mean()
    if mean_words > _TEXT_MEAN_WORDS:
        return TEXT, warns

    # Identifier check
    if unique_ratio > _ID_UNIQUE_RATIO:
        return IDENTIFIER, ["High unique ratio — likely an ID column"]

    # High-cardinality categorical
    if unique_ratio > _HIGH_CARD_RATIO:
        return HIGH_CARD_CAT, [f"High cardinality ({n_unique} unique / {n} rows)"]

    return CATEGORICAL, warns


# ─────────────────────────────────────────────────────────────────
# SINGLE DATASET AUDIT
# ─────────────────────────────────────────────────────────────────
def audit_dataset(
    df: pd.DataFrame,
    source: str = "dataset",
    id_column: Optional[str] = None,
    target_column: Optional[str] = None,
) -> AuditReport:
    """
    Inspect a single DataFrame and produce a structured AuditReport.

    Parameters
    ----------
    df : pd.DataFrame
    source : str
        Label used in reports ("train" | "test").
    id_column : str | None
        Known ID column (if already identified).
    target_column : str | None
        Known target column (if already identified).

    Returns
    -------
    AuditReport
    """
    n_rows, n_cols = df.shape
    col_reports: List[ColumnReport] = []
    constant_cols: List[str] = []
    near_constant_cols: List[str] = []
    leakage_candidates: List[str] = []
    suspicious: List[str] = []

    for col in df.columns:
        series = df[col]
        n_missing = series.isna().sum()
        pct_missing = 100.0 * n_missing / n_rows if n_rows > 0 else 0.0
        n_unique = series.nunique(dropna=True)
        unique_ratio = n_unique / n_rows if n_rows > 0 else 0.0
        sample_vals = series.dropna().head(5).tolist()

        inferred_type, warns = _infer_column_type(series)

        # Near-constant check
        if n_unique > 1:
            top_freq = series.value_counts(dropna=False).iloc[0] / n_rows
            if top_freq > _NEAR_CONST_RATIO:
                near_constant_cols.append(col)
                warns.append(f"Near-constant — top value covers {100*top_freq:.1f}% of rows")

        if inferred_type == CONSTANT:
            constant_cols.append(col)

        # Leakage heuristics
        low = col.lower()
        leakage_keywords = {
            "target", "label", "score", "rank", "price", "revenue",
            "sale", "profit", "outcome", "result", "answer", "ground",
            "future", "leaked",
        }
        if target_column and col == target_column:
            pass  # expected
        elif any(kw in low for kw in leakage_keywords) and col != target_column:
            leakage_candidates.append(col)
            warns.append("⚠️  Possible leakage: column name suggests it may encode the target")

        col_reports.append(ColumnReport(
            name=col,
            dtype=str(series.dtype),
            inferred_type=inferred_type,
            n_missing=int(n_missing),
            pct_missing=round(pct_missing, 2),
            n_unique=int(n_unique),
            unique_ratio=round(unique_ratio, 4),
            sample_values=sample_vals,
            warnings=warns,
        ))

    # ── Duplicate rows ───────────────────────────────────────────
    dup_rows = int(df.duplicated().sum())

    # ── Duplicate IDs ────────────────────────────────────────────
    dup_ids = 0
    if id_column and id_column in df.columns:
        dup_ids = int(df[id_column].duplicated().sum())

    # ── Probable target ──────────────────────────────────────────
    probable_target = target_column
    if probable_target is None:
        # Very rough heuristic: numeric column not named like an ID
        numeric_non_id = [
            c.name for c in col_reports
            if c.inferred_type == NUMERIC
            and c.inferred_type != IDENTIFIER
            and "id" not in c.name.lower()
            and "index" not in c.name.lower()
        ]
        probable_target = numeric_non_id[-1] if numeric_non_id else None

    # ── Probable ID ──────────────────────────────────────────────
    probable_id = id_column
    if probable_id is None:
        id_candidates = [
            c.name for c in col_reports
            if c.inferred_type == IDENTIFIER
            or ("id" in c.name.lower() and c.unique_ratio > 0.9)
            or "index" in c.name.lower()
        ]
        probable_id = id_candidates[0] if id_candidates else None

    return AuditReport(
        source=source,
        n_rows=n_rows,
        n_cols=n_cols,
        columns=col_reports,
        duplicate_rows=dup_rows,
        duplicate_id_count=dup_ids,
        probable_target=probable_target,
        probable_id=probable_id,
        suspicious=suspicious,
        constant_cols=constant_cols,
        near_constant_cols=near_constant_cols,
        leakage_candidates=leakage_candidates,
    )


# ─────────────────────────────────────────────────────────────────
# TRAIN / TEST COMPARISON
# ─────────────────────────────────────────────────────────────────
@dataclass
class ComparisonReport:
    train_only_cols: List[str]
    test_only_cols: List[str]
    common_cols: List[str]
    dtype_mismatches: Dict[str, Tuple[str, str]]    # col → (train_dtype, test_dtype)
    category_mismatches: Dict[str, Dict]             # col → {train_only, test_only}
    row_ratio: float                                  # train_rows / test_rows


def compare_train_test(
    train: pd.DataFrame,
    test: pd.DataFrame,
    target_column: Optional[str] = None,
    categorical_cols: Optional[List[str]] = None,
) -> ComparisonReport:
    """
    Compare train and test DataFrames for structural differences.

    Parameters
    ----------
    train, test : pd.DataFrame
    target_column : str | None
        Expected to be train-only (excluded from mismatch warnings).
    categorical_cols : list[str] | None
        Columns for which to compare category distributions.

    Returns
    -------
    ComparisonReport
    """
    train_cols: Set[str] = set(train.columns)
    test_cols: Set[str] = set(test.columns)

    train_cols_compare = train_cols - {target_column} if target_column else train_cols
    train_only = sorted((train_cols_compare - test_cols) | ({target_column} if target_column and target_column in train_cols else set()))
    test_only  = sorted(test_cols - train_cols_compare)
    common     = sorted(train_cols_compare & test_cols)

    dtype_mismatches: Dict[str, Tuple[str, str]] = {}
    for col in common:
        td, te = str(train[col].dtype), str(test[col].dtype)
        if td != te:
            dtype_mismatches[col] = (td, te)

    category_mismatches: Dict[str, Dict] = {}
    if categorical_cols:
        for col in categorical_cols:
            if col in common:
                train_cats = set(train[col].dropna().unique())
                test_cats  = set(test[col].dropna().unique())
                train_only_cats = train_cats - test_cats
                test_only_cats  = test_cats - train_cats
                if train_only_cats or test_only_cats:
                    category_mismatches[col] = {
                        "train_only": sorted(str(x) for x in train_only_cats)[:20],
                        "test_only":  sorted(str(x) for x in test_only_cats)[:20],
                    }

    row_ratio = len(train) / max(len(test), 1)

    return ComparisonReport(
        train_only_cols=train_only,
        test_only_cols=test_only,
        common_cols=common,
        dtype_mismatches=dtype_mismatches,
        category_mismatches=category_mismatches,
        row_ratio=row_ratio,
    )


# ─────────────────────────────────────────────────────────────────
# PRINT REPORT
# ─────────────────────────────────────────────────────────────────
def print_audit(report: AuditReport, verbose: bool = True) -> str:
    """
    Format and return an audit report as a readable string.

    Parameters
    ----------
    report : AuditReport
    verbose : bool
        If True, include per-column details. If False, summary only.
    """
    lines = [
        "",
        "=" * 65,
        f"  DATASET AUDIT — {report.source.upper()}",
        "=" * 65,
        f"  Rows          : {report.n_rows:,}",
        f"  Columns       : {report.n_cols}",
        f"  Duplicate rows: {report.duplicate_rows}",
        f"  Duplicate IDs : {report.duplicate_id_count}",
        f"  Probable target  : {report.probable_target or '❓ unknown'}",
        f"  Probable ID      : {report.probable_id or '❓ unknown'}",
        "",
        "  ── COLUMN TYPES ──────────────────────────────────────",
        f"  Numeric          : {', '.join(report.numeric_cols) or 'none'}",
        f"  Categorical      : {', '.join(report.categorical_cols) or 'none'}",
        f"  Text             : {', '.join(report.text_cols) or 'none'}",
        f"  Datetime         : {', '.join(report.datetime_cols) or 'none'}",
        f"  Identifier       : {', '.join(report.identifier_cols) or 'none'}",
        f"  URL              : {', '.join(report.url_cols) or 'none'}",
        f"  Image paths      : {', '.join(report.image_cols) or 'none'}",
        f"  Constant         : {', '.join(report.constant_cols) or 'none'}",
        f"  Near-constant    : {', '.join(report.near_constant_cols) or 'none'}",
        "",
        "  ── LEAKAGE CANDIDATES ────────────────────────────────",
        f"  {', '.join(report.leakage_candidates) or 'none detected'}",
    ]

    if verbose:
        lines += [
            "",
            "  ── PER-COLUMN DETAILS ────────────────────────────────",
        ]
        for cr in report.columns:
            lines.append(
                f"  [{cr.inferred_type:22s}] {cr.name:35s} "
                f"miss={cr.pct_missing:5.1f}%  uniq={cr.n_unique:6d}  "
                f"sample={cr.sample_values[:3]}"
            )
            for w in cr.warnings:
                lines.append(f"    ⚠️  {w}")

    lines.append("=" * 65 + "\n")
    report_str = "\n".join(lines)
    logger.info(report_str)
    return report_str


def print_comparison(cmp: ComparisonReport) -> str:
    lines = [
        "",
        "=" * 65,
        "  TRAIN / TEST COMPARISON",
        "=" * 65,
        f"  Train-only columns : {', '.join(cmp.train_only_cols) or 'none'}",
        f"  Test-only columns  : {', '.join(cmp.test_only_cols) or 'none'}",
        f"  Common columns     : {len(cmp.common_cols)}",
        f"  Train/Test rows    : {cmp.row_ratio:.2f}×",
    ]
    if cmp.dtype_mismatches:
        lines.append("  ── Dtype mismatches ──────────────────────────────────")
        for col, (td, te) in cmp.dtype_mismatches.items():
            lines.append(f"    ⚠️  {col}: train={td}  test={te}")
    if cmp.category_mismatches:
        lines.append("  ── Category mismatches ───────────────────────────────")
        for col, detail in cmp.category_mismatches.items():
            lines.append(
                f"    {col}: +{len(detail['train_only'])} train-only cats, "
                f"+{len(detail['test_only'])} test-only cats"
            )
    lines.append("=" * 65 + "\n")
    out = "\n".join(lines)
    logger.info(out)
    return out
