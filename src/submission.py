"""
src/submission.py
Amazon ML Challenge 2026 — Submission validation and packaging.

The submission pipeline runs EVERY check before writing the final file.
No silent failures. Every validation step is logged.

Pipeline:
  predictions
       ↓
  alignment (IDs)
       ↓
  post-processing (clip, round, log-inverse)
       ↓
  schema validation
       ↓
  NaN check
       ↓
  Inf check
       ↓
  duplicate ID check
       ↓
  row count check
       ↓
  column check
       ↓
  range check
       ↓
  write CSV
       ↓
  reload and re-validate
       ↓
  write sanity_report.txt
       ↓
  write submission_metadata.json
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

from src.logging_utils import get_logger
from src.utils import ensure_dir, save_json

logger = get_logger(__name__)


# ─────────────────────────────────────────────────────────────────
# POST-PROCESSING
# ─────────────────────────────────────────────────────────────────
def postprocess_predictions(
    predictions: np.ndarray,
    clip_min: Optional[float] = None,
    clip_max: Optional[float] = None,
    round_digits: Optional[int] = None,
    log_inverse: bool = False,
    expm1: bool = False,
) -> np.ndarray:
    """
    Apply controlled post-processing to raw model predictions.

    Every operation is explicit and logged — no hidden transforms.

    Parameters
    ----------
    predictions : np.ndarray
    clip_min, clip_max : float | None
        Clip predictions to [clip_min, clip_max].
    round_digits : int | None
        Round to N decimal places.
    log_inverse : bool
        Apply np.exp() (inverse of log transform on target).
    expm1 : bool
        Apply np.expm1() (inverse of log1p transform on target).

    Returns
    -------
    np.ndarray
    """
    if log_inverse and expm1:
        raise ValueError("Cannot specify both log_inverse=True and expm1=True. Choose one.")

    preds = predictions.copy().astype(np.float64)

    if log_inverse:
        logger.info("Post-process: applying exp() (log-inverse)")
        preds = np.exp(preds)

    if expm1:
        logger.info("Post-process: applying expm1() (log1p-inverse)")
        preds = np.expm1(preds)

    if clip_min is not None or clip_max is not None:
        before_min, before_max = preds.min(), preds.max()
        preds = np.clip(preds, clip_min, clip_max)
        logger.info("Post-process: clipped [%.4f, %.4f] → [%.4f, %.4f]",
                    before_min, before_max, preds.min(), preds.max())

    if round_digits is not None:
        preds = np.round(preds, round_digits)
        logger.info("Post-process: rounded to %d decimal places", round_digits)

    return preds


# ─────────────────────────────────────────────────────────────────
# VALIDATION CHECKS
# ─────────────────────────────────────────────────────────────────
@dataclass
class ValidationCheck:
    name: str
    passed: bool
    detail: str = ""


def _check(name: str, cond: bool, detail: str = "") -> ValidationCheck:
    return ValidationCheck(name=name, passed=cond, detail=detail)


def validate_submission(
    submission_df: pd.DataFrame,
    id_column: str,
    prediction_column: str,
    expected_row_count: Optional[int] = None,
    clip_min: Optional[float] = None,
    clip_max: Optional[float] = None,
    test_ids: Optional[pd.Series] = None,
) -> Tuple[bool, List[ValidationCheck]]:
    """
    Run all submission validation checks.

    Parameters
    ----------
    submission_df : pd.DataFrame
        The submission DataFrame to validate.
    id_column : str
    prediction_column : str
    expected_row_count : int | None
        If set, verify the row count matches.
    clip_min, clip_max : float | None
        If set, verify no predictions fall outside this range.
    test_ids : pd.Series | None
        Expected IDs from the test set. Checks for alignment.

    Returns
    -------
    (all_pass: bool, checks: list[ValidationCheck])
    """
    checks: List[ValidationCheck] = []

    # 1. Required columns present
    has_id  = id_column in submission_df.columns
    has_pred = prediction_column in submission_df.columns
    checks.append(_check("ID column present",         has_id,   f"Missing: '{id_column}'"))
    checks.append(_check("Prediction column present", has_pred, f"Missing: '{prediction_column}'"))

    if not (has_id and has_pred):
        return False, checks

    preds = submission_df[prediction_column]

    is_numeric = pd.api.types.is_numeric_dtype(preds)
    checks.append(_check("Predictions are numeric", is_numeric, f"Dtype is {preds.dtype}"))

    # 2. No NaN
    n_nan = preds.isna().sum()
    checks.append(_check("No NaN predictions", n_nan == 0, f"{n_nan} NaN values"))

    # 3. No Inf
    numeric_preds = pd.to_numeric(preds, errors="coerce")
    n_inf = np.isinf(numeric_preds.dropna().values).sum()
    checks.append(_check("No Inf predictions", n_inf == 0, f"{n_inf} Inf values"))

    # 4. Duplicate IDs
    n_dup_ids = submission_df[id_column].duplicated().sum()
    checks.append(_check("No duplicate IDs", n_dup_ids == 0, f"{n_dup_ids} duplicate IDs"))

    # 5. Row count
    n_rows = len(submission_df)
    if expected_row_count is not None:
        match = n_rows == expected_row_count
        checks.append(_check(
            "Row count matches test set",
            match,
            f"Got {n_rows}, expected {expected_row_count}",
        ))
    else:
        checks.append(_check("Non-empty submission", n_rows > 0, f"{n_rows} rows"))

    # 6. ID alignment
    if test_ids is not None:
        sub_ids_list = submission_df[id_column].astype(str).tolist()
        test_ids_list = test_ids.astype(str).tolist()
        
        sub_ids_set  = set(sub_ids_list)
        test_set = set(test_ids_list)
        missing  = test_set - sub_ids_set
        extra    = sub_ids_set - test_set
        checks.append(_check(
            "All test IDs present in submission",
            len(missing) == 0,
            f"{len(missing)} missing IDs (sample: {list(missing)[:5]})",
        ))
        checks.append(_check(
            "No extra IDs in submission",
            len(extra) == 0,
            f"{len(extra)} extra IDs",
        ))
        checks.append(_check(
            "ID order matches test set",
            sub_ids_list == test_ids_list,
            "Submission ID order differs from test set",
        ))

    # 7. Range check
    if clip_min is not None:
        n_below = (numeric_preds < clip_min).sum()
        checks.append(_check(
            f"All predictions ≥ {clip_min}",
            n_below == 0,
            f"{n_below} predictions below {clip_min}",
        ))
    if clip_max is not None:
        n_above = (numeric_preds > clip_max).sum()
        checks.append(_check(
            f"All predictions ≤ {clip_max}",
            n_above == 0,
            f"{n_above} predictions above {clip_max}",
        ))

    all_pass = all(c.passed for c in checks)
    return all_pass, checks


# ─────────────────────────────────────────────────────────────────
# CREATE SUBMISSION DATAFRAME
# ─────────────────────────────────────────────────────────────────
def create_submission_df(
    test_df: pd.DataFrame,
    predictions: np.ndarray,
    id_column: str,
    prediction_column: str,
) -> pd.DataFrame:
    """
    Align predictions with test IDs and create the submission DataFrame.

    Parameters
    ----------
    test_df : pd.DataFrame
        The original test dataset containing id_column.
    predictions : np.ndarray
        Model predictions, aligned to test_df row order.
    id_column : str
    prediction_column : str

    Returns
    -------
    pd.DataFrame with columns [id_column, prediction_column]
    """
    if id_column not in test_df.columns:
        raise ValueError(
            f"ID column '{id_column}' not found in test_df. "
            f"Available columns: {list(test_df.columns)}"
        )
    if len(predictions) != len(test_df):
        raise ValueError(
            f"Length mismatch: test_df has {len(test_df)} rows "
            f"but predictions has {len(predictions)} values."
        )

    sub = pd.DataFrame({
        id_column: test_df[id_column].values,
        prediction_column: predictions,
    })
    return sub


# ─────────────────────────────────────────────────────────────────
# WRITE SUBMISSION
# ─────────────────────────────────────────────────────────────────
def write_submission(
    submission_df: pd.DataFrame,
    id_column: str,
    prediction_column: str,
    output_dir: str = "submission",
    filename: str = "final_submission.csv",
    expected_row_count: Optional[int] = None,
    clip_min: Optional[float] = None,
    clip_max: Optional[float] = None,
    test_ids: Optional[pd.Series] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> Path:
    """
    Validate and write the final submission CSV.

    This is the ONLY way the final submission file should be created.
    Never manually write predictions to CSV.

    Parameters
    ----------
    submission_df : pd.DataFrame
    id_column, prediction_column : str
    output_dir : str
    filename : str
    expected_row_count : int | None
    clip_min, clip_max : float | None
    test_ids : pd.Series | None
    metadata : dict | None
        Additional metadata saved to submission_metadata.json.

    Returns
    -------
    Path to the written CSV file.

    Raises
    ------
    RuntimeError
        If any validation check fails.
    """
    out_dir = Path(output_dir)
    ensure_dir(out_dir)
    out_path = out_dir / filename

    # ── Pre-write validation ─────────────────────────────────────
    all_pass, checks = validate_submission(
        submission_df,
        id_column=id_column,
        prediction_column=prediction_column,
        expected_row_count=expected_row_count,
        clip_min=clip_min,
        clip_max=clip_max,
        test_ids=test_ids,
    )

    if not all_pass:
        fail_details = [c.detail for c in checks if not c.passed]
        raise RuntimeError(
            f"Submission validation FAILED before writing:\n"
            + "\n".join(f"  ❌ {d}" for d in fail_details)
        )

    # ── Write CSV ────────────────────────────────────────────────
    submission_df[[id_column, prediction_column]].to_csv(out_path, index=False)
    logger.info("Submission written → %s (%d rows)", out_path, len(submission_df))

    # ── Reload and re-validate ────────────────────────────────────
    reloaded = pd.read_csv(out_path)
    final_pass, final_checks = validate_submission(
        reloaded,
        id_column=id_column,
        prediction_column=prediction_column,
        expected_row_count=expected_row_count,
        clip_min=clip_min,
        clip_max=clip_max,
        test_ids=test_ids,
    )
    if not final_pass:
        raise RuntimeError("Submission re-validation after write FAILED — do not submit this file.")

    # ── Write sanity report ───────────────────────────────────────
    _write_sanity_report(out_dir, checks, submission_df, prediction_column)

    # ── Write metadata ────────────────────────────────────────────
    preds = submission_df[prediction_column]
    meta = {
        "timestamp": datetime.now().isoformat(),
        "filename": filename,
        "rows": len(submission_df),
        "id_column": id_column,
        "prediction_column": prediction_column,
        "prediction_stats": {
            "mean":   float(preds.mean()),
            "std":    float(preds.std()),
            "min":    float(preds.min()),
            "max":    float(preds.max()),
            "median": float(preds.median()),
        },
        "validation_status": "PASS",
    }
    if metadata:
        meta.update(metadata)
    save_json(meta, out_dir / "submission_metadata.json")

    logger.info("✅ Submission validation: PASS")
    return out_path


def _write_sanity_report(
    out_dir: Path,
    checks: List[ValidationCheck],
    submission_df: pd.DataFrame,
    prediction_column: str,
) -> None:
    """Write a human-readable sanity report."""
    lines = [
        "SUBMISSION SANITY REPORT",
        "=" * 50,
        f"Timestamp: {datetime.now().isoformat()}",
        f"Rows: {len(submission_df)}",
        "",
        "VALIDATION CHECKS",
        "-" * 50,
    ]
    for c in checks:
        icon = "PASS" if c.passed else "FAIL"
        line = f"[{icon}] {c.name}"
        if c.detail:
            line += f"  →  {c.detail}"
        lines.append(line)

    preds = submission_df[prediction_column]
    lines += [
        "",
        "PREDICTION STATISTICS",
        "-" * 50,
        f"Mean   : {preds.mean():.6f}",
        f"Std    : {preds.std():.6f}",
        f"Min    : {preds.min():.6f}",
        f"Max    : {preds.max():.6f}",
        f"Median : {preds.median():.6f}",
        f"NaN    : {preds.isna().sum()}",
        "",
        "=" * 50,
    ]

    report_path = out_dir / "sanity_report.txt"
    report_path.write_text("\n".join(lines), encoding="utf-8")
    logger.info("Sanity report → %s", report_path)
