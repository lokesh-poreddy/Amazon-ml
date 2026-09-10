"""
tests/test_submission.py
Amazon ML Challenge 2026 — Submission pipeline tests.

Tests the critical validation logic that prevents a bad submission
from being written to disk.
"""

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.submission import (
    create_submission_df,
    postprocess_predictions,
    validate_submission,
    write_submission,
)


# ─────────────────────────────────────────────────────────────────
# FIXTURES
# ─────────────────────────────────────────────────────────────────
@pytest.fixture
def clean_submission() -> pd.DataFrame:
    return pd.DataFrame({
        "sample_id": [f"test_{i}" for i in range(100)],
        "entity_value": np.random.uniform(0.1, 100.0, 100),
    })


@pytest.fixture
def test_df_with_ids() -> pd.DataFrame:
    return pd.DataFrame({
        "sample_id": [f"test_{i}" for i in range(100)],
        "feature_a": np.random.rand(100),
    })


# ─────────────────────────────────────────────────────────────────
# POSTPROCESS
# ─────────────────────────────────────────────────────────────────
def test_postprocess_clip() -> None:
    preds = np.array([-5.0, 0.5, 200.0])
    result = postprocess_predictions(preds, clip_min=0.0, clip_max=100.0)
    assert result.min() >= 0.0
    assert result.max() <= 100.0


def test_postprocess_round() -> None:
    preds = np.array([1.12345, 2.98765])
    result = postprocess_predictions(preds, round_digits=2)
    assert result[0] == pytest.approx(1.12)
    assert result[1] == pytest.approx(2.99)


def test_postprocess_expm1() -> None:
    preds = np.log1p(np.array([1.0, 2.0, 3.0]))
    result = postprocess_predictions(preds, expm1=True)
    np.testing.assert_allclose(result, [1.0, 2.0, 3.0], rtol=1e-5)


def test_postprocess_does_not_modify_original() -> None:
    preds = np.array([1.0, 2.0, 3.0])
    original = preds.copy()
    postprocess_predictions(preds, clip_min=0.0, clip_max=2.0)
    np.testing.assert_array_equal(preds, original)


# ─────────────────────────────────────────────────────────────────
# CREATE SUBMISSION DF
# ─────────────────────────────────────────────────────────────────
def test_create_submission_df_basic(test_df_with_ids: pd.DataFrame) -> None:
    preds = np.ones(100)
    sub = create_submission_df(test_df_with_ids, preds, "sample_id", "entity_value")
    assert list(sub.columns) == ["sample_id", "entity_value"]
    assert len(sub) == 100


def test_create_submission_df_length_mismatch(test_df_with_ids: pd.DataFrame) -> None:
    with pytest.raises(ValueError, match="Length mismatch"):
        create_submission_df(test_df_with_ids, np.ones(50), "sample_id", "entity_value")


def test_create_submission_df_missing_id_col(test_df_with_ids: pd.DataFrame) -> None:
    with pytest.raises(ValueError, match="ID column"):
        create_submission_df(test_df_with_ids, np.ones(100), "nonexistent_id", "entity_value")


# ─────────────────────────────────────────────────────────────────
# VALIDATE SUBMISSION
# ─────────────────────────────────────────────────────────────────
def test_validate_submission_clean_passes(clean_submission: pd.DataFrame) -> None:
    ok, checks = validate_submission(
        clean_submission,
        id_column="sample_id",
        prediction_column="entity_value",
        expected_row_count=100,
    )
    assert ok is True
    assert all(c.passed for c in checks)


def test_validate_submission_detects_nan() -> None:
    df = pd.DataFrame({"id": [1, 2, 3], "pred": [1.0, np.nan, 3.0]})
    ok, checks = validate_submission(df, "id", "pred")
    assert ok is False
    nan_check = next(c for c in checks if "NaN" in c.name)
    assert not nan_check.passed


def test_validate_submission_detects_inf() -> None:
    df = pd.DataFrame({"id": [1, 2], "pred": [1.0, np.inf]})
    ok, checks = validate_submission(df, "id", "pred")
    assert ok is False


def test_validate_submission_detects_duplicate_ids() -> None:
    df = pd.DataFrame({"id": [1, 1, 2], "pred": [1.0, 2.0, 3.0]})
    ok, checks = validate_submission(df, "id", "pred")
    assert ok is False
    dup_check = next(c for c in checks if "duplicate" in c.name.lower())
    assert not dup_check.passed


def test_validate_submission_wrong_row_count() -> None:
    df = pd.DataFrame({"id": [1, 2], "pred": [1.0, 2.0]})
    ok, checks = validate_submission(df, "id", "pred", expected_row_count=10)
    assert ok is False


def test_validate_submission_range_check() -> None:
    df = pd.DataFrame({"id": [1, 2, 3], "pred": [0.5, 1.5, -0.1]})
    ok, checks = validate_submission(df, "id", "pred", clip_min=0.0, clip_max=1.0)
    assert ok is False


# ─────────────────────────────────────────────────────────────────
# WRITE SUBMISSION (end-to-end)
# ─────────────────────────────────────────────────────────────────
def test_write_submission_creates_file(
    clean_submission: pd.DataFrame, tmp_path: Path
) -> None:
    out_path = write_submission(
        clean_submission,
        id_column="sample_id",
        prediction_column="entity_value",
        output_dir=str(tmp_path),
        filename="test_sub.csv",
        expected_row_count=100,
    )
    assert out_path.exists()
    reloaded = pd.read_csv(out_path)
    assert len(reloaded) == 100


def test_write_submission_creates_sanity_report(
    clean_submission: pd.DataFrame, tmp_path: Path
) -> None:
    write_submission(
        clean_submission,
        id_column="sample_id",
        prediction_column="entity_value",
        output_dir=str(tmp_path),
    )
    assert (tmp_path / "sanity_report.txt").exists()


def test_write_submission_creates_metadata(
    clean_submission: pd.DataFrame, tmp_path: Path
) -> None:
    write_submission(
        clean_submission,
        id_column="sample_id",
        prediction_column="entity_value",
        output_dir=str(tmp_path),
    )
    assert (tmp_path / "submission_metadata.json").exists()


def test_write_submission_fails_with_nan(tmp_path: Path) -> None:
    df = pd.DataFrame({"id": [1, 2], "pred": [1.0, np.nan]})
    with pytest.raises(RuntimeError, match="FAILED"):
        write_submission(df, "id", "pred", output_dir=str(tmp_path))
