"""
tests/test_schema.py
Amazon ML Challenge 2026 — Schema audit tests.
"""

import pandas as pd
import numpy as np
import pytest

from src.schema import (
    audit_dataset,
    compare_train_test,
    NUMERIC,
    CATEGORICAL,
    TEXT,
    IDENTIFIER,
    CONSTANT,
    URL,
    IMAGE_PATH,
)


# ─────────────────────────────────────────────────────────────────
# FIXTURES
# ─────────────────────────────────────────────────────────────────
@pytest.fixture
def mixed_df() -> pd.DataFrame:
    return pd.DataFrame({
        "id":          [f"ID_{i:04d}" for i in range(100)],
        "price":       np.random.rand(100) * 1000,
        "category":    np.random.choice(["A", "B", "C"], 100),
        "description": [f"This is a product description with some text words number {i}" for i in range(100)],
        "image_url":   [f"https://img.example.com/product_{i}.jpg" for i in range(100)],
        "constant_col": [42] * 100,
    })


@pytest.fixture
def train_df() -> pd.DataFrame:
    return pd.DataFrame({
        "id":     range(200),
        "feat_a": np.random.rand(200),
        "feat_b": np.random.choice(["X", "Y"], 200),
        "target": np.random.rand(200),
    })


@pytest.fixture
def test_df() -> pd.DataFrame:
    return pd.DataFrame({
        "id":     range(50),
        "feat_a": np.random.rand(50),
        "feat_b": np.random.choice(["X", "Y", "Z"], 50),  # extra category "Z"
    })


# ─────────────────────────────────────────────────────────────────
# AUDIT DATASET
# ─────────────────────────────────────────────────────────────────
def test_audit_shape(mixed_df: pd.DataFrame) -> None:
    report = audit_dataset(mixed_df)
    assert report.n_rows == 100
    assert report.n_cols == len(mixed_df.columns)


def test_audit_detects_constant(mixed_df: pd.DataFrame) -> None:
    report = audit_dataset(mixed_df)
    assert "constant_col" in report.constant_cols


def test_audit_detects_numeric(mixed_df: pd.DataFrame) -> None:
    report = audit_dataset(mixed_df)
    assert "price" in report.numeric_cols


def test_audit_detects_text(mixed_df: pd.DataFrame) -> None:
    report = audit_dataset(mixed_df)
    assert "description" in report.text_cols


def test_audit_detects_url_or_image(mixed_df: pd.DataFrame) -> None:
    report = audit_dataset(mixed_df)
    # image_url should be classified as URL or IMAGE_PATH
    url_and_img = report.url_cols + report.image_cols
    assert "image_url" in url_and_img


def test_audit_no_duplicate_rows(mixed_df: pd.DataFrame) -> None:
    report = audit_dataset(mixed_df)
    assert report.duplicate_rows == 0


def test_audit_detects_duplicate_rows() -> None:
    df = pd.DataFrame({"a": [1, 2, 1], "b": ["x", "y", "x"]})
    report = audit_dataset(df)
    assert report.duplicate_rows == 1


# ─────────────────────────────────────────────────────────────────
# TRAIN / TEST COMPARISON
# ─────────────────────────────────────────────────────────────────
def test_comparison_detects_target_only_in_train(
    train_df: pd.DataFrame, test_df: pd.DataFrame
) -> None:
    cmp = compare_train_test(train_df, test_df, target_column="target")
    assert "target" in cmp.train_only_cols


def test_comparison_detects_category_mismatch(
    train_df: pd.DataFrame, test_df: pd.DataFrame
) -> None:
    cmp = compare_train_test(
        train_df, test_df,
        target_column="target",
        categorical_cols=["feat_b"],
    )
    assert "feat_b" in cmp.category_mismatches
    # "Z" is test-only
    assert "Z" in cmp.category_mismatches["feat_b"]["test_only"]


def test_comparison_no_mismatch_clean_data() -> None:
    train = pd.DataFrame({"id": [1, 2], "x": [0.1, 0.2], "target": [1.0, 2.0]})
    test  = pd.DataFrame({"id": [3, 4], "x": [0.3, 0.4]})
    cmp = compare_train_test(train, test, target_column="target")
    assert "target" in cmp.train_only_cols
    assert len(cmp.test_only_cols) == 0
    assert len(cmp.dtype_mismatches) == 0
