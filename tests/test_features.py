"""
tests/test_features.py
Amazon ML Challenge 2026 — Feature engineering tests.
"""

import numpy as np
import pandas as pd
import pytest

from src.features import (
    categorical_features,
    create_features,
    numeric_features,
    text_structured_features,
    TfidfFeatureBuilder,
)


# ─────────────────────────────────────────────────────────────────
# FIXTURES
# ─────────────────────────────────────────────────────────────────
@pytest.fixture
def product_df() -> pd.DataFrame:
    return pd.DataFrame({
        "entity_name": [
            "voltage",
            "wattage",
            "item_weight",
            "maximum_weight_recommendation",
            "width",
        ],
        "product_name": [
            "Samsung TV 55 inch 4K 120Hz - Pack of 1",
            "Philips 100W LED Bulb B22 2700K warm white",
            "Apple iPhone 14 Pro 256GB - Weight 206g",
            "Bosch Drill 500W - Max Load 10kg",
            "IKEA Table 120cm x 60cm x 75cm oak finish",
        ],
        "brand": ["Samsung", "Philips", "Apple", "Bosch", "IKEA"],
        "price": [45000.0, 299.0, 79999.0, 3499.0, 5999.0],
        "category": ["electronics", "lighting", "mobile", "tools", "furniture"],
    })


# ─────────────────────────────────────────────────────────────────
# TEXT STRUCTURED FEATURES
# ─────────────────────────────────────────────────────────────────
def test_text_structured_basic_shape(product_df: pd.DataFrame) -> None:
    feats = text_structured_features(product_df, text_cols=["product_name"])
    assert len(feats) == len(product_df)
    assert len(feats.columns) > 5


def test_text_structured_char_count(product_df: pd.DataFrame) -> None:
    feats = text_structured_features(product_df, text_cols=["product_name"])
    col = [c for c in feats.columns if "char_count" in c]
    assert len(col) > 0
    assert (feats[col[0]] > 0).all()


def test_text_structured_detects_units(product_df: pd.DataFrame) -> None:
    feats = text_structured_features(product_df, text_cols=["product_name"])
    unit_col = [c for c in feats.columns if "has_unit" in c]
    assert len(unit_col) > 0
    # At least some products have unit mentions
    assert feats[unit_col[0]].sum() > 0


def test_text_structured_no_nan(product_df: pd.DataFrame) -> None:
    feats = text_structured_features(product_df, text_cols=["product_name"])
    assert not feats.isnull().values.any()


def test_text_structured_missing_col_warning(product_df: pd.DataFrame) -> None:
    # Should not raise — just warn and skip
    feats = text_structured_features(product_df, text_cols=["nonexistent_col"])
    assert feats.empty


# ─────────────────────────────────────────────────────────────────
# NUMERIC FEATURES
# ─────────────────────────────────────────────────────────────────
def test_numeric_features_passthrough(product_df: pd.DataFrame) -> None:
    feats = numeric_features(product_df, numeric_cols=["price"])
    assert len(feats) == len(product_df)
    assert any("price" in c for c in feats.columns)


def test_numeric_features_log_transform(product_df: pd.DataFrame) -> None:
    feats = numeric_features(product_df, numeric_cols=["price"], log_cols=["price"])
    log_col = [c for c in feats.columns if "log1p" in c]
    assert len(log_col) > 0
    assert (feats[log_col[0]] >= 0).all()


def test_numeric_features_ratio(product_df: pd.DataFrame) -> None:
    product_df = product_df.copy()
    product_df["cost"] = product_df["price"] * 0.6
    feats = numeric_features(
        product_df,
        numeric_cols=["price"],
        ratio_pairs=[("price", "cost")],
    )
    ratio_col = [c for c in feats.columns if "ratio" in c]
    assert len(ratio_col) > 0


# ─────────────────────────────────────────────────────────────────
# CATEGORICAL FEATURES
# ─────────────────────────────────────────────────────────────────
def test_categorical_features_freq_encoding(product_df: pd.DataFrame) -> None:
    feats = categorical_features(product_df, categorical_cols=["category"], train_df=product_df)
    freq_col = [c for c in feats.columns if "freq" in c]
    assert len(freq_col) > 0
    # All values should be in [0, 1]
    assert (feats[freq_col[0]] >= 0).all()
    assert (feats[freq_col[0]] <= 1).all()


def test_categorical_features_no_leakage_with_train_ref() -> None:
    """Verify that test categories not in train get 0 frequency."""
    train = pd.DataFrame({"cat": ["A", "A", "B", "B", "C"]})
    test  = pd.DataFrame({"cat": ["A", "D"]})  # D is unseen

    train_feats = categorical_features(train, ["cat"], train_df=train)
    test_feats  = categorical_features(test,  ["cat"], train_df=train)

    # Unseen category D should get frequency 0
    assert test_feats.iloc[1]["cat__freq_cat"] == pytest.approx(0.0)


# ─────────────────────────────────────────────────────────────────
# TFIDF FEATURE BUILDER
# ─────────────────────────────────────────────────────────────────
def test_tfidf_builder_fit_transform(product_df: pd.DataFrame) -> None:
    builder = TfidfFeatureBuilder(max_features=100)
    X = builder.fit_transform(product_df, text_cols=["product_name"])
    assert X.shape[0] == len(product_df)
    assert X.shape[1] <= 100


def test_tfidf_builder_fit_then_transform(product_df: pd.DataFrame) -> None:
    builder = TfidfFeatureBuilder(max_features=50)
    builder.fit(product_df, text_cols=["product_name"])
    X_train = builder.transform(product_df)
    X_val   = builder.transform(product_df.head(2))
    # Column count must be consistent
    assert X_train.shape[1] == X_val.shape[1]


# ─────────────────────────────────────────────────────────────────
# CREATE FEATURES (integration)
# ─────────────────────────────────────────────────────────────────
def test_create_features_integration(product_df: pd.DataFrame) -> None:
    feats = create_features(
        product_df,
        numeric_cols=["price"],
        categorical_cols=["category"],
        text_cols=["product_name"],
        train_df=product_df,
    )
    assert len(feats) == len(product_df)
    assert len(feats.columns) > 0
    assert not np.isinf(feats.select_dtypes(include=np.number).values).any()
