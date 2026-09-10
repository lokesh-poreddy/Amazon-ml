"""
tests/test_metrics.py
Amazon ML Challenge 2026 — Metric implementation tests.
"""

import numpy as np
import pytest

from src.metrics import (
    MetricSpec,
    evaluate,
    get_metric,
    list_metrics,
    mae,
    rmse,
    smape,
    r2,
    accuracy,
    f1_binary,
)


# ─────────────────────────────────────────────────────────────────
# SMAPE
# ─────────────────────────────────────────────────────────────────
def test_smape_perfect_predictions() -> None:
    y = np.array([1.0, 2.0, 3.0])
    assert smape(y, y) == pytest.approx(0.0, abs=1e-6)


def test_smape_known_value() -> None:
    # SMAPE(10, 20) = 200 * |10-20| / (10+20) = 200*10/30 ≈ 66.67
    result = smape(np.array([10.0]), np.array([20.0]))
    assert result == pytest.approx(200 * 10 / 30, abs=0.01)


def test_smape_range() -> None:
    y_true = np.array([1.0, 5.0, 100.0])
    y_pred = np.array([0.5, 10.0, 50.0])
    result = smape(y_true, y_pred)
    assert 0 <= result <= 200


# ─────────────────────────────────────────────────────────────────
# MAE
# ─────────────────────────────────────────────────────────────────
def test_mae_zero_error() -> None:
    y = np.array([1.0, 2.0, 3.0])
    assert mae(y, y) == pytest.approx(0.0)


def test_mae_known_value() -> None:
    assert mae(np.array([0.0, 0.0]), np.array([1.0, 3.0])) == pytest.approx(2.0)


# ─────────────────────────────────────────────────────────────────
# RMSE
# ─────────────────────────────────────────────────────────────────
def test_rmse_zero_error() -> None:
    y = np.array([1.0, 2.0, 3.0])
    assert rmse(y, y) == pytest.approx(0.0)


def test_rmse_known_value() -> None:
    # errors = [1, 1, 1], RMSE = 1
    y_true = np.array([0.0, 0.0, 0.0])
    y_pred = np.array([1.0, 1.0, 1.0])
    assert rmse(y_true, y_pred) == pytest.approx(1.0)


# ─────────────────────────────────────────────────────────────────
# R2
# ─────────────────────────────────────────────────────────────────
def test_r2_perfect() -> None:
    y = np.array([1.0, 2.0, 3.0, 4.0])
    assert r2(y, y) == pytest.approx(1.0)


def test_r2_baseline() -> None:
    y_true = np.array([1.0, 2.0, 3.0])
    # Predicting the mean is R2 = 0
    y_pred = np.full(3, np.mean(y_true))
    assert r2(y_true, y_pred) == pytest.approx(0.0, abs=1e-6)


# ─────────────────────────────────────────────────────────────────
# ACCURACY
# ─────────────────────────────────────────────────────────────────
def test_accuracy_perfect() -> None:
    y = np.array([0, 1, 0, 1])
    assert accuracy(y, y) == pytest.approx(1.0)


def test_accuracy_half() -> None:
    y_true = np.array([0, 0, 1, 1])
    y_pred = np.array([0, 1, 0, 1])
    assert accuracy(y_true, y_pred) == pytest.approx(0.5)


# ─────────────────────────────────────────────────────────────────
# METRIC REGISTRY
# ─────────────────────────────────────────────────────────────────
def test_get_metric_valid() -> None:
    spec = get_metric("smape")
    assert isinstance(spec, MetricSpec)
    assert spec.direction == "minimize"


def test_get_metric_maximize() -> None:
    spec = get_metric("r2")
    assert spec.direction == "maximize"


def test_get_metric_invalid() -> None:
    with pytest.raises(ValueError, match="Unknown metric"):
        get_metric("nonexistent_metric")


def test_list_metrics_nonempty() -> None:
    metrics = list_metrics()
    assert len(metrics) > 5
    assert "smape" in metrics
    assert "mae" in metrics


# ─────────────────────────────────────────────────────────────────
# EVALUATE
# ─────────────────────────────────────────────────────────────────
def test_evaluate_returns_result() -> None:
    y_true = np.array([1.0, 2.0, 3.0])
    y_pred = np.array([1.0, 2.0, 3.0])
    result = evaluate(y_true, y_pred, "smape")
    assert result.score == pytest.approx(0.0, abs=1e-6)
    assert result.n_samples == 3


def test_evaluate_length_mismatch() -> None:
    with pytest.raises(ValueError, match="Length mismatch"):
        evaluate(np.array([1.0, 2.0]), np.array([1.0]), "mae")


# ─────────────────────────────────────────────────────────────────
# MetricSpec.is_better
# ─────────────────────────────────────────────────────────────────
def test_metric_spec_is_better_minimize() -> None:
    spec = get_metric("smape")
    assert spec.is_better(0.1, 0.2) is True
    assert spec.is_better(0.5, 0.3) is False


def test_metric_spec_is_better_maximize() -> None:
    spec = get_metric("r2")
    assert spec.is_better(0.9, 0.8) is True
    assert spec.is_better(0.5, 0.7) is False
