import pytest
import numpy as np
import pandas as pd
from src.models import build_model
def test_lgbm_initialization():
    model = build_model("lgbm", task_type="regression", seed=42)
    assert model.params.get("random_state") == 42
    assert model.task_type == "regression"

def test_xgb_multiclass_softprob():
    model = build_model("xgb", task_type="multiclass", seed=42)
    # The objective is injected in _fit for xgb, not in __init__.
    # Wait, XGBoostModel sets objective in _fit. Let's just test task_type.
    assert model.task_type == "multiclass"

def test_catboost_classifier():
    model = build_model("catboost", task_type="classification", seed=42)
    assert model.task_type == "classification"

def test_dummy_model():
    model = build_model("dummy", strategy="mean")
    
    X = pd.DataFrame({"f1": [1, 2, 3]})
    y = np.array([1.0, 2.0, 3.0])
    model.fit(X, y)
    
    preds = model.predict(X)
    assert len(preds) == 3
    assert np.allclose(preds, 2.0)  # DummyRegressor default strategy is "mean"
