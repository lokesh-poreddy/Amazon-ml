import pytest
import numpy as np
from src.ensemble import OOFCandidate, score_weighted_selection, rank_avg
from src.metrics import get_metric

def test_rank_avg_without_yref():
    preds = [np.array([0.1, 0.4, 0.9]), np.array([0.2, 0.3, 0.8])]
    # Rankdata gives 1, 2, 3 -> divided by 3: 1/3, 2/3, 1
    avg = rank_avg(preds)
    assert np.all((avg >= 0) & (avg <= 1.0))

def test_rank_avg_with_yref():
    preds = [np.array([0.1, 0.4, 0.9]), np.array([0.2, 0.3, 0.8])]
    y_ref = np.array([10, 20, 30])
    avg = rank_avg(preds, y_ref=y_ref)
    assert np.all((avg >= 10) & (avg <= 30))

def test_score_weighted_selection():
    metric = get_metric("rmse")
    y_true = np.array([1, 2, 3])
    
    # Candidate 1: Perfect
    c1 = OOFCandidate("m1", np.array([1.0, 2.0, 3.0]), oof_score=0.0)
    # Candidate 2: Identical (high corr)
    c2 = OOFCandidate("m2", np.array([1.01, 2.01, 3.01]), oof_score=0.01)
    # Candidate 3: Divergent but okay
    c3 = OOFCandidate("m3", np.array([1.5, 1.5, 3.5]), oof_score=0.5)

    candidates = [c1, c2, c3]
    selected = score_weighted_selection(candidates, metric, y_true, correlation_threshold=0.98)
    
    # c1 selected first.
    # c2 rejected (corr > 0.98).
    # c3 checked: blend with c1 doesn't improve over 0.0, so c3 rejected.
    assert len(selected) == 1
    assert selected[0].name == "m1"
