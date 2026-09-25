"""
validation/metrics.py — Entity-level F0.5 evaluator + blocking metrics.

This is the OFFICIAL competition metric implementation.

F0.5 = (1.25 * P * R) / (0.25 * P + R)

Calculated PER Source 1 entity, then MACRO-AVERAGED.

Singletons: if truth={} and pred={}, F0.5 = 1.0.
            if truth={} and pred={S2-X}, F0.5 = 0.0.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

import numpy as np

from src.entity_resolution.core import EntityID, GroundTruth, Predictions, CandidateMap


@dataclass(slots=True)
class EntityF05Result:
    """Complete entity-level F0.5 evaluation result."""
    macro_f05: float
    macro_precision: float
    macro_recall: float
    n_entities: int
    n_singletons_true: int
    n_singletons_pred: int
    n_singletons_correct: int
    n_matched_true: int
    n_matched_pred: int
    false_merges: int        # FP: predicted match when truth is empty
    missed_matches: int      # FN: missed true match
    per_entity_f05: Dict[EntityID, float] = field(default_factory=dict)
    per_entity_precision: Dict[EntityID, float] = field(default_factory=dict)
    per_entity_recall: Dict[EntityID, float] = field(default_factory=dict)


def compute_entity_f05(
    truth: GroundTruth,
    predictions: Predictions,
) -> EntityF05Result:
    """
    Compute macro-averaged entity-level F0.5.

    Parameters
    ----------
    truth : dict[s1_id -> set of matched IDs]
    predictions : dict[s1_id -> set of predicted IDs]

    Returns
    -------
    EntityF05Result with all metrics
    """
    all_s1 = set(truth.keys())
    if not all_s1:
        return EntityF05Result(
            macro_f05=0.0, macro_precision=0.0, macro_recall=0.0,
            n_entities=0, n_singletons_true=0, n_singletons_pred=0,
            n_singletons_correct=0, n_matched_true=0, n_matched_pred=0,
            false_merges=0, missed_matches=0,
        )

    per_f05: Dict[EntityID, float] = {}
    per_prec: Dict[EntityID, float] = {}
    per_rec: Dict[EntityID, float] = {}

    n_singletons_true = 0
    n_singletons_pred = 0
    n_singletons_correct = 0
    n_matched_true = 0
    n_matched_pred = 0
    false_merges = 0
    missed_matches = 0

    for s1_id in all_s1:
        true_set = truth.get(s1_id, set())
        pred_set = predictions.get(s1_id, set())

        is_true_singleton = len(true_set) == 0
        is_pred_singleton = len(pred_set) == 0

        if is_true_singleton:
            n_singletons_true += 1
        else:
            n_matched_true += 1

        if is_pred_singleton:
            n_singletons_pred += 1
        else:
            n_matched_pred += 1

        # Singleton case: truth empty, pred empty → F0.5 = 1.0
        if is_true_singleton and is_pred_singleton:
            per_f05[s1_id] = 1.0
            per_prec[s1_id] = 1.0
            per_rec[s1_id] = 1.0
            n_singletons_correct += 1
            continue

        # truth empty, pred non-empty → F0.5 = 0.0 (false merge)
        if is_true_singleton and not is_pred_singleton:
            per_f05[s1_id] = 0.0
            per_prec[s1_id] = 0.0
            per_rec[s1_id] = 0.0
            false_merges += len(pred_set)
            continue

        # truth non-empty, pred empty → F0.5 = 0.0 (missed)
        if not is_true_singleton and is_pred_singleton:
            per_f05[s1_id] = 0.0
            per_prec[s1_id] = 0.0
            per_rec[s1_id] = 0.0
            missed_matches += len(true_set)
            continue

        # Both non-empty: compute TP/FP/FN
        tp = len(true_set & pred_set)
        fp = len(pred_set - true_set)
        fn = len(true_set - pred_set)

        false_merges += fp
        missed_matches += fn

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0

        if precision + recall == 0:
            f05 = 0.0
        else:
            f05 = (1.25 * precision * recall) / (0.25 * precision + recall)

        per_f05[s1_id] = f05
        per_prec[s1_id] = precision
        per_rec[s1_id] = recall

    macro_f05 = float(np.mean(list(per_f05.values())))
    macro_precision = float(np.mean(list(per_prec.values())))
    macro_recall = float(np.mean(list(per_rec.values())))

    return EntityF05Result(
        macro_f05=macro_f05,
        macro_precision=macro_precision,
        macro_recall=macro_recall,
        n_entities=len(all_s1),
        n_singletons_true=n_singletons_true,
        n_singletons_pred=n_singletons_pred,
        n_singletons_correct=n_singletons_correct,
        n_matched_true=n_matched_true,
        n_matched_pred=n_matched_pred,
        false_merges=false_merges,
        missed_matches=missed_matches,
        per_entity_f05=per_f05,
        per_entity_precision=per_prec,
        per_entity_recall=per_rec,
    )


# ──────────────────────────────────────────────────────────────
# BLOCKING METRICS
# ──────────────────────────────────────────────────────────────

@dataclass(slots=True)
class BlockingMetrics:
    """Blocking recall and reduction ratio."""
    candidate_recall: float        # fraction of true matches in candidates
    total_true_matches: int
    true_matches_found: int
    true_matches_missed: int
    total_candidates: int
    total_possible_pairs: int
    reduction_ratio: float
    missed_pairs: List[Tuple[EntityID, EntityID]] = field(default_factory=list)


def compute_blocking_metrics(
    truth: GroundTruth,
    candidate_map: CandidateMap,
    n_s2: int,
    n_s3: int,
) -> BlockingMetrics:
    """
    Evaluate blocking quality.

    Parameters
    ----------
    truth : s1_id -> true match IDs
    candidate_map : s1_id -> candidate IDs
    n_s2, n_s3 : total number of S2 and S3 records
    """
    total_true = 0
    found = 0
    missed_pairs: List[Tuple[EntityID, EntityID]] = []

    for s1_id, true_ids in truth.items():
        for true_id in true_ids:
            total_true += 1
            candidates = candidate_map.get(s1_id, set())
            if true_id in candidates:
                found += 1
            else:
                missed_pairs.append((s1_id, true_id))

    recall = found / total_true if total_true > 0 else 1.0

    # Total candidates generated
    total_candidates = sum(len(cands) for cands in candidate_map.values())

    # Total possible pairs (S1 × (S2 + S3))
    n_s1 = len(truth)
    total_possible = n_s1 * (n_s2 + n_s3)
    reduction = 1.0 - (total_candidates / total_possible) if total_possible > 0 else 0.0

    return BlockingMetrics(
        candidate_recall=recall,
        total_true_matches=total_true,
        true_matches_found=found,
        true_matches_missed=total_true - found,
        total_candidates=total_candidates,
        total_possible_pairs=total_possible,
        reduction_ratio=reduction,
        missed_pairs=missed_pairs[:100],  # Cap for memory
    )


# ──────────────────────────────────────────────────────────────
# PAIR-LEVEL METRICS (supporting, not primary)
# ──────────────────────────────────────────────────────────────

@dataclass(slots=True)
class PairMetrics:
    """Pair-level classification metrics."""
    accuracy: float
    precision: float
    recall: float
    f1: float
    f05: float
    n_positive: int
    n_negative: int
    n_total: int


def compute_pair_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> PairMetrics:
    """Compute pair-level binary classification metrics."""
    y_true = np.asarray(y_true, dtype=int).ravel()
    y_pred = np.asarray(y_pred, dtype=int).ravel()

    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))

    n = len(y_true)
    accuracy = (tp + tn) / n if n > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0

    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    f05 = (1.25 * precision * recall) / (0.25 * precision + recall) if (0.25 * precision + recall) > 0 else 0.0

    return PairMetrics(
        accuracy=accuracy,
        precision=precision,
        recall=recall,
        f1=f1,
        f05=f05,
        n_positive=int(np.sum(y_true == 1)),
        n_negative=int(np.sum(y_true == 0)),
        n_total=n,
    )
