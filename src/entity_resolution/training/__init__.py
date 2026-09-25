"""
training/trainer.py — Training pipeline with entity-aware validation.

Key design decisions:
    1. Validation is entity-level (S1 entities), not pair-level
    2. Hard negative mining is built in
    3. OOF predictions are generated for threshold optimization
    4. All training artifacts are deterministic
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

from src.entity_resolution.core import EntityID, GroundTruth
from src.entity_resolution.blocking import CandidateSet
from src.entity_resolution.features import FeatureBuilder, ALL_FEATURES
from src.entity_resolution.models import PairScorer
from src.entity_resolution.validation import (
    compute_entity_f05, compute_pair_metrics,
    EntityF05Result, PairMetrics,
)
from src.entity_resolution.decision import EntityResolver, ResolverConfig


@dataclass
class TrainingResult:
    """Output of a training run."""
    model_name: str
    pair_metrics_train: Optional[PairMetrics]
    pair_metrics_val: Optional[PairMetrics]
    entity_f05_val: Optional[EntityF05Result]
    best_threshold: float
    best_f05: float
    oof_scores: Optional[pd.DataFrame]  # s1_id, candidate_id, score, label
    feature_importance: Optional[pd.DataFrame]
    training_time: float
    config: Dict[str, Any] = field(default_factory=dict)


def build_training_pairs(
    features_df: pd.DataFrame,
    ground_truth: GroundTruth,
    neg_ratio: int = 3,
    hard_neg_ratio: int = 2,
    seed: int = 42,
) -> Tuple[pd.DataFrame, np.ndarray]:
    """
    Build labeled training data from features + ground truth.

    For each S1 entity:
        - All true matches → label=1
        - Hard negatives (high similarity, wrong entity) → label=0
        - Random negatives → label=0

    Parameters
    ----------
    features_df : DataFrame with s1_id, candidate_id, and feature columns
    ground_truth : s1_id -> true match IDs
    neg_ratio : total negatives per positive
    hard_neg_ratio : hard negatives per positive (subset of neg_ratio)
    seed : random seed

    Returns
    -------
    (X_df, y) where X_df has feature columns and y is binary labels
    """
    rng = np.random.RandomState(seed)

    labeled_rows = []
    labels = []

    for s1_id, true_ids in ground_truth.items():
        s1_pairs = features_df[features_df["s1_id"] == s1_id]
        if len(s1_pairs) == 0:
            continue

        # Positive pairs
        for _, row in s1_pairs.iterrows():
            cand_id = row["candidate_id"]
            if cand_id in true_ids:
                labeled_rows.append(row)
                labels.append(1)

        # Negative pairs
        neg_pairs = s1_pairs[
            ~s1_pairs["candidate_id"].isin(true_ids)
        ]

        if len(neg_pairs) == 0:
            continue

        n_pos = len(true_ids)
        n_neg_wanted = max(n_pos * neg_ratio, 1)

        if len(neg_pairs) <= n_neg_wanted:
            for _, row in neg_pairs.iterrows():
                labeled_rows.append(row)
                labels.append(0)
        else:
            # Hard negatives: highest similarity negatives
            name_sim_col = "name_edit_sim"
            if name_sim_col in neg_pairs.columns:
                sorted_neg = neg_pairs.sort_values(
                    name_sim_col, ascending=False
                )
            else:
                sorted_neg = neg_pairs

            n_hard = min(n_pos * hard_neg_ratio, len(sorted_neg))
            for _, row in sorted_neg.head(n_hard).iterrows():
                labeled_rows.append(row)
                labels.append(0)

            # Random negatives for remainder
            remaining = sorted_neg.iloc[n_hard:]
            n_random = min(n_neg_wanted - n_hard, len(remaining))
            if n_random > 0:
                sampled = remaining.sample(n=n_random, random_state=rng)
                for _, row in sampled.iterrows():
                    labeled_rows.append(row)
                    labels.append(0)

    result_df = pd.DataFrame(labeled_rows)
    y = np.array(labels, dtype=np.int32)

    return result_df, y


def entity_aware_cv(
    features_df: pd.DataFrame,
    ground_truth: GroundTruth,
    scorer: PairScorer,
    n_folds: int = 5,
    seed: int = 42,
    neg_ratio: int = 3,
    hard_neg_ratio: int = 2,
) -> TrainingResult:
    """
    Entity-aware cross-validation.

    Splits on S1 entities to prevent leakage.
    Simulates the full inference pipeline per fold.
    """
    start_time = time.time()

    s1_ids = list(ground_truth.keys())
    s1_ids_in_features = set(features_df["s1_id"].unique())
    s1_ids = [sid for sid in s1_ids if sid in s1_ids_in_features]

    if len(s1_ids) < n_folds:
        n_folds = max(2, len(s1_ids))

    # Entity-level group fold
    groups = np.array([s1_ids.index(sid) % n_folds if sid in s1_ids
                       else 0 for sid in s1_ids])
    fold_assignments = {}
    fold_size = len(s1_ids) // n_folds
    rng = np.random.RandomState(seed)
    shuffled = list(s1_ids)
    rng.shuffle(shuffled)

    for i, sid in enumerate(shuffled):
        fold_assignments[sid] = i % n_folds

    oof_records = []
    fold_f05s = []

    feat_cols = [c for c in features_df.columns
                 if c not in ("s1_id", "candidate_id")]

    for fold in range(n_folds):
        val_s1 = {sid for sid, f in fold_assignments.items() if f == fold}
        train_s1 = {sid for sid, f in fold_assignments.items() if f != fold}

        # Build train/val pair datasets
        train_features = features_df[features_df["s1_id"].isin(train_s1)]
        val_features = features_df[features_df["s1_id"].isin(val_s1)]

        train_gt = {sid: ground_truth[sid] for sid in train_s1
                    if sid in ground_truth}
        val_gt = {sid: ground_truth[sid] for sid in val_s1
                  if sid in ground_truth}

        # Build labeled training data
        train_df, y_train = build_training_pairs(
            train_features, train_gt,
            neg_ratio=neg_ratio, hard_neg_ratio=hard_neg_ratio,
            seed=seed + fold,
        )

        if len(train_df) == 0 or len(val_features) == 0:
            continue

        X_train = train_df[feat_cols].values.astype(np.float32)

        # Build labeled validation data
        val_df, y_val = build_training_pairs(
            val_features, val_gt,
            neg_ratio=neg_ratio, hard_neg_ratio=hard_neg_ratio,
            seed=seed + fold + 100,
        )

        if len(val_df) == 0:
            continue

        X_val = val_df[feat_cols].values.astype(np.float32)

        # Train
        scorer.fit(
            X_train, y_train,
            eval_set=(X_val, y_val),
            early_stopping_rounds=50,
        )

        # Score all validation candidates
        X_val_all = val_features[feat_cols].values.astype(np.float32)
        val_scores = scorer.predict_proba(X_val_all)

        # Store OOF predictions
        for idx, (_, row) in enumerate(val_features.iterrows()):
            s1_id = row["s1_id"]
            cand_id = row["candidate_id"]
            true_ids = ground_truth.get(s1_id, set())
            oof_records.append({
                "s1_id": s1_id,
                "candidate_id": cand_id,
                "score": float(val_scores[idx]),
                "label": 1 if cand_id in true_ids else 0,
                "fold": fold,
            })

        # Entity-level evaluation for this fold
        from src.entity_resolution.core import ScoredCandidates
        scored: ScoredCandidates = {}
        for idx, (_, row) in enumerate(val_features.iterrows()):
            s1_id = row["s1_id"]
            cand_id = row["candidate_id"]
            scored.setdefault(s1_id, []).append(
                (cand_id, float(val_scores[idx]))
            )

        resolver = EntityResolver()
        best_thresh, best_f05 = resolver.optimize_threshold(
            scored, val_gt
        )
        fold_f05s.append(best_f05)

    # Aggregate results
    oof_df = pd.DataFrame(oof_records) if oof_records else None
    training_time = time.time() - start_time

    # Overall OOF metrics
    if oof_df is not None and len(oof_df) > 0:
        pair_metrics = compute_pair_metrics(
            oof_df["label"].values,
            (oof_df["score"].values >= 0.5).astype(int),
        )
    else:
        pair_metrics = None

    mean_f05 = float(np.mean(fold_f05s)) if fold_f05s else 0.0

    # Feature importance from last fold
    feat_imp = None
    if hasattr(scorer, "feature_importance"):
        try:
            feat_imp = scorer.feature_importance(feat_cols)
        except Exception:
            pass

    return TrainingResult(
        model_name=scorer.name,
        pair_metrics_train=None,
        pair_metrics_val=pair_metrics,
        entity_f05_val=None,
        best_threshold=0.5,
        best_f05=mean_f05,
        oof_scores=oof_df,
        feature_importance=feat_imp,
        training_time=training_time,
    )
