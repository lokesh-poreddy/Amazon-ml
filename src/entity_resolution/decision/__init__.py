"""
decision/resolver.py — Entity-level decision engine.

Converts pair-level scores into entity-level match sets.

Handles:
    - Zero matches (singleton)
    - Single match
    - Multiple matches

Architecture:
    PairScores → Threshold → Ranking → Consistency → Final Set

The resolver is precision-focused because F0.5 penalizes false merges.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

import numpy as np

from src.entity_resolution.core import (
    EntityID, Predictions, ScoredCandidates
)


@dataclass
class ResolverConfig:
    """Configuration for the entity-level resolver."""
    # Absolute threshold: score must exceed this to be considered a match
    match_threshold: float = 0.5
    # Minimum margin between best score and threshold
    min_margin: float = 0.0
    # Maximum number of matches per S1 entity (0 = unlimited)
    max_matches: int = 0
    # If True, apply conservative singleton detection
    conservative_singletons: bool = True
    # Score gap between best and second-best to consider multi-match
    multi_match_gap: float = 0.1
    # Minimum score for multi-match candidates
    multi_match_threshold: float = 0.6


class EntityResolver:
    """
    Entity-level decision engine.

    Converts scored candidate pairs into final match sets.

    The resolver does NOT simply argmax. It evaluates:
        1. Absolute confidence
        2. Score margin
        3. Number of strong candidates
        4. Evidence strength
    """

    def __init__(self, config: Optional[ResolverConfig] = None):
        self.config = config or ResolverConfig()

    def resolve(
        self,
        scored_candidates: ScoredCandidates,
        all_s1_ids: Optional[List[EntityID]] = None,
    ) -> Predictions:
        """
        Produce final match sets for every S1 entity.

        Parameters
        ----------
        scored_candidates : dict[s1_id -> list of (cand_id, score)]
        all_s1_ids : all S1 entity IDs (to ensure singletons are included)

        Returns
        -------
        Predictions: dict[s1_id -> set of matched IDs]
        """
        predictions: Predictions = {}

        # Include all S1 entities, even those with no candidates
        if all_s1_ids:
            for s1_id in all_s1_ids:
                predictions[s1_id] = set()

        for s1_id, candidates in scored_candidates.items():
            if not candidates:
                predictions[s1_id] = set()
                continue

            # Sort by score descending
            sorted_cands = sorted(candidates, key=lambda x: x[1], reverse=True)

            matches = self._resolve_entity(sorted_cands)
            predictions[s1_id] = matches

        return predictions

    def _resolve_entity(
        self,
        sorted_candidates: List[Tuple[EntityID, float]],
    ) -> Set[EntityID]:
        """Resolve matches for a single S1 entity."""
        if not sorted_candidates:
            return set()

        cfg = self.config
        matches: Set[EntityID] = set()

        best_id, best_score = sorted_candidates[0]

        # Check if best candidate meets threshold
        if best_score < cfg.match_threshold:
            return set()  # Singleton

        # First match
        matches.add(best_id)

        # Check for multi-match
        if len(sorted_candidates) > 1:
            for cand_id, score in sorted_candidates[1:]:
                if score < cfg.match_threshold:
                    break

                # For multi-match: candidate must be independently strong
                if score >= cfg.multi_match_threshold:
                    # Check gap from best
                    gap = best_score - score
                    if gap <= cfg.multi_match_gap or score >= cfg.match_threshold:
                        matches.add(cand_id)
                    else:
                        break
                else:
                    break

        # Apply max_matches cap
        if cfg.max_matches > 0 and len(matches) > cfg.max_matches:
            # Keep top-scored
            top = sorted_candidates[:cfg.max_matches]
            matches = {cid for cid, _ in top}

        return matches

    def optimize_threshold(
        self,
        scored_candidates: ScoredCandidates,
        truth: Dict[EntityID, Set[EntityID]],
        thresholds: Optional[List[float]] = None,
    ) -> Tuple[float, float]:
        """
        Find optimal threshold by grid search on entity-level F0.5.

        Returns (best_threshold, best_f05).
        """
        from src.entity_resolution.validation import compute_entity_f05

        if thresholds is None:
            thresholds = [
                i / 100.0 for i in range(10, 96, 1)
            ]

        best_threshold = 0.5
        best_f05 = 0.0

        for thresh in thresholds:
            cfg = ResolverConfig(
                match_threshold=thresh,
                multi_match_threshold=max(thresh, 0.5),
                multi_match_gap=self.config.multi_match_gap,
                max_matches=self.config.max_matches,
            )
            resolver = EntityResolver(cfg)
            preds = resolver.resolve(scored_candidates, list(truth.keys()))
            result = compute_entity_f05(truth, preds)

            if result.macro_f05 > best_f05:
                best_f05 = result.macro_f05
                best_threshold = thresh

        return best_threshold, best_f05
