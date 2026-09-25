"""
core/types.py — Fundamental type definitions for Entity Resolution.

All type aliases and enums used across the ER engine live here.
No business logic. No imports beyond stdlib + typing.
"""

from __future__ import annotations

from enum import IntEnum, auto
from typing import Dict, FrozenSet, List, Mapping, Set, Tuple

# Entity ID is always a string like "S1-00001"
EntityID = str

# Source identifier: 1, 2, or 3
class Source(IntEnum):
    S1 = 1
    S2 = 2
    S3 = 3

    @classmethod
    def from_id(cls, entity_id: str) -> "Source":
        """Extract source from entity ID prefix."""
        prefix = entity_id[:2].upper()
        if prefix == "S1":
            return cls.S1
        elif prefix == "S2":
            return cls.S2
        elif prefix == "S3":
            return cls.S3
        raise ValueError(f"Cannot determine source from entity_id='{entity_id}'")


# Candidate pair: (s1_id, candidate_id)
CandidatePair = Tuple[EntityID, EntityID]

# Ground truth mapping: s1_id -> set of matched entity IDs
GroundTruth = Dict[EntityID, Set[EntityID]]

# Prediction mapping: s1_id -> set of predicted entity IDs
Predictions = Dict[EntityID, Set[EntityID]]

# Candidate mapping: s1_id -> set of candidate entity IDs
CandidateMap = Dict[EntityID, Set[EntityID]]

# Score for a single pair
PairScore = float

# Scored candidates: s1_id -> list of (candidate_id, score)
ScoredCandidates = Dict[EntityID, List[Tuple[EntityID, PairScore]]]
