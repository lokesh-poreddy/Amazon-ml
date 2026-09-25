"""
blocking/base.py — Abstract blocker interface.

Every blocker implements generate() which takes reference (S1) records
and target (S2 or S3) records, and returns candidate pairs with provenance.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, FrozenSet, List, Optional, Set, Tuple

import pandas as pd

from src.entity_resolution.core import EntityID, Source


@dataclass(slots=True)
class CandidateRecord:
    """A single candidate pair with provenance."""
    s1_id: EntityID
    candidate_id: EntityID
    candidate_source: int  # 2 or 3
    blockers_hit: Set[str] = field(default_factory=set)
    block_scores: Dict[str, float] = field(default_factory=dict)

    @property
    def block_count(self) -> int:
        return len(self.blockers_hit)


class CandidateSet:
    """
    Collection of candidate pairs with provenance tracking.

    Internally stores pairs as a dict: (s1_id, candidate_id) -> CandidateRecord.
    Deduplicates automatically. When the same pair is found by multiple
    blockers, provenance is merged.
    """

    def __init__(self):
        self._pairs: Dict[Tuple[EntityID, EntityID], CandidateRecord] = {}

    def add(self, s1_id: EntityID, candidate_id: EntityID,
            candidate_source: int, blocker_name: str,
            score: float = 1.0) -> None:
        key = (s1_id, candidate_id)
        if key in self._pairs:
            rec = self._pairs[key]
            rec.blockers_hit.add(blocker_name)
            rec.block_scores[blocker_name] = score
        else:
            self._pairs[key] = CandidateRecord(
                s1_id=s1_id,
                candidate_id=candidate_id,
                candidate_source=candidate_source,
                blockers_hit={blocker_name},
                block_scores={blocker_name: score},
            )

    def merge(self, other: "CandidateSet") -> None:
        for key, rec in other._pairs.items():
            if key in self._pairs:
                self._pairs[key].blockers_hit |= rec.blockers_hit
                self._pairs[key].block_scores.update(rec.block_scores)
            else:
                self._pairs[key] = rec

    def get_candidates_for(self, s1_id: EntityID) -> List[CandidateRecord]:
        return [rec for key, rec in self._pairs.items() if key[0] == s1_id]

    def get_candidate_ids_for(self, s1_id: EntityID) -> Set[EntityID]:
        return {rec.candidate_id for key, rec in self._pairs.items()
                if key[0] == s1_id}

    def all_s1_ids(self) -> Set[EntityID]:
        return {key[0] for key in self._pairs}

    def __len__(self) -> int:
        return len(self._pairs)

    def __contains__(self, pair: Tuple[EntityID, EntityID]) -> bool:
        return pair in self._pairs

    def to_dataframe(self) -> pd.DataFrame:
        rows = []
        for (s1_id, cand_id), rec in self._pairs.items():
            rows.append({
                "s1_id": s1_id,
                "candidate_id": cand_id,
                "candidate_source": rec.candidate_source,
                "block_count": rec.block_count,
                "blockers": ",".join(sorted(rec.blockers_hit)),
            })
        return pd.DataFrame(rows) if rows else pd.DataFrame(
            columns=["s1_id", "candidate_id", "candidate_source",
                     "block_count", "blockers"]
        )

    def to_candidate_map(self) -> Dict[EntityID, Set[EntityID]]:
        result: Dict[EntityID, Set[EntityID]] = {}
        for (s1_id, cand_id) in self._pairs:
            result.setdefault(s1_id, set()).add(cand_id)
        return result


class Blocker(ABC):
    """Abstract base class for all blocking strategies."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Short identifier for this blocker."""
        ...

    @abstractmethod
    def generate(
        self,
        reference_df: pd.DataFrame,
        target_df: pd.DataFrame,
    ) -> CandidateSet:
        """
        Generate candidate pairs.

        Parameters
        ----------
        reference_df : S1 normalized records
        target_df : S2 or S3 normalized records

        Returns
        -------
        CandidateSet with provenance
        """
        ...
