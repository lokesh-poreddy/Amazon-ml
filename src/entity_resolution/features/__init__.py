"""
features/builder.py — Pair feature engine.

For every candidate pair (S1, Candidate), constructs a feature vector
with multiple independent views:

    NAME FEATURES          ~15 features
    ADDRESS FEATURES       ~12 features
    COUNTRY FEATURES       ~3 features
    CROSS-FIELD FEATURES   ~6 features
    PROVENANCE FEATURES    ~5 features
    MISSINGNESS FEATURES   ~4 features
    RARITY FEATURES        ~4 features

Total: ~49 features per pair.

Feature names are deterministic and documented.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Set, Tuple

import numpy as np
import pandas as pd

from src.entity_resolution.core import EntityID, Source
from src.entity_resolution.blocking import CandidateRecord, CandidateSet
from src.entity_resolution.similarity import (
    normalized_levenshtein,
    jaro_winkler_similarity,
    token_jaccard,
    token_overlap_coeff,
    token_dice,
    token_containment,
    char_ngram_similarity,
    numeric_token_overlap,
    numeric_containment,
    exact_match,
    prefix_match,
    containment_match,
    length_ratio,
    token_count_diff,
    length_diff,
)


# ──────────────────────────────────────────────────────────────
# FEATURE NAMES — stable ordering for reproducibility
# ──────────────────────────────────────────────────────────────

NAME_FEATURES = [
    "name_exact",
    "name_norm_exact",
    "name_compact_exact",
    "name_edit_sim",
    "name_jaro_winkler",
    "name_token_jaccard",
    "name_token_overlap",
    "name_token_dice",
    "name_token_containment",
    "name_char3_sim",
    "name_char4_sim",
    "name_prefix_match",
    "name_containment",
    "name_length_ratio",
    "name_length_diff",
    "name_token_count_diff",
]

ADDRESS_FEATURES = [
    "addr_exact",
    "addr_norm_exact",
    "addr_edit_sim",
    "addr_token_jaccard",
    "addr_token_overlap",
    "addr_token_containment",
    "addr_char3_sim",
    "addr_numeric_overlap",
    "addr_numeric_containment",
    "addr_postal_match",
    "addr_length_ratio",
    "addr_length_diff",
    "addr_token_count_diff",
]

COUNTRY_FEATURES = [
    "country_exact",
    "country_norm_exact",
    "country_either_missing",
]

CROSS_FEATURES = [
    "name_strong_addr_weak",
    "name_weak_addr_strong",
    "name_strong_addr_strong",
    "name_addr_consistency",
    "postal_name_agreement",
    "evidence_strength",
]

PROVENANCE_FEATURES = [
    "block_count",
    "block_exact_name",
    "block_token_name",
    "block_rare_token",
    "block_char_ngram",
    "block_tfidf",
    "block_fuzzy",
    "block_postal",
    "block_address",
    "block_country",
]

MISSINGNESS_FEATURES = [
    "s1_name_missing",
    "cand_name_missing",
    "s1_addr_missing",
    "cand_addr_missing",
    "s1_country_missing",
    "cand_country_missing",
]

RARITY_FEATURES = [
    "s1_name_token_count",
    "cand_name_token_count",
    "s1_addr_token_count",
    "cand_addr_token_count",
]

ALL_FEATURES = (
    NAME_FEATURES + ADDRESS_FEATURES + COUNTRY_FEATURES +
    CROSS_FEATURES + PROVENANCE_FEATURES + MISSINGNESS_FEATURES +
    RARITY_FEATURES
)


class FeatureBuilder:
    """
    Construct feature vectors for candidate pairs.

    Usage:
        builder = FeatureBuilder(record_store_df)
        features_df = builder.build(candidate_set, s1_ids)
    """

    def __init__(self, records_df: pd.DataFrame):
        """
        Parameters
        ----------
        records_df : DataFrame with NORM_COLS, containing all sources.
        """
        # Build fast lookup: entity_id -> row dict
        self._lookup: Dict[EntityID, Dict] = {}
        for _, row in records_df.iterrows():
            eid = str(row["entity_id"])
            self._lookup[eid] = {
                "name_raw": str(row.get("name_raw", "")),
                "name_norm": str(row.get("name_norm", "")),
                "name_compact": str(row.get("name_compact", "")),
                "name_tokens": row.get("name_tokens", ()),
                "address_raw": str(row.get("address_raw", "")),
                "address_norm": str(row.get("address_norm", "")),
                "address_tokens": row.get("address_tokens", ()),
                "postal": str(row.get("postal", "")),
                "numeric_tokens": row.get("numeric_tokens", ()),
                "country_raw": str(row.get("country_raw", "")),
                "country_norm": str(row.get("country_norm", "")),
            }

    def _get(self, eid: EntityID) -> Dict:
        return self._lookup.get(eid, {
            "name_raw": "", "name_norm": "", "name_compact": "",
            "name_tokens": (), "address_raw": "", "address_norm": "",
            "address_tokens": (), "postal": "", "numeric_tokens": (),
            "country_raw": "", "country_norm": "",
        })

    def compute_pair_features(
        self,
        s1_id: EntityID,
        cand_id: EntityID,
        provenance: Optional[CandidateRecord] = None,
    ) -> Dict[str, float]:
        """Compute all features for a single pair."""
        s1 = self._get(s1_id)
        cand = self._get(cand_id)
        f: Dict[str, float] = {}

        # ── NAME FEATURES ────────────────────────────────────
        f["name_exact"] = exact_match(s1["name_raw"], cand["name_raw"])
        f["name_norm_exact"] = exact_match(s1["name_norm"], cand["name_norm"])
        f["name_compact_exact"] = exact_match(
            s1["name_compact"], cand["name_compact"]
        )
        f["name_edit_sim"] = normalized_levenshtein(
            s1["name_norm"], cand["name_norm"]
        )
        f["name_jaro_winkler"] = jaro_winkler_similarity(
            s1["name_norm"], cand["name_norm"]
        )
        f["name_token_jaccard"] = token_jaccard(
            s1["name_tokens"], cand["name_tokens"]
        )
        f["name_token_overlap"] = token_overlap_coeff(
            s1["name_tokens"], cand["name_tokens"]
        )
        f["name_token_dice"] = token_dice(
            s1["name_tokens"], cand["name_tokens"]
        )
        f["name_token_containment"] = token_containment(
            s1["name_tokens"], cand["name_tokens"]
        )
        f["name_char3_sim"] = char_ngram_similarity(
            s1["name_norm"], cand["name_norm"], n=3
        )
        f["name_char4_sim"] = char_ngram_similarity(
            s1["name_norm"], cand["name_norm"], n=4
        )
        f["name_prefix_match"] = prefix_match(
            s1["name_norm"], cand["name_norm"]
        )
        f["name_containment"] = containment_match(
            s1["name_norm"], cand["name_norm"]
        )
        f["name_length_ratio"] = length_ratio(
            s1["name_norm"], cand["name_norm"]
        )
        f["name_length_diff"] = float(length_diff(
            s1["name_norm"], cand["name_norm"]
        ))
        f["name_token_count_diff"] = float(token_count_diff(
            s1["name_tokens"], cand["name_tokens"]
        ))

        # ── ADDRESS FEATURES ─────────────────────────────────
        f["addr_exact"] = exact_match(s1["address_raw"], cand["address_raw"])
        f["addr_norm_exact"] = exact_match(
            s1["address_norm"], cand["address_norm"]
        )
        f["addr_edit_sim"] = normalized_levenshtein(
            s1["address_norm"], cand["address_norm"]
        )
        f["addr_token_jaccard"] = token_jaccard(
            s1["address_tokens"], cand["address_tokens"]
        )
        f["addr_token_overlap"] = token_overlap_coeff(
            s1["address_tokens"], cand["address_tokens"]
        )
        f["addr_token_containment"] = token_containment(
            s1["address_tokens"], cand["address_tokens"]
        )
        f["addr_char3_sim"] = char_ngram_similarity(
            s1["address_norm"], cand["address_norm"], n=3
        )
        f["addr_numeric_overlap"] = numeric_token_overlap(
            s1["numeric_tokens"], cand["numeric_tokens"]
        )
        f["addr_numeric_containment"] = numeric_containment(
            s1["numeric_tokens"], cand["numeric_tokens"]
        )
        f["addr_postal_match"] = (
            1.0 if s1["postal"] and cand["postal"] and
            s1["postal"] == cand["postal"] else 0.0
        )
        f["addr_length_ratio"] = length_ratio(
            s1["address_norm"], cand["address_norm"]
        )
        f["addr_length_diff"] = float(length_diff(
            s1["address_norm"], cand["address_norm"]
        ))
        f["addr_token_count_diff"] = float(token_count_diff(
            s1["address_tokens"], cand["address_tokens"]
        ))

        # ── COUNTRY FEATURES ─────────────────────────────────
        f["country_exact"] = exact_match(
            s1["country_raw"], cand["country_raw"]
        )
        f["country_norm_exact"] = exact_match(
            s1["country_norm"], cand["country_norm"]
        )
        f["country_either_missing"] = (
            1.0 if not s1["country_norm"] or not cand["country_norm"] else 0.0
        )

        # ── CROSS-FIELD FEATURES ─────────────────────────────
        name_sim = f["name_edit_sim"]
        addr_sim = f["addr_edit_sim"]
        f["name_strong_addr_weak"] = (
            1.0 if name_sim > 0.8 and addr_sim < 0.3 else 0.0
        )
        f["name_weak_addr_strong"] = (
            1.0 if name_sim < 0.5 and addr_sim > 0.7 else 0.0
        )
        f["name_strong_addr_strong"] = (
            1.0 if name_sim > 0.7 and addr_sim > 0.5 else 0.0
        )
        f["name_addr_consistency"] = (name_sim + addr_sim) / 2.0
        f["postal_name_agreement"] = (
            1.0 if f["addr_postal_match"] == 1.0 and name_sim > 0.5 else 0.0
        )
        f["evidence_strength"] = max(
            name_sim,
            addr_sim,
            f["name_token_jaccard"],
            f["addr_token_jaccard"],
        )

        # ── PROVENANCE FEATURES ──────────────────────────────
        if provenance is not None:
            f["block_count"] = float(provenance.block_count)
            blockers = provenance.blockers_hit
            f["block_exact_name"] = 1.0 if "exact_name" in blockers else 0.0
            f["block_token_name"] = 1.0 if "token_name" in blockers else 0.0
            f["block_rare_token"] = 1.0 if "rare_token" in blockers else 0.0
            f["block_char_ngram"] = 1.0 if "char_ngram" in blockers else 0.0
            f["block_tfidf"] = 1.0 if "tfidf_name" in blockers else 0.0
            f["block_fuzzy"] = 1.0 if "fuzzy_topk" in blockers else 0.0
            f["block_postal"] = 1.0 if "postal" in blockers else 0.0
            f["block_address"] = 1.0 if "address_token" in blockers else 0.0
            f["block_country"] = 1.0 if "country" in blockers else 0.0
        else:
            for feat in PROVENANCE_FEATURES:
                f[feat] = 0.0

        # ── MISSINGNESS FEATURES ─────────────────────────────
        f["s1_name_missing"] = 1.0 if not s1["name_norm"] else 0.0
        f["cand_name_missing"] = 1.0 if not cand["name_norm"] else 0.0
        f["s1_addr_missing"] = 1.0 if not s1["address_norm"] else 0.0
        f["cand_addr_missing"] = 1.0 if not cand["address_norm"] else 0.0
        f["s1_country_missing"] = 1.0 if not s1["country_norm"] else 0.0
        f["cand_country_missing"] = 1.0 if not cand["country_norm"] else 0.0

        # ── RARITY FEATURES ──────────────────────────────────
        f["s1_name_token_count"] = float(len(s1["name_tokens"]))
        f["cand_name_token_count"] = float(len(cand["name_tokens"]))
        f["s1_addr_token_count"] = float(len(s1["address_tokens"]))
        f["cand_addr_token_count"] = float(len(cand["address_tokens"]))

        return f

    def build(
        self,
        candidate_set: CandidateSet,
        s1_ids: Optional[List[EntityID]] = None,
    ) -> pd.DataFrame:
        """
        Build feature matrix for all candidate pairs.

        Returns DataFrame with columns:
            s1_id, candidate_id, <ALL_FEATURES>
        """
        rows = []
        pairs = candidate_set._pairs

        for (s1_id, cand_id), prov in pairs.items():
            if s1_ids is not None and s1_id not in set(s1_ids):
                continue
            features = self.compute_pair_features(s1_id, cand_id, prov)
            features["s1_id"] = s1_id
            features["candidate_id"] = cand_id
            rows.append(features)

        if not rows:
            # Return empty DataFrame with correct columns
            cols = ["s1_id", "candidate_id"] + ALL_FEATURES
            return pd.DataFrame(columns=cols)

        df = pd.DataFrame(rows)
        # Ensure column order
        id_cols = ["s1_id", "candidate_id"]
        feat_cols = [c for c in ALL_FEATURES if c in df.columns]
        return df[id_cols + feat_cols]

    def build_for_pair(
        self,
        s1_id: EntityID,
        cand_id: EntityID,
    ) -> np.ndarray:
        """Build feature vector for a single pair as numpy array."""
        features = self.compute_pair_features(s1_id, cand_id)
        return np.array([features.get(f, 0.0) for f in ALL_FEATURES],
                        dtype=np.float32)
