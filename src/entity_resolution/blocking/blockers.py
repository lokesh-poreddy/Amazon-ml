"""
blocking/blockers.py — Concrete blocking strategies.

Each blocker uses inverted indexes for O(1) average lookup.
No Cartesian joins. No nested loops over the full target set.

Implemented blockers:
    ExactNameBlocker        — exact normalized name match
    ExactCompactNameBlocker — exact compact (no-space) name match
    TokenNameBlocker        — shared name token inverted index
    RareTokenBlocker        — tokens with IDF above threshold
    CharNgramBlocker        — character n-gram TF-IDF retrieval
    PostalBlocker           — exact postal code match
    CountryBlocker          — exact country match
    AddressTokenBlocker     — shared address token inverted index
    NameCountryBlocker      — name + country compound key
    TfidfBlocker            — TF-IDF cosine retrieval for names
    FuzzyTopKBlocker        — RapidFuzz top-k retrieval
"""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd

from src.entity_resolution.blocking import Blocker, CandidateSet
from src.entity_resolution.core import EntityID, Source


# ──────────────────────────────────────────────────────────────
# UTILITY: build inverted index from a column
# ──────────────────────────────────────────────────────────────

def _build_inverted_index(
    df: pd.DataFrame,
    key_col: str,
    id_col: str = "entity_id",
) -> Dict[str, List[EntityID]]:
    """Build key -> list of entity_ids."""
    idx: Dict[str, List[EntityID]] = defaultdict(list)
    for eid, key in zip(df[id_col], df[key_col]):
        if key and str(key).strip():
            idx[str(key).strip()].append(str(eid))
    return dict(idx)


def _build_token_inverted_index(
    df: pd.DataFrame,
    tokens_col: str,
    id_col: str = "entity_id",
) -> Dict[str, List[EntityID]]:
    """Build token -> list of entity_ids from a column of token tuples."""
    idx: Dict[str, List[EntityID]] = defaultdict(list)
    for eid, tokens in zip(df[id_col], df[tokens_col]):
        if tokens:
            for tok in tokens:
                if tok:
                    idx[tok].append(str(eid))
    return dict(idx)


# ──────────────────────────────────────────────────────────────
# EXACT NAME BLOCKER
# ──────────────────────────────────────────────────────────────

class ExactNameBlocker(Blocker):
    """Block on exact normalized business name."""

    @property
    def name(self) -> str:
        return "exact_name"

    def generate(self, reference_df: pd.DataFrame,
                 target_df: pd.DataFrame) -> CandidateSet:
        candidates = CandidateSet()
        target_idx = _build_inverted_index(target_df, "name_norm")

        for _, row in reference_df.iterrows():
            key = str(row["name_norm"]).strip()
            if key and key in target_idx:
                for cand_id in target_idx[key]:
                    source = Source.from_id(cand_id).value
                    candidates.add(
                        str(row["entity_id"]), cand_id,
                        source, self.name, score=1.0
                    )
        return candidates


# ──────────────────────────────────────────────────────────────
# EXACT COMPACT NAME BLOCKER
# ──────────────────────────────────────────────────────────────

class ExactCompactNameBlocker(Blocker):
    """Block on name with all spaces removed — catches spacing differences."""

    @property
    def name(self) -> str:
        return "exact_compact_name"

    def generate(self, reference_df: pd.DataFrame,
                 target_df: pd.DataFrame) -> CandidateSet:
        candidates = CandidateSet()
        target_idx = _build_inverted_index(target_df, "name_compact")

        for _, row in reference_df.iterrows():
            key = str(row["name_compact"]).strip()
            if key and key in target_idx:
                for cand_id in target_idx[key]:
                    source = Source.from_id(cand_id).value
                    candidates.add(
                        str(row["entity_id"]), cand_id,
                        source, self.name, score=1.0
                    )
        return candidates


# ──────────────────────────────────────────────────────────────
# TOKEN NAME BLOCKER
# ──────────────────────────────────────────────────────────────

class TokenNameBlocker(Blocker):
    """
    Block on shared name tokens using an inverted index.

    A candidate is generated if the S1 record shares at least
    `min_shared` tokens with a target record.
    """

    def __init__(self, min_shared: int = 1):
        self.min_shared = min_shared

    @property
    def name(self) -> str:
        return "token_name"

    def generate(self, reference_df: pd.DataFrame,
                 target_df: pd.DataFrame) -> CandidateSet:
        candidates = CandidateSet()
        token_idx = _build_token_inverted_index(target_df, "name_tokens")

        # Common stopwords to skip
        stopwords = {
            "the", "and", "of", "in", "at", "to", "for", "a", "an",
            "private", "limited", "llc", "incorporated", "corporation",
            "company", "group", "international",
        }

        for _, row in reference_df.iterrows():
            s1_id = str(row["entity_id"])
            tokens = row["name_tokens"]
            if not tokens:
                continue

            # Count candidate hits
            cand_counts: Counter = Counter()
            for tok in tokens:
                if tok in stopwords or len(tok) <= 1:
                    continue
                if tok in token_idx:
                    for cand_id in token_idx[tok]:
                        cand_counts[cand_id] += 1

            for cand_id, count in cand_counts.items():
                if count >= self.min_shared:
                    source = Source.from_id(cand_id).value
                    # Score = fraction of reference tokens matched
                    n_meaningful = sum(
                        1 for t in tokens
                        if t not in stopwords and len(t) > 1
                    )
                    score = count / max(n_meaningful, 1)
                    candidates.add(s1_id, cand_id, source, self.name, score)

        return candidates


# ──────────────────────────────────────────────────────────────
# RARE TOKEN BLOCKER
# ──────────────────────────────────────────────────────────────

class RareTokenBlocker(Blocker):
    """
    Block on rare tokens (high IDF).

    Tokens appearing in fewer than `max_df` documents are considered rare.
    A match on a rare token is strong evidence.
    """

    def __init__(self, max_df: int = 10):
        self.max_df = max_df

    @property
    def name(self) -> str:
        return "rare_token"

    def generate(self, reference_df: pd.DataFrame,
                 target_df: pd.DataFrame) -> CandidateSet:
        candidates = CandidateSet()
        token_idx = _build_token_inverted_index(target_df, "name_tokens")

        # Filter to rare tokens
        rare_idx = {
            tok: ids for tok, ids in token_idx.items()
            if len(ids) <= self.max_df and len(tok) > 2
        }

        for _, row in reference_df.iterrows():
            s1_id = str(row["entity_id"])
            tokens = row["name_tokens"]
            if not tokens:
                continue

            for tok in tokens:
                if tok in rare_idx:
                    rarity_score = 1.0 / max(len(rare_idx[tok]), 1)
                    for cand_id in rare_idx[tok]:
                        source = Source.from_id(cand_id).value
                        candidates.add(
                            s1_id, cand_id, source, self.name, rarity_score
                        )

        return candidates


# ──────────────────────────────────────────────────────────────
# POSTAL BLOCKER
# ──────────────────────────────────────────────────────────────

class PostalBlocker(Blocker):
    """Block on exact postal/ZIP/PIN code."""

    @property
    def name(self) -> str:
        return "postal"

    def generate(self, reference_df: pd.DataFrame,
                 target_df: pd.DataFrame) -> CandidateSet:
        candidates = CandidateSet()
        target_idx = _build_inverted_index(target_df, "postal")

        for _, row in reference_df.iterrows():
            key = str(row["postal"]).strip()
            if key and key in target_idx:
                s1_id = str(row["entity_id"])
                for cand_id in target_idx[key]:
                    source = Source.from_id(cand_id).value
                    candidates.add(s1_id, cand_id, source, self.name, 1.0)

        return candidates


# ──────────────────────────────────────────────────────────────
# COUNTRY BLOCKER
# ──────────────────────────────────────────────────────────────

class CountryBlocker(Blocker):
    """Block on exact country. Used as a FILTER, not standalone."""

    @property
    def name(self) -> str:
        return "country"

    def generate(self, reference_df: pd.DataFrame,
                 target_df: pd.DataFrame) -> CandidateSet:
        candidates = CandidateSet()
        target_idx = _build_inverted_index(target_df, "country_norm")

        for _, row in reference_df.iterrows():
            key = str(row["country_norm"]).strip()
            if key and key in target_idx:
                s1_id = str(row["entity_id"])
                for cand_id in target_idx[key]:
                    source = Source.from_id(cand_id).value
                    candidates.add(s1_id, cand_id, source, self.name, 1.0)

        return candidates


# ──────────────────────────────────────────────────────────────
# NAME + COUNTRY COMPOUND BLOCKER
# ──────────────────────────────────────────────────────────────

class NameCountryBlocker(Blocker):
    """Block on normalized name + country compound key."""

    @property
    def name(self) -> str:
        return "name_country"

    def generate(self, reference_df: pd.DataFrame,
                 target_df: pd.DataFrame) -> CandidateSet:
        candidates = CandidateSet()

        # Build compound index
        target_compound: Dict[str, List[EntityID]] = defaultdict(list)
        for _, row in target_df.iterrows():
            nn = str(row["name_norm"]).strip()
            cn = str(row["country_norm"]).strip()
            if nn:
                key = f"{nn}||{cn}"
                target_compound[key].append(str(row["entity_id"]))

        for _, row in reference_df.iterrows():
            nn = str(row["name_norm"]).strip()
            cn = str(row["country_norm"]).strip()
            if nn:
                key = f"{nn}||{cn}"
                if key in target_compound:
                    s1_id = str(row["entity_id"])
                    for cand_id in target_compound[key]:
                        source = Source.from_id(cand_id).value
                        candidates.add(
                            s1_id, cand_id, source, self.name, 1.0
                        )

        return candidates


# ──────────────────────────────────────────────────────────────
# ADDRESS TOKEN BLOCKER
# ──────────────────────────────────────────────────────────────

class AddressTokenBlocker(Blocker):
    """Block on shared address tokens with minimum overlap."""

    def __init__(self, min_shared: int = 2):
        self.min_shared = min_shared

    @property
    def name(self) -> str:
        return "address_token"

    def generate(self, reference_df: pd.DataFrame,
                 target_df: pd.DataFrame) -> CandidateSet:
        candidates = CandidateSet()
        token_idx = _build_token_inverted_index(target_df, "address_tokens")

        address_stops = {
            "street", "road", "avenue", "drive", "lane", "floor",
            "building", "suite", "apartment", "number", "near",
            "north", "south", "east", "west", "post", "office",
        }

        for _, row in reference_df.iterrows():
            s1_id = str(row["entity_id"])
            tokens = row["address_tokens"]
            if not tokens:
                continue

            cand_counts: Counter = Counter()
            for tok in tokens:
                if tok in address_stops or len(tok) <= 1:
                    continue
                if tok in token_idx:
                    for cand_id in token_idx[tok]:
                        cand_counts[cand_id] += 1

            for cand_id, count in cand_counts.items():
                if count >= self.min_shared:
                    source = Source.from_id(cand_id).value
                    n_meaningful = sum(
                        1 for t in tokens
                        if t not in address_stops and len(t) > 1
                    )
                    score = count / max(n_meaningful, 1)
                    candidates.add(s1_id, cand_id, source, self.name, score)

        return candidates


# ──────────────────────────────────────────────────────────────
# CHARACTER N-GRAM BLOCKER
# ──────────────────────────────────────────────────────────────

class CharNgramBlocker(Blocker):
    """
    Block using character n-gram TF-IDF retrieval.

    Uses scikit-learn TfidfVectorizer with char_wb analyzer.
    Returns top-k candidates per S1 record by cosine similarity.
    """

    def __init__(self, ngram_range: Tuple[int, int] = (3, 4),
                 top_k: int = 50, min_score: float = 0.1):
        self.ngram_range = ngram_range
        self.top_k = top_k
        self.min_score = min_score

    @property
    def name(self) -> str:
        return "char_ngram"

    def generate(self, reference_df: pd.DataFrame,
                 target_df: pd.DataFrame) -> CandidateSet:
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.metrics.pairwise import cosine_similarity

        candidates = CandidateSet()

        ref_names = reference_df["name_norm"].fillna("").tolist()
        tgt_names = target_df["name_norm"].fillna("").tolist()
        ref_ids = reference_df["entity_id"].tolist()
        tgt_ids = target_df["entity_id"].tolist()

        if not tgt_names or not ref_names:
            return candidates

        # Fit on target, transform both
        vectorizer = TfidfVectorizer(
            analyzer="char_wb",
            ngram_range=self.ngram_range,
            max_features=50000,
            sublinear_tf=True,
        )

        try:
            tgt_matrix = vectorizer.fit_transform(tgt_names)
            ref_matrix = vectorizer.transform(ref_names)
        except ValueError:
            return candidates

        # Batch cosine similarity
        batch_size = 500
        for start in range(0, len(ref_ids), batch_size):
            end = min(start + batch_size, len(ref_ids))
            sim_matrix = cosine_similarity(
                ref_matrix[start:end], tgt_matrix
            )

            for i in range(sim_matrix.shape[0]):
                ref_idx = start + i
                s1_id = str(ref_ids[ref_idx])

                scores = sim_matrix[i]
                # Get top-k
                if len(scores) <= self.top_k:
                    top_indices = np.argsort(scores)[::-1]
                else:
                    top_indices = np.argpartition(
                        scores, -self.top_k
                    )[-self.top_k:]
                    top_indices = top_indices[
                        np.argsort(scores[top_indices])[::-1]
                    ]

                for idx in top_indices:
                    score = float(scores[idx])
                    if score < self.min_score:
                        continue
                    cand_id = str(tgt_ids[idx])
                    source = Source.from_id(cand_id).value
                    candidates.add(
                        s1_id, cand_id, source, self.name, score
                    )

        return candidates


# ──────────────────────────────────────────────────────────────
# TF-IDF TOKEN BLOCKER
# ──────────────────────────────────────────────────────────────

class TfidfBlocker(Blocker):
    """
    Block using word-level TF-IDF cosine retrieval on names.
    """

    def __init__(self, top_k: int = 30, min_score: float = 0.15):
        self.top_k = top_k
        self.min_score = min_score

    @property
    def name(self) -> str:
        return "tfidf_name"

    def generate(self, reference_df: pd.DataFrame,
                 target_df: pd.DataFrame) -> CandidateSet:
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.metrics.pairwise import cosine_similarity

        candidates = CandidateSet()

        ref_names = reference_df["name_norm"].fillna("").tolist()
        tgt_names = target_df["name_norm"].fillna("").tolist()
        ref_ids = reference_df["entity_id"].tolist()
        tgt_ids = target_df["entity_id"].tolist()

        if not tgt_names or not ref_names:
            return candidates

        vectorizer = TfidfVectorizer(
            analyzer="word",
            ngram_range=(1, 2),
            max_features=50000,
            sublinear_tf=True,
        )

        try:
            tgt_matrix = vectorizer.fit_transform(tgt_names)
            ref_matrix = vectorizer.transform(ref_names)
        except ValueError:
            return candidates

        batch_size = 500
        for start in range(0, len(ref_ids), batch_size):
            end = min(start + batch_size, len(ref_ids))
            sim_matrix = cosine_similarity(
                ref_matrix[start:end], tgt_matrix
            )

            for i in range(sim_matrix.shape[0]):
                ref_idx = start + i
                s1_id = str(ref_ids[ref_idx])

                scores = sim_matrix[i]
                if len(scores) <= self.top_k:
                    top_indices = np.argsort(scores)[::-1]
                else:
                    top_indices = np.argpartition(
                        scores, -self.top_k
                    )[-self.top_k:]
                    top_indices = top_indices[
                        np.argsort(scores[top_indices])[::-1]
                    ]

                for idx in top_indices:
                    score = float(scores[idx])
                    if score < self.min_score:
                        continue
                    cand_id = str(tgt_ids[idx])
                    source = Source.from_id(cand_id).value
                    candidates.add(
                        s1_id, cand_id, source, self.name, score
                    )

        return candidates


# ──────────────────────────────────────────────────────────────
# FUZZY TOP-K BLOCKER
# ──────────────────────────────────────────────────────────────

class FuzzyTopKBlocker(Blocker):
    """
    Block using RapidFuzz top-k fuzzy matching on names.

    Uses process.cdist for batch computation when available.
    Falls back to sequential matching if RapidFuzz is not installed.
    """

    def __init__(self, top_k: int = 20, min_score: float = 60.0):
        self.top_k = top_k
        self.min_score = min_score

    @property
    def name(self) -> str:
        return "fuzzy_topk"

    def generate(self, reference_df: pd.DataFrame,
                 target_df: pd.DataFrame) -> CandidateSet:
        candidates = CandidateSet()

        ref_names = reference_df["name_norm"].fillna("").tolist()
        tgt_names = target_df["name_norm"].fillna("").tolist()
        ref_ids = reference_df["entity_id"].tolist()
        tgt_ids = target_df["entity_id"].tolist()

        if not tgt_names or not ref_names:
            return candidates

        try:
            from rapidfuzz import fuzz, process

            for i, s1_name in enumerate(ref_names):
                if not s1_name:
                    continue
                s1_id = str(ref_ids[i])

                results = process.extract(
                    s1_name, tgt_names,
                    scorer=fuzz.WRatio,
                    limit=self.top_k,
                    score_cutoff=self.min_score,
                )

                for match_name, score, idx in results:
                    cand_id = str(tgt_ids[idx])
                    source = Source.from_id(cand_id).value
                    candidates.add(
                        s1_id, cand_id, source, self.name,
                        score / 100.0
                    )

        except ImportError:
            # Fallback: use SequenceMatcher
            from difflib import SequenceMatcher

            for i, s1_name in enumerate(ref_names):
                if not s1_name:
                    continue
                s1_id = str(ref_ids[i])

                scores = []
                for j, tgt_name in enumerate(tgt_names):
                    if not tgt_name:
                        continue
                    ratio = SequenceMatcher(
                        None, s1_name, tgt_name
                    ).ratio()
                    if ratio >= self.min_score / 100.0:
                        scores.append((j, ratio))

                scores.sort(key=lambda x: x[1], reverse=True)
                for j, score in scores[:self.top_k]:
                    cand_id = str(tgt_ids[j])
                    source = Source.from_id(cand_id).value
                    candidates.add(
                        s1_id, cand_id, source, self.name, score
                    )

        return candidates


# ──────────────────────────────────────────────────────────────
# MULTI-BLOCK CANDIDATE GENERATOR
# ──────────────────────────────────────────────────────────────

class CandidateGenerator:
    """
    Runs multiple blockers and unions their candidates.

    Architecture:

        S1 RECORD
            │
        ┌───┼───┐───┐───┐
        ▼   ▼   ▼   ▼   ▼
      B1  B2  B3  B4  B5
        │   │   │   │   │
        └───┼───┘───┘───┘
            ▼
        UNION CANDIDATES
            │
            ▼
        DEDUPLICATE
            │
            ▼
        CANDIDATE SET
    """

    def __init__(self, blockers: Optional[List[Blocker]] = None):
        self.blockers: List[Blocker] = blockers or []

    def add_blocker(self, blocker: Blocker) -> None:
        self.blockers.append(blocker)

    def generate(
        self,
        reference_df: pd.DataFrame,
        s2_df: pd.DataFrame,
        s3_df: Optional[pd.DataFrame] = None,
        all_s1_ids: Optional[List[EntityID]] = None,
    ) -> CandidateSet:
        """
        Generate candidates from S2 and optionally S3.

        Runs each blocker against S2 and S3 separately,
        then merges into a single CandidateSet with full provenance.
        """
        combined = CandidateSet()

        for blocker in self.blockers:
            # S2 candidates
            s2_candidates = blocker.generate(reference_df, s2_df)
            combined.merge(s2_candidates)

            # S3 candidates if provided
            if s3_df is not None and len(s3_df) > 0:
                s3_candidates = blocker.generate(reference_df, s3_df)
                combined.merge(s3_candidates)

        return combined
