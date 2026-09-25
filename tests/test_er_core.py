"""
tests/test_er_core.py — Comprehensive tests for the Entity Resolution engine.

Tests cover:
    - Record normalization
    - Source identification
    - Ground truth loading
    - Blocking (exact, token, rare token, postal)
    - Candidate set operations
    - Feature computation
    - Similarity functions
    - Entity-level F0.5 metric
    - Blocking metrics
    - Decision engine (singleton, single match, multi-match)
    - Submission validation
    - End-to-end integration
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# ── Core types ──────────────────────────────────────────────
from src.entity_resolution.core import Source, EntityID

# ── Records ─────────────────────────────────────────────────
from src.entity_resolution.core.records import (
    normalize_name,
    normalize_address,
    normalize_country,
    extract_tokens,
    extract_postal,
    extract_numeric_tokens,
    load_source_tsv,
    load_ground_truth,
    RecordStore,
)

# ── Blocking ────────────────────────────────────────────────
from src.entity_resolution.blocking import (
    CandidateSet, Blocker, CandidateRecord,
)
from src.entity_resolution.blocking.blockers import (
    ExactNameBlocker,
    ExactCompactNameBlocker,
    TokenNameBlocker,
    RareTokenBlocker,
    PostalBlocker,
    NameCountryBlocker,
    CandidateGenerator,
)

# ── Similarity ──────────────────────────────────────────────
from src.entity_resolution.similarity import (
    levenshtein_distance,
    normalized_levenshtein,
    jaro_similarity,
    jaro_winkler_similarity,
    token_jaccard,
    token_overlap_coeff,
    char_ngram_similarity,
    exact_match,
    length_ratio,
)

# ── Features ────────────────────────────────────────────────
from src.entity_resolution.features import FeatureBuilder, ALL_FEATURES

# ── Validation ──────────────────────────────────────────────
from src.entity_resolution.validation import (
    compute_entity_f05, compute_blocking_metrics, compute_pair_metrics,
)

# ── Decision ────────────────────────────────────────────────
from src.entity_resolution.decision import (
    EntityResolver, ResolverConfig,
)

# ── Submission ──────────────────────────────────────────────
from src.entity_resolution.submission import (
    validate_submission, write_matching_results, write_candidate_pairs,
)


# ═══════════════════════════════════════════════════════════
# FIXTURES
# ═══════════════════════════════════════════════════════════

@pytest.fixture
def sample_s1_df():
    """Small S1 DataFrame for testing."""
    return pd.DataFrame({
        "entity_id": ["S1-001", "S1-002", "S1-003"],
        "source": [1, 1, 1],
        "name_raw": ["Apollo Pharmacy", "Tata Motors Ltd", "ABC Textiles"],
        "name_norm": ["apollo pharmacy", "tata motors limited", "abc textiles"],
        "name_compact": ["apollopharmacy", "tatamotorslimited", "abctextiles"],
        "name_tokens": [
            ("apollo", "pharmacy"),
            ("tata", "motors", "limited"),
            ("abc", "textiles"),
        ],
        "address_raw": ["123 MG Road, Chennai 600001", "Pune 411001", ""],
        "address_norm": [
            "123 mahatma gandhi road chennai 600001",
            "pune 411001",
            "",
        ],
        "address_tokens": [
            ("123", "mahatma", "gandhi", "road", "chennai", "600001"),
            ("pune", "411001"),
            (),
        ],
        "postal": ["600001", "411001", ""],
        "numeric_tokens": [("123", "600001"), ("411001",), ()],
        "country_raw": ["India", "India", "India"],
        "country_norm": ["in", "in", "in"],
    })


@pytest.fixture
def sample_s2_df():
    """Small S2 DataFrame for testing."""
    return pd.DataFrame({
        "entity_id": ["S2-101", "S2-102", "S2-103", "S2-104"],
        "source": [2, 2, 2, 2],
        "name_raw": [
            "Apollo Pharmacy Pvt Ltd",
            "Tata Motors",
            "XYZ Industries",
            "ABC Textile Corp",
        ],
        "name_norm": [
            "apollo pharmacy private limited",
            "tata motors",
            "xyz industries",
            "abc textile corporation",
        ],
        "name_compact": [
            "apollopharmacyprivatelimited",
            "tatamotors",
            "xyzindustries",
            "abctextilecorporation",
        ],
        "name_tokens": [
            ("apollo", "pharmacy", "private", "limited"),
            ("tata", "motors"),
            ("xyz", "industries"),
            ("abc", "textile", "corporation"),
        ],
        "address_raw": [
            "123 MG Rd, Chennai - 600001",
            "Pune",
            "Mumbai 400001",
            "Delhi",
        ],
        "address_norm": [
            "123 mahatma gandhi road chennai 600001",
            "pune",
            "mumbai 400001",
            "delhi",
        ],
        "address_tokens": [
            ("123", "mahatma", "gandhi", "road", "chennai", "600001"),
            ("pune",),
            ("mumbai", "400001"),
            ("delhi",),
        ],
        "postal": ["600001", "", "400001", ""],
        "numeric_tokens": [("123", "600001"), (), ("400001",), ()],
        "country_raw": ["India", "India", "India", "India"],
        "country_norm": ["in", "in", "in", "in"],
    })


@pytest.fixture
def sample_ground_truth():
    """Ground truth for test data."""
    return {
        "S1-001": {"S2-101"},           # Apollo matched
        "S1-002": {"S2-102"},           # Tata matched
        "S1-003": set(),                 # ABC singleton
    }


# ═══════════════════════════════════════════════════════════
# TEST: Source identification
# ═══════════════════════════════════════════════════════════

class TestSource:
    def test_from_s1_id(self):
        assert Source.from_id("S1-00001") == Source.S1

    def test_from_s2_id(self):
        assert Source.from_id("S2-00001") == Source.S2

    def test_from_s3_id(self):
        assert Source.from_id("S3-00001") == Source.S3

    def test_invalid_id(self):
        with pytest.raises(ValueError):
            Source.from_id("XX-00001")


# ═══════════════════════════════════════════════════════════
# TEST: Normalization
# ═══════════════════════════════════════════════════════════

class TestNormalization:
    def test_name_lowercase(self):
        assert "apollo" in normalize_name("APOLLO")

    def test_name_ampersand(self):
        assert "and" in normalize_name("A & B Corp")

    def test_name_legal_suffix_pvt(self):
        result = normalize_name("ABC Pvt Ltd")
        assert "private" in result
        assert "limited" in result

    def test_name_legal_suffix_corp(self):
        result = normalize_name("XYZ Corp")
        assert "corporation" in result

    def test_name_empty(self):
        assert normalize_name("") == ""
        assert normalize_name(None) == ""

    def test_address_abbreviations(self):
        result = normalize_address("123 MG Rd, Chennai")
        assert "road" in result

    def test_address_st(self):
        result = normalize_address("456 Main St")
        assert "street" in result

    def test_address_empty(self):
        assert normalize_address("") == ""

    def test_country_us(self):
        assert normalize_country("United States") == "us"
        assert normalize_country("USA") == "us"
        assert normalize_country("US") == "us"

    def test_country_india(self):
        assert normalize_country("India") == "in"
        assert normalize_country("IN") == "in"

    def test_country_france(self):
        assert normalize_country("France") == "fr"
        assert normalize_country("FR") == "fr"

    def test_country_unknown_preserved(self):
        """Unknown countries should NOT be discarded."""
        result = normalize_country("Atlantis")
        assert result == "atlantis"

    def test_country_empty(self):
        assert normalize_country("") == ""

    def test_extract_tokens(self):
        assert extract_tokens("apollo pharmacy") == ("apollo", "pharmacy")

    def test_extract_tokens_empty(self):
        assert extract_tokens("") == ()

    def test_extract_postal_us(self):
        assert extract_postal("New York, NY 10001") == "10001"

    def test_extract_postal_india(self):
        assert extract_postal("Chennai 600001") == "600001"

    def test_extract_postal_missing(self):
        assert extract_postal("No postal here") == ""

    def test_extract_numeric(self):
        assert "123" in extract_numeric_tokens("123 Main St, Floor 4")
        assert "4" in extract_numeric_tokens("123 Main St, Floor 4")


# ═══════════════════════════════════════════════════════════
# TEST: Similarity functions
# ═══════════════════════════════════════════════════════════

class TestSimilarity:
    def test_levenshtein_identical(self):
        assert levenshtein_distance("abc", "abc") == 0

    def test_levenshtein_one_edit(self):
        assert levenshtein_distance("abc", "ab") == 1

    def test_normalized_levenshtein_identical(self):
        assert normalized_levenshtein("abc", "abc") == 1.0

    def test_normalized_levenshtein_different(self):
        sim = normalized_levenshtein("abc", "xyz")
        assert 0.0 <= sim <= 1.0

    def test_jaro_identical(self):
        assert jaro_similarity("abc", "abc") == 1.0

    def test_jaro_winkler_identical(self):
        assert jaro_winkler_similarity("abc", "abc") == 1.0

    def test_jaro_winkler_prefix_bonus(self):
        """JW should give higher score than Jaro for common prefix."""
        jw = jaro_winkler_similarity("abcdef", "abcxyz")
        j = jaro_similarity("abcdef", "abcxyz")
        assert jw >= j

    def test_token_jaccard_identical(self):
        assert token_jaccard(["a", "b"], ["a", "b"]) == 1.0

    def test_token_jaccard_disjoint(self):
        assert token_jaccard(["a", "b"], ["c", "d"]) == 0.0

    def test_token_jaccard_partial(self):
        j = token_jaccard(["a", "b", "c"], ["a", "b", "d"])
        assert 0 < j < 1

    def test_token_overlap_coeff(self):
        assert token_overlap_coeff(["a", "b"], ["a", "b", "c"]) == 1.0

    def test_char_ngram_identical(self):
        assert char_ngram_similarity("apollo", "apollo", n=3) == 1.0

    def test_char_ngram_different(self):
        sim = char_ngram_similarity("apollo", "xyz", n=3)
        assert sim == 0.0

    def test_exact_match(self):
        assert exact_match("a", "a") == 1.0
        assert exact_match("a", "b") == 0.0

    def test_length_ratio(self):
        assert length_ratio("abc", "abcdef") == 0.5


# ═══════════════════════════════════════════════════════════
# TEST: Candidate set
# ═══════════════════════════════════════════════════════════

class TestCandidateSet:
    def test_add_and_deduplicate(self):
        cs = CandidateSet()
        cs.add("S1-001", "S2-101", 2, "exact_name", 1.0)
        cs.add("S1-001", "S2-101", 2, "token_name", 0.8)
        assert len(cs) == 1  # Deduplicated

    def test_provenance_merged(self):
        cs = CandidateSet()
        cs.add("S1-001", "S2-101", 2, "exact_name", 1.0)
        cs.add("S1-001", "S2-101", 2, "token_name", 0.8)
        cands = cs.get_candidates_for("S1-001")
        assert len(cands) == 1
        assert cands[0].block_count == 2
        assert "exact_name" in cands[0].blockers_hit
        assert "token_name" in cands[0].blockers_hit

    def test_multiple_candidates(self):
        cs = CandidateSet()
        cs.add("S1-001", "S2-101", 2, "exact_name")
        cs.add("S1-001", "S2-102", 2, "token_name")
        assert len(cs) == 2
        assert len(cs.get_candidate_ids_for("S1-001")) == 2

    def test_to_dataframe(self):
        cs = CandidateSet()
        cs.add("S1-001", "S2-101", 2, "exact_name")
        df = cs.to_dataframe()
        assert len(df) == 1
        assert "s1_id" in df.columns
        assert "candidate_id" in df.columns

    def test_to_candidate_map(self):
        cs = CandidateSet()
        cs.add("S1-001", "S2-101", 2, "exact_name")
        cs.add("S1-001", "S2-102", 2, "token_name")
        cm = cs.to_candidate_map()
        assert cm["S1-001"] == {"S2-101", "S2-102"}

    def test_merge(self):
        cs1 = CandidateSet()
        cs1.add("S1-001", "S2-101", 2, "exact_name")
        cs2 = CandidateSet()
        cs2.add("S1-001", "S2-102", 2, "token_name")
        cs1.merge(cs2)
        assert len(cs1) == 2


# ═══════════════════════════════════════════════════════════
# TEST: Blockers
# ═══════════════════════════════════════════════════════════

class TestBlockers:
    def test_exact_name_blocker(self, sample_s1_df, sample_s2_df):
        blocker = ExactNameBlocker()
        candidates = blocker.generate(sample_s1_df, sample_s2_df)
        # "tata motors limited" != "tata motors" (exact won't match)
        # Check at least some candidates generated
        assert isinstance(candidates, CandidateSet)

    def test_token_name_blocker(self, sample_s1_df, sample_s2_df):
        blocker = TokenNameBlocker(min_shared=1)
        candidates = blocker.generate(sample_s1_df, sample_s2_df)
        # "apollo" token should match S2-101
        cands_001 = candidates.get_candidate_ids_for("S1-001")
        assert "S2-101" in cands_001

    def test_token_name_blocker_tata(self, sample_s1_df, sample_s2_df):
        blocker = TokenNameBlocker(min_shared=1)
        candidates = blocker.generate(sample_s1_df, sample_s2_df)
        cands_002 = candidates.get_candidate_ids_for("S1-002")
        assert "S2-102" in cands_002

    def test_postal_blocker(self, sample_s1_df, sample_s2_df):
        blocker = PostalBlocker()
        candidates = blocker.generate(sample_s1_df, sample_s2_df)
        cands_001 = candidates.get_candidate_ids_for("S1-001")
        assert "S2-101" in cands_001  # Both have 600001

    def test_name_country_blocker(self, sample_s1_df, sample_s2_df):
        blocker = NameCountryBlocker()
        candidates = blocker.generate(sample_s1_df, sample_s2_df)
        # Only exact name+country matches
        assert isinstance(candidates, CandidateSet)

    def test_candidate_generator(self, sample_s1_df, sample_s2_df):
        gen = CandidateGenerator([
            ExactNameBlocker(),
            TokenNameBlocker(min_shared=1),
            PostalBlocker(),
        ])
        candidates = gen.generate(sample_s1_df, sample_s2_df)
        # Should find Apollo match
        cands_001 = candidates.get_candidate_ids_for("S1-001")
        assert "S2-101" in cands_001


# ═══════════════════════════════════════════════════════════
# TEST: Features
# ═══════════════════════════════════════════════════════════

class TestFeatures:
    def test_feature_builder(self, sample_s1_df, sample_s2_df):
        all_df = pd.concat([sample_s1_df, sample_s2_df], ignore_index=True)
        builder = FeatureBuilder(all_df)

        cs = CandidateSet()
        cs.add("S1-001", "S2-101", 2, "exact_name", 1.0)
        features = builder.build(cs)
        assert len(features) == 1
        assert "name_edit_sim" in features.columns
        assert "addr_edit_sim" in features.columns

    def test_feature_values(self, sample_s1_df, sample_s2_df):
        all_df = pd.concat([sample_s1_df, sample_s2_df], ignore_index=True)
        builder = FeatureBuilder(all_df)
        feats = builder.compute_pair_features("S1-001", "S2-101")
        # Apollo Pharmacy vs Apollo Pharmacy Private Limited
        assert feats["name_edit_sim"] > 0.4  # "apollo pharmacy" vs "apollo pharmacy private limited"
        assert feats["addr_postal_match"] == 1.0  # Both 600001

    def test_all_features_present(self, sample_s1_df, sample_s2_df):
        all_df = pd.concat([sample_s1_df, sample_s2_df], ignore_index=True)
        builder = FeatureBuilder(all_df)
        feats = builder.compute_pair_features("S1-001", "S2-101")
        for fname in ALL_FEATURES:
            assert fname in feats, f"Missing feature: {fname}"


# ═══════════════════════════════════════════════════════════
# TEST: Entity-level F0.5 Metric
# ═══════════════════════════════════════════════════════════

class TestEntityF05:
    def test_perfect_score(self):
        truth = {"S1-001": {"S2-101"}, "S1-002": set()}
        pred = {"S1-001": {"S2-101"}, "S1-002": set()}
        result = compute_entity_f05(truth, pred)
        assert result.macro_f05 == 1.0

    def test_all_wrong(self):
        truth = {"S1-001": {"S2-101"}, "S1-002": set()}
        pred = {"S1-001": set(), "S1-002": {"S2-999"}}
        result = compute_entity_f05(truth, pred)
        assert result.macro_f05 == 0.0

    def test_singleton_correct(self):
        truth = {"S1-001": set()}
        pred = {"S1-001": set()}
        result = compute_entity_f05(truth, pred)
        assert result.macro_f05 == 1.0

    def test_singleton_false_merge(self):
        truth = {"S1-001": set()}
        pred = {"S1-001": {"S2-999"}}
        result = compute_entity_f05(truth, pred)
        assert result.macro_f05 == 0.0

    def test_missed_match(self):
        truth = {"S1-001": {"S2-101"}}
        pred = {"S1-001": set()}
        result = compute_entity_f05(truth, pred)
        assert result.macro_f05 == 0.0

    def test_partial_match(self):
        truth = {"S1-001": {"S2-101", "S2-102"}}
        pred = {"S1-001": {"S2-101"}}
        result = compute_entity_f05(truth, pred)
        # P=1, R=0.5, F0.5 = 1.25*1*0.5/(0.25*1+0.5) = 0.625/0.75 ≈ 0.833
        assert abs(result.macro_f05 - 0.8333) < 0.01

    def test_extra_match(self):
        truth = {"S1-001": {"S2-101"}}
        pred = {"S1-001": {"S2-101", "S2-999"}}
        result = compute_entity_f05(truth, pred)
        # P=0.5, R=1, F0.5 = 1.25*0.5*1/(0.25*0.5+1) = 0.625/1.125 ≈ 0.556
        assert abs(result.macro_f05 - 0.5556) < 0.01

    def test_macro_averaging(self):
        truth = {
            "S1-001": {"S2-101"},
            "S1-002": set(),
        }
        pred = {
            "S1-001": {"S2-101"},  # F0.5 = 1.0
            "S1-002": set(),        # F0.5 = 1.0
        }
        result = compute_entity_f05(truth, pred)
        assert result.macro_f05 == 1.0

    def test_f05_precision_heavy(self):
        """F0.5 should penalize FP more than FN."""
        truth = {"S1-001": {"S2-101"}}

        # Extra prediction (FP) — precision drops
        pred_fp = {"S1-001": {"S2-101", "S2-999"}}
        r_fp = compute_entity_f05(truth, pred_fp)

        # Missing prediction (FN) — recall drops
        truth_multi = {"S1-001": {"S2-101", "S2-102"}}
        pred_fn = {"S1-001": {"S2-101"}}
        r_fn = compute_entity_f05(truth_multi, pred_fn)

        # F0.5 should be higher when precision is maintained (FN case)
        assert r_fn.macro_f05 > r_fp.macro_f05


# ═══════════════════════════════════════════════════════════
# TEST: Blocking Metrics
# ═══════════════════════════════════════════════════════════

class TestBlockingMetrics:
    def test_perfect_recall(self):
        truth = {"S1-001": {"S2-101"}, "S1-002": {"S2-102"}}
        candidates = {"S1-001": {"S2-101", "S2-103"}, "S1-002": {"S2-102"}}
        result = compute_blocking_metrics(truth, candidates, n_s2=4, n_s3=0)
        assert result.candidate_recall == 1.0

    def test_missing_recall(self):
        truth = {"S1-001": {"S2-101"}}
        candidates = {"S1-001": {"S2-103"}}  # Wrong candidate
        result = compute_blocking_metrics(truth, candidates, n_s2=4, n_s3=0)
        assert result.candidate_recall == 0.0

    def test_singleton_recall(self):
        truth = {"S1-001": set()}  # Singleton
        candidates = {"S1-001": set()}
        result = compute_blocking_metrics(truth, candidates, n_s2=4, n_s3=0)
        assert result.candidate_recall == 1.0  # No true matches to find


# ═══════════════════════════════════════════════════════════
# TEST: Decision Engine
# ═══════════════════════════════════════════════════════════

class TestResolver:
    def test_singleton(self):
        resolver = EntityResolver(ResolverConfig(match_threshold=0.5))
        scored = {"S1-001": [("S2-101", 0.2)]}  # Below threshold
        preds = resolver.resolve(scored, ["S1-001"])
        assert preds["S1-001"] == set()  # Singleton

    def test_single_match(self):
        resolver = EntityResolver(ResolverConfig(match_threshold=0.5))
        scored = {"S1-001": [("S2-101", 0.9)]}
        preds = resolver.resolve(scored, ["S1-001"])
        assert preds["S1-001"] == {"S2-101"}

    def test_multi_match(self):
        resolver = EntityResolver(ResolverConfig(
            match_threshold=0.5,
            multi_match_threshold=0.5,
            multi_match_gap=0.3,
        ))
        scored = {"S1-001": [("S2-101", 0.95), ("S2-102", 0.90)]}
        preds = resolver.resolve(scored, ["S1-001"])
        assert preds["S1-001"] == {"S2-101", "S2-102"}

    def test_no_candidates(self):
        resolver = EntityResolver()
        scored = {}
        preds = resolver.resolve(scored, ["S1-001"])
        assert preds["S1-001"] == set()

    def test_threshold_optimization(self, sample_ground_truth):
        resolver = EntityResolver()
        scored = {
            "S1-001": [("S2-101", 0.9), ("S2-102", 0.3)],
            "S1-002": [("S2-102", 0.85)],
            "S1-003": [("S2-104", 0.2)],
        }
        best_thresh, best_f05 = resolver.optimize_threshold(
            scored, sample_ground_truth,
            thresholds=[0.3, 0.5, 0.7, 0.9],
        )
        assert 0.0 <= best_f05 <= 1.0
        assert 0.0 < best_thresh < 1.0


# ═══════════════════════════════════════════════════════════
# TEST: Submission Validation
# ═══════════════════════════════════════════════════════════

class TestSubmissionValidation:
    def test_valid_submission(self):
        preds = {"S1-001": {"S2-101"}, "S1-002": set()}
        cands = {"S1-001": {"S2-101", "S2-102"}, "S1-002": set()}
        s1_ids = ["S1-001", "S1-002"]
        s2_ids = {"S2-101", "S2-102"}
        s3_ids = set()

        ok, checks = validate_submission(preds, cands, s1_ids, s2_ids, s3_ids)
        assert ok

    def test_missing_s1(self):
        preds = {"S1-001": {"S2-101"}}  # Missing S1-002
        cands = {"S1-001": {"S2-101"}}
        s1_ids = ["S1-001", "S1-002"]
        s2_ids = {"S2-101"}
        s3_ids = set()

        ok, checks = validate_submission(preds, cands, s1_ids, s2_ids, s3_ids)
        assert not ok

    def test_match_not_in_candidates(self):
        preds = {"S1-001": {"S2-101"}}
        cands = {"S1-001": {"S2-102"}}  # S2-101 NOT in candidates
        s1_ids = ["S1-001"]
        s2_ids = {"S2-101", "S2-102"}
        s3_ids = set()

        ok, checks = validate_submission(preds, cands, s1_ids, s2_ids, s3_ids)
        assert not ok

    def test_s1_in_matches(self):
        preds = {"S1-001": {"S1-002"}}  # S1 ID in matches!
        cands = {"S1-001": {"S1-002"}}
        s1_ids = ["S1-001"]
        s2_ids = set()
        s3_ids = set()

        ok, checks = validate_submission(preds, cands, s1_ids, s2_ids, s3_ids)
        assert not ok


# ═══════════════════════════════════════════════════════════
# TEST: Submission File Writing
# ═══════════════════════════════════════════════════════════

class TestSubmissionWriting:
    def test_write_matching_results(self, tmp_path):
        preds = {"S1-001": {"S2-101", "S3-201"}, "S1-002": set()}
        path = write_matching_results(
            preds, str(tmp_path / "matching_results.tsv"),
            s1_order=["S1-001", "S1-002"],
        )
        df = pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False)
        assert len(df) == 2
        assert df.iloc[0]["source1_entity_id"] == "S1-001"
        assert "S2-101" in df.iloc[0]["matched_entity_ids"]
        assert df.iloc[1]["matched_entity_ids"] == ""

    def test_write_candidate_pairs(self, tmp_path):
        cands = {"S1-001": {"S2-101", "S2-102"}}
        path = write_candidate_pairs(
            cands, str(tmp_path / "candidate_pairs.tsv"),
        )
        df = pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False)
        assert len(df) == 1
        assert "S2-101" in df.iloc[0]["candidate_entity_ids"]


# ═══════════════════════════════════════════════════════════
# TEST: Ground Truth Loading
# ═══════════════════════════════════════════════════════════

class TestGroundTruthLoading:
    def test_load_ground_truth(self, tmp_path):
        gt_path = tmp_path / "ground_truth.tsv"
        gt_path.write_text(
            "source1_entity_id\tmatched_entity_ids\n"
            "S1-001\tS2-101,S2-102\n"
            "S1-002\tS3-201\n"
            "S1-003\t\n"
        )
        gt = load_ground_truth(str(gt_path))
        assert gt["S1-001"] == {"S2-101", "S2-102"}
        assert gt["S1-002"] == {"S3-201"}
        assert gt["S1-003"] == set()


# ═══════════════════════════════════════════════════════════
# TEST: TSV Loading
# ═══════════════════════════════════════════════════════════

class TestTSVLoading:
    def test_load_source_tsv(self, tmp_path):
        tsv_path = tmp_path / "source1.tsv"
        tsv_path.write_text(
            "entity_id\tbusiness_name\tbusiness_address\tcountry\n"
            "S1-001\tApollo Pharmacy\t123 MG Road Chennai 600001\tIndia\n"
            "S1-002\tTata Motors Ltd\tPune 411001\tIndia\n"
        )
        df = load_source_tsv(str(tsv_path), expected_source=Source.S1)
        assert len(df) == 2
        assert "name_norm" in df.columns
        assert "address_norm" in df.columns
        assert "country_norm" in df.columns
        assert df.iloc[0]["country_norm"] == "in"


# ═══════════════════════════════════════════════════════════
# TEST: End-to-end Integration
# ═══════════════════════════════════════════════════════════

class TestIntegration:
    def test_full_pipeline(self, sample_s1_df, sample_s2_df,
                          sample_ground_truth):
        """Test complete pipeline: block → feature → score → decide → evaluate."""
        from src.entity_resolution.inference import ERPipeline, PipelineConfig

        config = PipelineConfig(
            use_exact_name=True,
            use_token_name=True,
            use_postal=True,
            use_compact_name=False,
            use_rare_token=False,
            use_name_country=False,
            use_address_token=False,
            use_char_ngram=False,
            use_tfidf=False,
            use_fuzzy=False,
            match_threshold=0.3,  # Low threshold for test
        )

        pipeline = ERPipeline(config=config)
        result = pipeline.run(
            sample_s1_df, sample_s2_df,
            ground_truth=sample_ground_truth,
        )

        # Pipeline should produce predictions for all S1 entities
        assert "S1-001" in result.predictions
        assert "S1-002" in result.predictions
        assert "S1-003" in result.predictions

        # Blocking metrics should be computed
        assert result.blocking_metrics is not None
        assert result.blocking_metrics.candidate_recall >= 0.0

        # Entity F0.5 should be computed
        assert result.entity_f05 is not None
        assert 0.0 <= result.entity_f05.macro_f05 <= 1.0

        # Features should be generated
        assert result.features_df is not None

    def test_pipeline_empty_s3(self, sample_s1_df, sample_s2_df,
                               sample_ground_truth):
        """Test pipeline with no S3 data."""
        from src.entity_resolution.inference import ERPipeline, PipelineConfig

        config = PipelineConfig(
            use_exact_name=True,
            use_token_name=True,
            match_threshold=0.3,
        )
        pipeline = ERPipeline(config=config)
        result = pipeline.run(sample_s1_df, sample_s2_df)
        assert result.predictions is not None


# ═══════════════════════════════════════════════════════════
# TEST: Pair Metrics
# ═══════════════════════════════════════════════════════════

class TestPairMetrics:
    def test_perfect(self):
        y_true = np.array([1, 0, 1, 0])
        y_pred = np.array([1, 0, 1, 0])
        m = compute_pair_metrics(y_true, y_pred)
        assert m.accuracy == 1.0
        assert m.precision == 1.0
        assert m.recall == 1.0

    def test_all_wrong(self):
        y_true = np.array([1, 0])
        y_pred = np.array([0, 1])
        m = compute_pair_metrics(y_true, y_pred)
        assert m.precision == 0.0
        assert m.recall == 0.0
