"""
inference/pipeline.py — End-to-end ER inference pipeline.

Architecture:
    LOAD → NORMALIZE → INDEX → BLOCK → FEATURE → SCORE → DECIDE → SUBMIT

This is the single entry point for both validation and test inference.
Every component is injected through the pipeline, not hard-coded.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd

from src.entity_resolution.core import (
    EntityID, Source, GroundTruth, Predictions, CandidateMap,
    ScoredCandidates,
)
from src.entity_resolution.core.records import (
    RecordStore, load_source_tsv, load_ground_truth,
)
from src.entity_resolution.blocking import CandidateSet
from src.entity_resolution.blocking.blockers import (
    CandidateGenerator,
    ExactNameBlocker,
    ExactCompactNameBlocker,
    TokenNameBlocker,
    RareTokenBlocker,
    PostalBlocker,
    NameCountryBlocker,
    AddressTokenBlocker,
    CharNgramBlocker,
    TfidfBlocker,
    FuzzyTopKBlocker,
)
from src.entity_resolution.features import FeatureBuilder, ALL_FEATURES
from src.entity_resolution.models import PairScorer, RuleBasedScorer
from src.entity_resolution.decision import EntityResolver, ResolverConfig
from src.entity_resolution.validation import (
    compute_entity_f05, compute_blocking_metrics,
    EntityF05Result, BlockingMetrics,
)
from src.entity_resolution.submission import (
    write_submission_package, validate_submission,
)

logger = logging.getLogger(__name__)


@dataclass
class PipelineConfig:
    """Configuration for the full ER pipeline."""
    # Blocking
    use_exact_name: bool = True
    use_compact_name: bool = True
    use_token_name: bool = True
    use_rare_token: bool = True
    use_postal: bool = True
    use_name_country: bool = True
    use_address_token: bool = True
    use_char_ngram: bool = False  # Expensive, off by default
    use_tfidf: bool = False       # Expensive, off by default
    use_fuzzy: bool = False       # Expensive, off by default

    # Blocking params
    token_min_shared: int = 1
    rare_token_max_df: int = 10
    address_min_shared: int = 2
    char_ngram_top_k: int = 50
    tfidf_top_k: int = 30
    fuzzy_top_k: int = 20

    # Decision
    match_threshold: float = 0.5
    multi_match_threshold: float = 0.6
    multi_match_gap: float = 0.1
    max_matches: int = 0

    # Output
    output_dir: str = "submission/output"


@dataclass
class PipelineResult:
    """Complete pipeline result."""
    predictions: Predictions
    candidate_map: CandidateMap
    scored_candidates: ScoredCandidates
    features_df: Optional[pd.DataFrame]
    blocking_metrics: Optional[BlockingMetrics]
    entity_f05: Optional[EntityF05Result]
    timing: Dict[str, float] = field(default_factory=dict)
    config: Optional[PipelineConfig] = None


class ERPipeline:
    """
    End-to-end Entity Resolution pipeline.

    Usage:
        pipeline = ERPipeline(config, scorer)
        result = pipeline.run(s1_df, s2_df, s3_df)
        # OR for validation:
        result = pipeline.run(s1_df, s2_df, s3_df, ground_truth=gt)
    """

    def __init__(
        self,
        config: Optional[PipelineConfig] = None,
        scorer: Optional[PairScorer] = None,
    ):
        self.config = config or PipelineConfig()
        self.scorer = scorer or RuleBasedScorer()
        self._build_candidate_generator()

    def _build_candidate_generator(self):
        """Build candidate generator from config."""
        cfg = self.config
        gen = CandidateGenerator()

        if cfg.use_exact_name:
            gen.add_blocker(ExactNameBlocker())
        if cfg.use_compact_name:
            gen.add_blocker(ExactCompactNameBlocker())
        if cfg.use_token_name:
            gen.add_blocker(TokenNameBlocker(min_shared=cfg.token_min_shared))
        if cfg.use_rare_token:
            gen.add_blocker(RareTokenBlocker(max_df=cfg.rare_token_max_df))
        if cfg.use_postal:
            gen.add_blocker(PostalBlocker())
        if cfg.use_name_country:
            gen.add_blocker(NameCountryBlocker())
        if cfg.use_address_token:
            gen.add_blocker(AddressTokenBlocker(
                min_shared=cfg.address_min_shared
            ))
        if cfg.use_char_ngram:
            gen.add_blocker(CharNgramBlocker(top_k=cfg.char_ngram_top_k))
        if cfg.use_tfidf:
            gen.add_blocker(TfidfBlocker(top_k=cfg.tfidf_top_k))
        if cfg.use_fuzzy:
            gen.add_blocker(FuzzyTopKBlocker(top_k=cfg.fuzzy_top_k))

        self.candidate_generator = gen

    def run(
        self,
        s1_df: pd.DataFrame,
        s2_df: pd.DataFrame,
        s3_df: Optional[pd.DataFrame] = None,
        ground_truth: Optional[GroundTruth] = None,
    ) -> PipelineResult:
        """
        Execute the full pipeline.

        Parameters
        ----------
        s1_df : Normalized S1 records
        s2_df : Normalized S2 records
        s3_df : Normalized S3 records (optional)
        ground_truth : For validation only
        """
        timing: Dict[str, float] = {}
        s1_ids = s1_df["entity_id"].tolist()

        # ── STEP 1: Candidate Generation ─────────────────────
        t0 = time.time()
        logger.info("Generating candidates...")
        candidate_set = self.candidate_generator.generate(
            s1_df, s2_df, s3_df, s1_ids
        )
        timing["blocking"] = time.time() - t0
        logger.info(
            "Candidates generated: %d pairs from %d S1 entities",
            len(candidate_set), len(s1_ids)
        )

        candidate_map = candidate_set.to_candidate_map()

        # Ensure every S1 has an entry
        for s1_id in s1_ids:
            if s1_id not in candidate_map:
                candidate_map[s1_id] = set()

        # ── STEP 2: Blocking Metrics (if ground truth available) ──
        blocking_metrics = None
        if ground_truth:
            blocking_metrics = compute_blocking_metrics(
                ground_truth, candidate_map,
                n_s2=len(s2_df), n_s3=len(s3_df) if s3_df is not None else 0,
            )
            logger.info(
                "Blocking recall: %.4f | Candidates: %d | Reduction: %.4f",
                blocking_metrics.candidate_recall,
                blocking_metrics.total_candidates,
                blocking_metrics.reduction_ratio,
            )

        # ── STEP 3: Feature Engineering ──────────────────────
        t0 = time.time()
        logger.info("Building features...")

        # Combine all records for feature lookup
        all_dfs = [s1_df, s2_df]
        if s3_df is not None:
            all_dfs.append(s3_df)
        all_records = pd.concat(all_dfs, ignore_index=True)

        feature_builder = FeatureBuilder(all_records)
        features_df = feature_builder.build(candidate_set, s1_ids)
        timing["features"] = time.time() - t0
        logger.info("Features built: %d pairs × %d features",
                     len(features_df), len(ALL_FEATURES))

        # ── STEP 4: Scoring ──────────────────────────────────
        t0 = time.time()
        logger.info("Scoring candidates with %s...", self.scorer.name)

        scored_candidates: ScoredCandidates = {}

        if len(features_df) > 0:
            feat_cols = [c for c in ALL_FEATURES if c in features_df.columns]

            if isinstance(self.scorer, RuleBasedScorer):
                scores = self.scorer.predict_proba_df(features_df)
            else:
                X = features_df[feat_cols].values.astype(np.float32)
                scores = self.scorer.predict_proba(X)

            for idx in range(len(features_df)):
                s1_id = features_df.iloc[idx]["s1_id"]
                cand_id = features_df.iloc[idx]["candidate_id"]
                score = float(scores[idx])
                scored_candidates.setdefault(s1_id, []).append(
                    (cand_id, score)
                )

        timing["scoring"] = time.time() - t0

        # ── STEP 5: Entity-Level Decision ────────────────────
        t0 = time.time()
        logger.info("Resolving entity decisions...")

        resolver_config = ResolverConfig(
            match_threshold=self.config.match_threshold,
            multi_match_threshold=self.config.multi_match_threshold,
            multi_match_gap=self.config.multi_match_gap,
            max_matches=self.config.max_matches,
        )
        resolver = EntityResolver(resolver_config)
        predictions = resolver.resolve(scored_candidates, s1_ids)

        timing["decision"] = time.time() - t0

        # ── STEP 6: Evaluation (if ground truth available) ───
        entity_f05 = None
        if ground_truth:
            entity_f05 = compute_entity_f05(ground_truth, predictions)
            logger.info(
                "Entity F0.5: %.4f | Precision: %.4f | Recall: %.4f",
                entity_f05.macro_f05,
                entity_f05.macro_precision,
                entity_f05.macro_recall,
            )
            logger.info(
                "Singletons: true=%d pred=%d correct=%d | "
                "False merges: %d | Missed: %d",
                entity_f05.n_singletons_true,
                entity_f05.n_singletons_pred,
                entity_f05.n_singletons_correct,
                entity_f05.false_merges,
                entity_f05.missed_matches,
            )

        timing["total"] = sum(timing.values())

        return PipelineResult(
            predictions=predictions,
            candidate_map=candidate_map,
            scored_candidates=scored_candidates,
            features_df=features_df,
            blocking_metrics=blocking_metrics,
            entity_f05=entity_f05,
            timing=timing,
            config=self.config,
        )

    def run_and_submit(
        self,
        s1_df: pd.DataFrame,
        s2_df: pd.DataFrame,
        s3_df: Optional[pd.DataFrame] = None,
        ground_truth: Optional[GroundTruth] = None,
    ) -> PipelineResult:
        """Run pipeline and write submission files."""
        result = self.run(s1_df, s2_df, s3_df, ground_truth)

        # Write submission
        s1_ids = s1_df["entity_id"].tolist()
        s2_ids = set(s2_df["entity_id"].tolist())
        s3_ids = set(s3_df["entity_id"].tolist()) if s3_df is not None else set()

        write_submission_package(
            result.predictions,
            result.candidate_map,
            s1_ids,
            s2_ids,
            s3_ids,
            output_dir=self.config.output_dir,
        )

        logger.info("Submission written to %s", self.config.output_dir)
        return result
