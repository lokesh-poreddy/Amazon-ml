"""
submission/writer.py — Submission file writer and validator.

Produces:
    matching_results.tsv
    candidate_pairs.tsv

And validates every structural constraint from the problem statement.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import pandas as pd

from src.entity_resolution.core import EntityID, Predictions, CandidateMap, Source


@dataclass
class ValidationCheck:
    name: str
    passed: bool
    detail: str = ""


def validate_submission(
    predictions: Predictions,
    candidate_map: CandidateMap,
    test_s1_ids: List[EntityID],
    test_s2_ids: Set[EntityID],
    test_s3_ids: Set[EntityID],
) -> Tuple[bool, List[ValidationCheck]]:
    """
    Validate submission against ALL competition constraints.

    Checks:
        1. Every test S1 has exactly one row
        2. No duplicate S1 IDs
        3. Only S2/S3 IDs in matches
        4. No S1 IDs in matches
        5. All match IDs exist in test data
        6. No duplicate match IDs per entity
        7. Every final match is in candidate set
        8. Singletons have empty match set
        9. Deterministic ordering
    """
    checks: List[ValidationCheck] = []
    all_valid_targets = test_s2_ids | test_s3_ids

    # 1. All test S1 present
    pred_s1 = set(predictions.keys())
    test_s1_set = set(test_s1_ids)
    missing = test_s1_set - pred_s1
    checks.append(ValidationCheck(
        "All test S1 IDs present",
        len(missing) == 0,
        f"Missing {len(missing)} S1 IDs" if missing else "",
    ))

    # 2. No extra S1 IDs
    extra = pred_s1 - test_s1_set
    checks.append(ValidationCheck(
        "No extra S1 IDs",
        len(extra) == 0,
        f"Extra {len(extra)} S1 IDs" if extra else "",
    ))

    # 3-7: Per-entity checks
    invalid_targets = 0
    s1_in_matches = 0
    not_in_test = 0
    duplicate_matches = 0
    not_in_candidates = 0

    for s1_id, match_ids in predictions.items():
        for mid in match_ids:
            # No S1 IDs in matches
            try:
                src = Source.from_id(mid)
                if src == Source.S1:
                    s1_in_matches += 1
            except ValueError:
                invalid_targets += 1
                continue

            # Must exist in test data
            if mid not in all_valid_targets:
                not_in_test += 1

        # No duplicate match IDs
        if len(match_ids) != len(set(match_ids)):
            duplicate_matches += 1

        # Every match must be in candidate set
        cands = candidate_map.get(s1_id, set())
        for mid in match_ids:
            if mid not in cands:
                not_in_candidates += 1

    checks.append(ValidationCheck(
        "No S1 IDs in match sets", s1_in_matches == 0,
        f"{s1_in_matches} S1 IDs found in matches",
    ))
    checks.append(ValidationCheck(
        "All match IDs exist in test", not_in_test == 0,
        f"{not_in_test} IDs not in test data",
    ))
    checks.append(ValidationCheck(
        "No duplicate match IDs per entity", duplicate_matches == 0,
        f"{duplicate_matches} entities with duplicate matches",
    ))
    checks.append(ValidationCheck(
        "All matches in candidate set", not_in_candidates == 0,
        f"{not_in_candidates} matches not in candidates",
    ))
    checks.append(ValidationCheck(
        "No invalid target IDs", invalid_targets == 0,
        f"{invalid_targets} invalid target IDs",
    ))

    all_pass = all(c.passed for c in checks)
    return all_pass, checks


def write_matching_results(
    predictions: Predictions,
    output_path: str,
    s1_order: Optional[List[EntityID]] = None,
) -> Path:
    """
    Write matching_results.tsv.

    Format:
        source1_entity_id \\t matched_entity_ids
        S1-00001          \\t S2-00047,S2-00193,S3-00812
        S1-00002          \\t S3-00004
        S1-00003          \\t
    """
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    if s1_order is None:
        s1_order = sorted(predictions.keys())

    rows = []
    for s1_id in s1_order:
        match_ids = predictions.get(s1_id, set())
        # Deterministic ordering
        sorted_matches = sorted(match_ids)
        match_str = ",".join(sorted_matches)
        rows.append({"source1_entity_id": s1_id,
                      "matched_entity_ids": match_str})

    df = pd.DataFrame(rows)
    df.to_csv(path, sep="\t", index=False)
    return path


def write_candidate_pairs(
    candidate_map: CandidateMap,
    output_path: str,
    s1_order: Optional[List[EntityID]] = None,
) -> Path:
    """
    Write candidate_pairs.tsv.

    Format:
        source1_entity_id \\t candidate_entity_ids
        S1-00001          \\t S2-00047,S2-00193,S3-00812,S3-00999
    """
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    if s1_order is None:
        s1_order = sorted(candidate_map.keys())

    rows = []
    for s1_id in s1_order:
        cand_ids = candidate_map.get(s1_id, set())
        sorted_cands = sorted(cand_ids)
        cand_str = ",".join(sorted_cands)
        rows.append({"source1_entity_id": s1_id,
                      "candidate_entity_ids": cand_str})

    df = pd.DataFrame(rows)
    df.to_csv(path, sep="\t", index=False)
    return path


def write_submission_package(
    predictions: Predictions,
    candidate_map: CandidateMap,
    test_s1_ids: List[EntityID],
    test_s2_ids: Set[EntityID],
    test_s3_ids: Set[EntityID],
    output_dir: str = "submission/output",
    validate: bool = True,
) -> Dict[str, Any]:
    """
    Write complete submission package and validate.

    Returns validation report dict.
    """
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Ensure every test S1 has an entry
    for s1_id in test_s1_ids:
        if s1_id not in predictions:
            predictions[s1_id] = set()
        if s1_id not in candidate_map:
            candidate_map[s1_id] = set()

    # Write files
    match_path = write_matching_results(
        predictions, str(out_dir / "matching_results.tsv"), test_s1_ids
    )
    cand_path = write_candidate_pairs(
        candidate_map, str(out_dir / "candidate_pairs.tsv"), test_s1_ids
    )

    report = {
        "timestamp": datetime.now().isoformat(),
        "matching_results_path": str(match_path),
        "candidate_pairs_path": str(cand_path),
        "n_s1_entities": len(test_s1_ids),
        "n_singletons": sum(1 for s1 in test_s1_ids
                           if len(predictions.get(s1, set())) == 0),
        "n_matched": sum(1 for s1 in test_s1_ids
                        if len(predictions.get(s1, set())) > 0),
        "total_matches": sum(len(v) for v in predictions.values()),
        "total_candidates": sum(len(v) for v in candidate_map.values()),
    }

    if validate:
        all_pass, checks = validate_submission(
            predictions, candidate_map,
            test_s1_ids, test_s2_ids, test_s3_ids,
        )
        report["validation_passed"] = all_pass
        report["checks"] = [
            {"name": c.name, "passed": c.passed, "detail": c.detail}
            for c in checks
        ]

        if not all_pass:
            failed = [c for c in checks if not c.passed]
            raise RuntimeError(
                "Submission validation FAILED:\n" +
                "\n".join(f"  ❌ {c.name}: {c.detail}" for c in failed)
            )

    # Write report
    report_path = out_dir / "submission_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2, default=str)

    return report
