"""
experiments/tracker.py — Experiment and submission ledger.

Tracks every run and submission with full reproducibility metadata.
"""

from __future__ import annotations

import csv
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional


class ExperimentTracker:
    """
    Tracks experiments and submissions.

    Maintains:
        experiments.csv - all experiment runs
        submissions.csv - leaderboard submissions
    """

    EXPERIMENT_FIELDS = [
        "experiment_id", "timestamp", "git_commit",
        "blocking_version", "feature_version", "model_version",
        "threshold", "candidate_recall", "candidate_count",
        "entity_f05", "precision", "recall",
        "n_entities", "n_singletons", "false_merges", "missed_matches",
        "runtime_sec", "memory_mb", "observations",
    ]

    SUBMISSION_FIELDS = [
        "submission_id", "timestamp", "experiment_id",
        "blocking_version", "feature_version", "model_version",
        "threshold", "validation_f05", "public_score",
        "n_entities", "n_singletons", "total_matches",
        "submission_number", "observations",
    ]

    def __init__(self, log_dir: str = "logs"):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.exp_path = self.log_dir / "experiments.csv"
        self.sub_path = self.log_dir / "submissions.csv"

        # Initialize files if they don't exist
        if not self.exp_path.exists():
            self._write_header(self.exp_path, self.EXPERIMENT_FIELDS)
        if not self.sub_path.exists():
            self._write_header(self.sub_path, self.SUBMISSION_FIELDS)

    def _write_header(self, path: Path, fields: List[str]):
        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()

    def log_experiment(self, **kwargs) -> str:
        """Log an experiment run. Returns experiment_id."""
        exp_id = kwargs.get(
            "experiment_id",
            f"E-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
        )
        kwargs["experiment_id"] = exp_id
        kwargs.setdefault("timestamp", datetime.now().isoformat())

        row = {field: kwargs.get(field, "") for field in self.EXPERIMENT_FIELDS}

        with open(self.exp_path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=self.EXPERIMENT_FIELDS)
            writer.writerow(row)

        return exp_id

    def log_submission(self, **kwargs) -> str:
        """Log a submission. Returns submission_id."""
        sub_id = kwargs.get(
            "submission_id",
            f"SUB-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
        )
        kwargs["submission_id"] = sub_id
        kwargs.setdefault("timestamp", datetime.now().isoformat())

        row = {field: kwargs.get(field, "") for field in self.SUBMISSION_FIELDS}

        with open(self.sub_path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=self.SUBMISSION_FIELDS)
            writer.writerow(row)

        return sub_id

    def get_experiments(self) -> List[Dict[str, str]]:
        """Load all experiments."""
        if not self.exp_path.exists():
            return []
        with open(self.exp_path) as f:
            return list(csv.DictReader(f))

    def get_submissions(self) -> List[Dict[str, str]]:
        """Load all submissions."""
        if not self.sub_path.exists():
            return []
        with open(self.sub_path) as f:
            return list(csv.DictReader(f))

    def submission_count_today(self) -> int:
        """Count submissions made today."""
        today = datetime.now().strftime("%Y-%m-%d")
        subs = self.get_submissions()
        return sum(1 for s in subs if s.get("timestamp", "").startswith(today))

    def can_submit(self, max_per_day: int = 5) -> bool:
        """Check if we're within submission limit."""
        return self.submission_count_today() < max_per_day
