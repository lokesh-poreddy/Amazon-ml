"""
src/experiment_tracker.py
Amazon ML Challenge 2026 — Lightweight experiment logger.

Tracks every experiment to logs/experiments.csv.

Every experiment must answer:
  1. WHAT did we change?
  2. WHY did we change it?
  3. WHAT happened?
  4. Do we KEEP it?

No MLOps platform needed — a CSV is enough for 72 hours.
"""

from __future__ import annotations

import csv
import os
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

from src.logging_utils import get_logger

logger = get_logger(__name__)

_FIELDNAMES = [
    "experiment_id",
    "timestamp",
    "hypothesis",
    "task",
    "feature_version",
    "split_version",
    "model",
    "params",
    "cv_mean",
    "cv_std",
    "runtime_seconds",
    "memory_mb",
    "notes",
    "decision",
]


@dataclass
class ExperimentRecord:
    """A single row in the experiment log."""

    experiment_id: str
    hypothesis: str
    task: str = "UNKNOWN"
    feature_version: str = "v0"
    split_version: str = "v0"
    model: str = ""
    params: str = ""            # JSON string or short description
    cv_mean: Optional[float] = None
    cv_std: Optional[float] = None
    runtime_seconds: Optional[float] = None
    memory_mb: Optional[float] = None
    notes: str = ""
    decision: str = "PENDING"  # KEEP | REJECT | PENDING | REFERENCE
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat(timespec="seconds"))


class ExperimentTracker:
    """
    Append experiments to a CSV log file.

    Usage
    -----
    tracker = ExperimentTracker("logs/experiments.csv")
    tracker.log(ExperimentRecord(
        experiment_id="E001",
        hypothesis="Baseline LightGBM",
        model="lgbm",
        cv_mean=0.421,
        cv_std=0.012,
        decision="REFERENCE",
    ))
    tracker.show()
    """

    def __init__(self, path: str = "logs/experiments.csv") -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._ensure_header()

    def _ensure_header(self) -> None:
        if not self.path.exists() or self.path.stat().st_size == 0:
            with open(self.path, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=_FIELDNAMES)
                writer.writeheader()

    def log(self, record: ExperimentRecord) -> None:
        """Append one experiment to the CSV."""
        row = {k: getattr(record, k, "") for k in _FIELDNAMES}
        # Round floats for readability
        for k in ("cv_mean", "cv_std", "runtime_seconds", "memory_mb"):
            v = row.get(k)
            if v is not None:
                try:
                    row[k] = round(float(v), 6)
                except (TypeError, ValueError):
                    pass
        with open(self.path, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=_FIELDNAMES)
            writer.writerow(row)
        logger.info(
            "Experiment logged: [%s] %s | %s=%.6f | %s",
            record.experiment_id, record.hypothesis,
            "cv_mean", record.cv_mean or 0.0, record.decision,
        )

    def load(self) -> pd.DataFrame:
        """Load all experiments as a DataFrame."""
        if not self.path.exists():
            return pd.DataFrame(columns=_FIELDNAMES)
        return pd.read_csv(self.path)

    def show(self, n: int = 20) -> pd.DataFrame:
        """Display the latest N experiments."""
        df = self.load()
        if df.empty:
            logger.info("No experiments logged yet.")
            return df
        cols = ["experiment_id", "hypothesis", "model", "cv_mean", "cv_std", "decision"]
        display_cols = [c for c in cols if c in df.columns]
        return df[display_cols].tail(n)

    def best_experiment(self, metric_direction: str = "minimize") -> Optional[pd.Series]:
        """Return the experiment with the best cv_mean."""
        df = self.load().dropna(subset=["cv_mean"])
        if df.empty:
            return None
        if metric_direction == "minimize":
            return df.loc[df["cv_mean"].idxmin()]
        return df.loc[df["cv_mean"].idxmax()]

    def comparison_table(self) -> pd.DataFrame:
        """Return a formatted comparison table for reporting."""
        df = self.load()
        if df.empty:
            return df
        return df[["experiment_id", "model", "feature_version", "cv_mean", "cv_std",
                   "runtime_seconds", "decision"]].sort_values("cv_mean")


# ─────────────────────────────────────────────────────────────────
# CONVENIENCE SINGLETON
# ─────────────────────────────────────────────────────────────────
_tracker: Optional[ExperimentTracker] = None


def get_tracker(path: str = "logs/experiments.csv") -> ExperimentTracker:
    global _tracker
    if _tracker is None:
        _tracker = ExperimentTracker(path)
    return _tracker
