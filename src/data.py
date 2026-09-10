"""
src/data.py
Amazon ML Challenge 2026 — Data discovery and multi-format loading.

Key design decisions:
  - discover_files() recursively scans a directory for all data formats,
    making reasonable guesses about train/test files without hard-coding names.
  - Each loader (CSV, Parquet, JSON, JSONL) raises explicit errors with
    helpful diagnostics instead of silent failures.
  - debug_sample() applies controlled subsampling to avoid accidentally
    training on a debug sample in the final run.
"""

from __future__ import annotations

import hashlib
import io
import json
import os
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

from src.logging_utils import get_logger

logger = get_logger(__name__)


# ─────────────────────────────────────────────────────────────────
# FILE DISCOVERY
# ─────────────────────────────────────────────────────────────────
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".gif", ".bmp", ".tiff"}

TRAIN_KEYWORDS = {"train", "training", "trn"}
TEST_KEYWORDS  = {"test", "testing", "tst", "val", "valid", "validation", "eval", "sample_submission"}


@dataclass
class DiscoveredFiles:
    """Structured inventory of files found in the data directory."""

    root: Path
    csv: List[Path] = field(default_factory=list)
    parquet: List[Path] = field(default_factory=list)
    json: List[Path] = field(default_factory=list)
    jsonl: List[Path] = field(default_factory=list)
    txt: List[Path] = field(default_factory=list)
    zip: List[Path] = field(default_factory=list)
    images: List[Path] = field(default_factory=list)
    other: List[Path] = field(default_factory=list)

    probable_train: List[Path] = field(default_factory=list)
    probable_test: List[Path] = field(default_factory=list)
    probable_submission: List[Path] = field(default_factory=list)

    def summary(self) -> str:
        lines = [
            f"\n{'='*60}",
            f"  FILE DISCOVERY — {self.root}",
            f"{'='*60}",
            f"  CSV       : {len(self.csv)}",
            f"  Parquet   : {len(self.parquet)}",
            f"  JSON      : {len(self.json)}",
            f"  JSONL     : {len(self.jsonl)}",
            f"  TXT       : {len(self.txt)}",
            f"  ZIP       : {len(self.zip)}",
            f"  Images    : {len(self.images)}",
            f"  Other     : {len(self.other)}",
            "",
            "  GUESSES (review before using):",
        ]
        if self.probable_train:
            for p in self.probable_train:
                lines.append(f"    TRAIN  → {p.relative_to(self.root)}")
        else:
            lines.append("    TRAIN  → ⚠️  Could not identify — set manually")
        if self.probable_test:
            for p in self.probable_test:
                lines.append(f"    TEST   → {p.relative_to(self.root)}")
        else:
            lines.append("    TEST   → ⚠️  Could not identify — set manually")
        if self.probable_submission:
            for p in self.probable_submission:
                lines.append(f"    SUBMIT → {p.relative_to(self.root)}")
        lines.append("=" * 60)
        return "\n".join(lines)


SUBMISSION_KEYWORDS = {"sample_submission", "submission_sample", "submission"}


def _score_candidate(
    path: Path,
    all_tabular: List[Path],
) -> Dict[str, Any]:
    """
    Score a tabular file as a train/test/submission candidate.

    Returns a dict with:
      train_score, test_score, submission_score, confidence descriptors.

    Scoring factors:
      - Filename keyword match (primary signal)
      - "sample" prefix (strong indicator of sample_submission)
      - File size (train is usually the largest tabular file)
      - Not a known submission file (adjusts test/train scores down)
    """
    stem = path.stem.lower()
    size = path.stat().st_size if path.exists() else 0
    max_size = max((p.stat().st_size for p in all_tabular if p.exists()), default=1)
    size_ratio = size / max(max_size, 1)

    train_score = 0.0
    test_score  = 0.0
    sub_score   = 0.0

    # Strong keyword signals
    if any(kw == stem or kw in stem for kw in TRAIN_KEYWORDS):
        train_score += 1.0
    if any(kw == stem or kw in stem for kw in TEST_KEYWORDS - {"val", "valid", "validation", "eval"}):
        test_score += 1.0
    if any(kw == stem or kw in stem for kw in {"val", "valid", "validation", "eval"}):
        # validation files could be either train-like or test-like
        train_score += 0.3
        test_score  += 0.3

    # Submission signals
    if "submission" in stem:
        sub_score   += 1.0
        train_score -= 1.0
        test_score  -= 0.5
    if "sample" in stem:
        sub_score   += 0.5
        train_score -= 0.5

    # Size heuristic: train is usually the biggest file
    train_score += 0.4 * size_ratio
    test_score  += 0.2 * (1.0 - size_ratio)   # test tends to be smaller

    return {
        "path":         path,
        "train_score":  round(train_score, 3),
        "test_score":   round(test_score, 3),
        "sub_score":    round(sub_score, 3),
        "size_bytes":   size,
    }


def discover_files(root: Union[str, Path]) -> DiscoveredFiles:
    """
    Recursively scan a directory for all competition-relevant files.

    Parameters
    ----------
    root : str | Path
        Root directory to scan (typically ``data/raw``).

    Returns
    -------
    DiscoveredFiles
        Structured inventory with heuristic train/test guesses.
    """
    root = Path(root)
    if not root.exists():
        raise FileNotFoundError(
            f"Data directory not found: {root}\n"
            "Create it and place your competition data inside."
        )

    result = DiscoveredFiles(root=root)

    all_paths: List[Path] = sorted(
        p for p in root.rglob("*") if p.is_file()
    )

    tabular_paths: List[Path] = []

    for p in all_paths:
        ext = p.suffix.lower()
        if ext == ".csv":
            result.csv.append(p)
            tabular_paths.append(p)
        elif ext in (".parquet", ".pq"):
            result.parquet.append(p)
            tabular_paths.append(p)
        elif ext == ".json":
            result.json.append(p)
            tabular_paths.append(p)
        elif ext in (".jsonl", ".ndjson"):
            result.jsonl.append(p)
            tabular_paths.append(p)
        elif ext == ".txt":
            result.txt.append(p)
        elif ext == ".zip":
            result.zip.append(p)
        elif ext in IMAGE_EXTENSIONS:
            result.images.append(p)
        else:
            result.other.append(p)

    # ── Score and rank all tabular files ─────────────────────────
    scored = [_score_candidate(p, tabular_paths) for p in tabular_paths]

    train_candidates = sorted(
        [s for s in scored if s["train_score"] > 0],
        key=lambda s: s["train_score"], reverse=True,
    )
    test_candidates = sorted(
        [s for s in scored if s["test_score"] > 0],
        key=lambda s: s["test_score"], reverse=True,
    )
    sub_candidates = sorted(
        [s for s in scored if s["sub_score"] > 0.5],
        key=lambda s: s["sub_score"], reverse=True,
    )

    result.probable_train      = [s["path"] for s in train_candidates[:2]]
    result.probable_test       = [s["path"] for s in test_candidates[:2]
                                  if s["path"] not in result.probable_train]
    result.probable_submission = [s["path"] for s in sub_candidates[:2]]

    # Log with confidence descriptors
    for s in train_candidates[:2]:
        logger.info("Candidate TRAIN: %s (score=%.2f)", s["path"].name, s["train_score"])
    for s in test_candidates[:2]:
        if s["path"] not in result.probable_train:
            logger.info("Candidate TEST:  %s (score=%.2f)", s["path"].name, s["test_score"])
    for s in sub_candidates[:2]:
        logger.info("Candidate SUBMISSION: %s (score=%.2f)", s["path"].name, s["sub_score"])

    logger.info(result.summary())
    return result


# ─────────────────────────────────────────────────────────────────
# DATA LOADERS
# ─────────────────────────────────────────────────────────────────
def load_csv(
    path: Union[str, Path],
    dtype: Optional[Dict[str, Any]] = None,
    low_memory: bool = False,
    encoding: str = "utf-8",
    **kwargs: Any,
) -> pd.DataFrame:
    """
    Load a CSV file into a DataFrame with explicit error handling.

    Parameters
    ----------
    path : str | Path
    dtype : dict | None
        Column-level dtype overrides.
    low_memory : bool
        Passed to pd.read_csv — set True for very large files.
    encoding : str
        File encoding. Falls back to latin-1 if utf-8 fails.
    **kwargs
        Additional arguments passed to pd.read_csv.

    Returns
    -------
    pd.DataFrame
    """
    path = Path(path)
    _require_file(path)

    logger.info("Loading CSV: %s", path)
    try:
        df = pd.read_csv(path, dtype=dtype, low_memory=low_memory, encoding=encoding, **kwargs)
    except UnicodeDecodeError:
        logger.warning("UTF-8 decode failed for %s — retrying with latin-1.", path)
        df = pd.read_csv(path, dtype=dtype, low_memory=low_memory, encoding="latin-1", **kwargs)

    logger.info("Loaded CSV: %d rows × %d cols from %s", len(df), df.shape[1], path.name)
    return df


def load_parquet(
    path: Union[str, Path],
    columns: Optional[List[str]] = None,
    **kwargs: Any,
) -> pd.DataFrame:
    """
    Load a Parquet file.

    Parameters
    ----------
    path : str | Path
    columns : list[str] | None
        Subset of columns to load (memory optimisation).
    """
    path = Path(path)
    _require_file(path)
    logger.info("Loading Parquet: %s", path)
    df = pd.read_parquet(path, columns=columns, **kwargs)
    logger.info("Loaded Parquet: %d rows × %d cols from %s", len(df), df.shape[1], path.name)
    return df


def load_json(path: Union[str, Path], **kwargs: Any) -> pd.DataFrame:
    """
    Load a JSON file (records, columns, or list-of-dicts orientation).

    Attempts ``pd.read_json`` first; falls back to ``json.load``.
    """
    path = Path(path)
    _require_file(path)
    logger.info("Loading JSON: %s", path)
    try:
        df = pd.read_json(path, **kwargs)
    except ValueError:
        with open(path, "r", encoding="utf-8") as f:
            raw = json.load(f)
        if isinstance(raw, list):
            df = pd.DataFrame(raw)
        elif isinstance(raw, dict):
            df = pd.DataFrame([raw])
        else:
            raise ValueError(
                f"Unsupported JSON structure in {path}. "
                "Expected a list or dict at the top level."
            )
    logger.info("Loaded JSON: %d rows × %d cols from %s", len(df), df.shape[1], path.name)
    return df


def load_jsonl(
    path: Union[str, Path],
    max_lines: Optional[int] = None,
) -> pd.DataFrame:
    """
    Load a JSON Lines (JSONL) file.

    Parameters
    ----------
    path : str | Path
    max_lines : int | None
        If set, only the first N lines are loaded (useful for inspection).
    """
    path = Path(path)
    _require_file(path)
    logger.info("Loading JSONL: %s (max_lines=%s)", path, max_lines)

    records: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if max_lines is not None and i >= max_lines:
                break
            line = line.strip()
            if line:
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError as exc:
                    raise ValueError(f"Invalid JSON on line {i+1} of {path}: {exc}") from exc

    df = pd.DataFrame(records)
    logger.info("Loaded JSONL: %d rows × %d cols from %s", len(df), df.shape[1], path.name)
    return df


def load_zip(
    path: Union[str, Path],
    inner_file: Optional[str] = None,
    loader: str = "csv",
    **kwargs: Any,
) -> pd.DataFrame:
    """
    Load a DataFrame from a file inside a ZIP archive.

    Parameters
    ----------
    path : str | Path
        Path to the .zip file.
    inner_file : str | None
        Name of the file inside the ZIP. If None, the first file is used.
    loader : str
        "csv" | "parquet" | "json" | "jsonl"
    """
    path = Path(path)
    _require_file(path)

    with zipfile.ZipFile(path) as zf:
        names = zf.namelist()
        if not names:
            raise ValueError(f"ZIP file is empty: {path}")
        target = inner_file if inner_file else names[0]
        if target not in names:
            raise ValueError(
                f"'{target}' not found in {path}. Available: {names}"
            )
        logger.info("Loading '%s' from ZIP: %s", target, path)
        data = zf.read(target)

    buf = io.BytesIO(data)
    if loader == "csv":
        return pd.read_csv(buf, **kwargs)
    elif loader == "parquet":
        return pd.read_parquet(buf, **kwargs)
    elif loader == "json":
        return pd.read_json(buf, **kwargs)
    elif loader == "jsonl":
        # JSONL is not directly supported by pd.read_json with BytesIO reliably;
        # decode and parse line-by-line.
        text = buf.read().decode("utf-8", errors="replace")
        records: List[Dict[str, Any]] = []
        for i, line in enumerate(text.splitlines()):
            line = line.strip()
            if line:
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError as exc:
                    raise ValueError(
                        f"Invalid JSON on line {i+1} inside ZIP '{path}': {exc}"
                    ) from exc
        return pd.DataFrame(records)
    else:
        raise ValueError(
            f"Unsupported loader for ZIP: '{loader}'. "
            "Choose from: csv | parquet | json | jsonl"
        )


# ─────────────────────────────────────────────────────────────────
# AUTO-LOADER (infer format from extension)
# ─────────────────────────────────────────────────────────────────
def load_data(path: Union[str, Path], **kwargs: Any) -> pd.DataFrame:
    """
    Automatically detect file format and load into a DataFrame.

    Supports: .csv, .parquet, .pq, .json, .jsonl/.ndjson, .zip

    Parameters
    ----------
    path : str | Path
    **kwargs
        Forwarded to the specific loader.

    Returns
    -------
    pd.DataFrame
    """
    path = Path(path)
    ext = path.suffix.lower()
    if ext == ".csv":
        return load_csv(path, **kwargs)
    elif ext in (".parquet", ".pq"):
        return load_parquet(path, **kwargs)
    elif ext == ".json":
        return load_json(path, **kwargs)
    elif ext in (".jsonl", ".ndjson"):
        return load_jsonl(path, **kwargs)
    elif ext == ".zip":
        return load_zip(path, **kwargs)
    else:
        raise ValueError(
            f"Cannot determine loader for extension '{ext}' ({path}).\n"
            "Use load_csv(), load_parquet(), load_json(), or load_jsonl() directly."
        )


# ─────────────────────────────────────────────────────────────────
# DEBUG SUBSAMPLING
# ─────────────────────────────────────────────────────────────────
def debug_sample(
    df: pd.DataFrame,
    n: int = 2000,
    seed: int = 42,
    mode: str = "debug",
) -> pd.DataFrame:
    """
    Return a stratified/random subsample for rapid iteration.

    WARNING: Do NOT call this in final_run mode.

    Parameters
    ----------
    df : pd.DataFrame
    n : int
        Maximum rows in the sample.
    seed : int
    mode : str
        Label for logging. Use "debug" to make it obvious.

    Returns
    -------
    pd.DataFrame
    """
    if mode == "final":
        logger.warning("debug_sample() called in 'final' mode — returning full dataset.")
        return df
    if len(df) <= n:
        return df
    sample = df.sample(n=n, random_state=seed).reset_index(drop=True)
    logger.warning(
        "⚠️  DEBUG SAMPLE ACTIVE: %d / %d rows (seed=%d). "
        "DO NOT USE FOR FINAL SUBMISSION.",
        n, len(df), seed,
    )
    return sample


# ─────────────────────────────────────────────────────────────────
# IMAGE CACHE UTILITIES
# ─────────────────────────────────────────────────────────────────
def url_cache_key(url: str, prefix_len: int = 24) -> str:
    """
    Produce a stable filename-safe cache key for a URL.

    Parameters
    ----------
    url : str
    prefix_len : int
        Number of hex characters to use (default 24 → 96-bit key).

    Returns
    -------
    str
    """
    return hashlib.sha256(url.encode()).hexdigest()[:prefix_len]


# ─────────────────────────────────────────────────────────────────
# PRIVATE HELPERS
# ─────────────────────────────────────────────────────────────────
def _require_file(path: Path) -> None:
    """Raise a clear FileNotFoundError if path does not exist."""
    if not path.exists():
        raise FileNotFoundError(
            f"Data file not found: {path}\n"
            f"Checked location: {path.resolve()}\n"
            "Did you place the competition data in data/raw/ ?"
        )
