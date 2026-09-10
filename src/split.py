"""
src/split.py
Amazon ML Challenge 2026 — Flexible validation splitting.

Supports:
  KFold | StratifiedKFold | GroupKFold | StratifiedGroupKFold
  TimeSeriesSplit | Single Holdout

The correct split strategy depends on the DATA GENERATION PROCESS:
  - IID rows          → KFold or StratifiedKFold
  - Repeated entities → GroupKFold (prevent entity leakage)
  - Temporal data     → TimeSeriesSplit (future cannot inform past)
  - Imbalanced classes→ StratifiedKFold

Never choose a split blindly. Document the choice.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Generator, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.model_selection import (
    GroupKFold,
    KFold,
    StratifiedGroupKFold,
    StratifiedKFold,
    TimeSeriesSplit,
    train_test_split,
)

from src.logging_utils import get_logger

logger = get_logger(__name__)


# ─────────────────────────────────────────────────────────────────
# SPLIT RESULT
# ─────────────────────────────────────────────────────────────────
@dataclass
class SplitInfo:
    """Summary of a single fold."""
    fold: int
    train_idx: np.ndarray
    val_idx: np.ndarray
    n_train: int
    n_val: int
    strategy: str

    @property
    def val_fraction(self) -> float:
        return self.n_val / (self.n_train + self.n_val)


# ─────────────────────────────────────────────────────────────────
# MAKE_SPLITS — main entry point
# ─────────────────────────────────────────────────────────────────
def make_splits(
    df: pd.DataFrame,
    strategy: str = "kfold",
    n_folds: int = 5,
    target_col: Optional[str] = None,
    group_col: Optional[str] = None,
    time_col: Optional[str] = None,
    shuffle: bool = True,
    seed: int = 42,
    holdout_fraction: float = 0.2,
) -> List[SplitInfo]:
    """
    Generate train/validation index splits for cross-validation.

    Parameters
    ----------
    df : pd.DataFrame
        The full training dataset.
    strategy : str
        "kfold" | "stratified" | "group" | "stratified_group" |
        "timeseries" | "holdout"
    n_folds : int
        Number of folds (ignored for "holdout").
    target_col : str | None
        Required for "stratified" and "stratified_group".
    group_col : str | None
        Required for "group" and "stratified_group".
    time_col : str | None
        Required for "timeseries" — data will be sorted by this column.
    shuffle : bool
        Shuffle before splitting (not applicable to timeseries).
    seed : int
        Random state.
    holdout_fraction : float
        Fraction held out for "holdout" strategy.

    Returns
    -------
    list[SplitInfo]
        One SplitInfo per fold.

    Raises
    ------
    ValueError
        If required columns are missing for the chosen strategy.
    """
    _validate_strategy_requirements(
        strategy, df, target_col, group_col, time_col
    )

    n = len(df)
    idx = np.arange(n)
    splits: List[SplitInfo] = []

    if strategy == "kfold":
        kf = KFold(n_splits=n_folds, shuffle=shuffle, random_state=seed)
        for fold, (tr, va) in enumerate(kf.split(idx)):
            splits.append(_make_si(fold, tr, va, strategy))

    elif strategy == "stratified":
        y_bins = _stratify_bins(df[target_col])
        skf = StratifiedKFold(n_splits=n_folds, shuffle=shuffle, random_state=seed)
        for fold, (tr, va) in enumerate(skf.split(idx, y_bins)):
            splits.append(_make_si(fold, tr, va, strategy))

    elif strategy == "group":
        groups = df[group_col].values
        gkf = GroupKFold(n_splits=n_folds)
        for fold, (tr, va) in enumerate(gkf.split(idx, groups=groups)):
            splits.append(_make_si(fold, tr, va, strategy))

    elif strategy == "stratified_group":
        y_bins = _stratify_bins(df[target_col])
        groups = df[group_col].values
        sgkf = StratifiedGroupKFold(n_splits=n_folds, shuffle=shuffle, random_state=seed)
        for fold, (tr, va) in enumerate(sgkf.split(idx, y_bins, groups=groups)):
            splits.append(_make_si(fold, tr, va, strategy))

    elif strategy == "timeseries":
        # Sort data chronologically
        sorted_idx = df.sort_values(time_col).index.values
        df_sorted_pos = {orig_idx: pos for pos, orig_idx in enumerate(sorted_idx)}
        pos_arr = np.array([df_sorted_pos[i] for i in df.index])

        tss = TimeSeriesSplit(n_splits=n_folds)
        for fold, (tr_pos, va_pos) in enumerate(tss.split(pos_arr)):
            # Map positions back to original dataframe positions
            tr_orig = np.where(np.isin(pos_arr, tr_pos))[0]
            va_orig = np.where(np.isin(pos_arr, va_pos))[0]
            splits.append(_make_si(fold, tr_orig, va_orig, strategy))

    elif strategy == "holdout":
        if group_col:
            # Group-aware holdout: all rows of a group go to same split
            groups = df[group_col].unique()
            rng = np.random.RandomState(seed)
            val_groups = set(rng.choice(groups, size=max(1, int(len(groups) * holdout_fraction)), replace=False))
            tr = np.where(~df[group_col].isin(val_groups))[0]
            va = np.where(df[group_col].isin(val_groups))[0]
        else:
            tr, va = train_test_split(idx, test_size=holdout_fraction, random_state=seed, shuffle=shuffle)
        splits.append(_make_si(0, tr, va, strategy))

    else:
        raise ValueError(
            f"Unknown split strategy: '{strategy}'. "
            "Choose from: kfold | stratified | group | stratified_group | timeseries | holdout"
        )

    logger.info(
        "Split strategy: %s | Folds: %d | Avg train size: %.0f | Avg val size: %.0f",
        strategy, len(splits),
        np.mean([s.n_train for s in splits]),
        np.mean([s.n_val for s in splits]),
    )
    return splits


# ─────────────────────────────────────────────────────────────────
# SAVE / LOAD SPLITS  (for reproducibility)
# ─────────────────────────────────────────────────────────────────
def save_splits(splits: List[SplitInfo], path: Union[str, Path]) -> None:
    """Save split indices to an .npz file for later reuse."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    arrays: dict = {}
    for si in splits:
        arrays[f"fold_{si.fold}_train"] = si.train_idx
        arrays[f"fold_{si.fold}_val"]   = si.val_idx
    np.savez(path, strategy=np.array([splits[0].strategy]), **arrays)
    logger.info("Saved %d splits → %s", len(splits), path)


def load_splits(path: Union[str, Path]) -> List[SplitInfo]:
    """Load split indices from an .npz file."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Split file not found: {path}")
    data = np.load(path, allow_pickle=True)
    strategy = str(data["strategy"][0])

    splits: List[SplitInfo] = []
    fold = 0
    while f"fold_{fold}_train" in data:
        tr = data[f"fold_{fold}_train"]
        va = data[f"fold_{fold}_val"]
        splits.append(_make_si(fold, tr, va, strategy))
        fold += 1
    logger.info("Loaded %d splits from %s", len(splits), path)
    return splits


# ─────────────────────────────────────────────────────────────────
# PRIVATE HELPERS
# ─────────────────────────────────────────────────────────────────
def _make_si(fold: int, tr: np.ndarray, va: np.ndarray, strategy: str) -> SplitInfo:
    return SplitInfo(
        fold=fold,
        train_idx=tr,
        val_idx=va,
        n_train=len(tr),
        n_val=len(va),
        strategy=strategy,
    )


def _stratify_bins(series: pd.Series, n_bins: int = 10) -> pd.Series:
    """
    Convert a continuous target to discrete bins for stratified splitting.

    Falls back to raw values if the series is already categorical/integer.
    """
    if series.dtype.kind in ("i", "u", "b") or not series.dtype.kind == "f":
        return series.astype(str)
    try:
        return pd.cut(series, bins=n_bins, labels=False, duplicates="drop").astype(str)
    except Exception:
        return series.astype(str)


def _validate_strategy_requirements(
    strategy: str,
    df: pd.DataFrame,
    target_col: Optional[str],
    group_col: Optional[str],
    time_col: Optional[str],
) -> None:
    if strategy in ("stratified", "stratified_group"):
        if target_col is None or target_col not in df.columns:
            raise ValueError(
                f"Strategy '{strategy}' requires target_col='{target_col}' "
                f"but it is not in the DataFrame. Available: {list(df.columns[:10])}"
            )
    if strategy in ("group", "stratified_group", "holdout") and group_col:
        if group_col not in df.columns:
            raise ValueError(
                f"Strategy '{strategy}' requires group_col='{group_col}' "
                f"but it is not in the DataFrame."
            )
    if strategy == "timeseries":
        if time_col is None or time_col not in df.columns:
            raise ValueError(
                f"Strategy 'timeseries' requires time_col='{time_col}' "
                f"but it is not in the DataFrame."
            )
