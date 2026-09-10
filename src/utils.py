"""
src/utils.py
Amazon ML Challenge 2026 — General-purpose utilities.

Covers:
  - seed_everything()        reproducible seeding across frameworks
  - get_device()             CUDA / MPS / CPU detection
  - Timer / timed()          wall-clock timing
  - memory_usage()           current process memory
  - reduce_mem_usage()       dtype downcasting for DataFrames
  - ensure_dir()             safe directory creation
  - save_json() / load_json_file()
  - now_str()                timestamp string for filenames
  - check_final_package()    pre-submission system check
"""

from __future__ import annotations

import gc
import json
import os
import random
import time
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Generator, List, Optional, Sequence, Union

import numpy as np
import pandas as pd

from src.logging_utils import get_logger

logger = get_logger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent


# ─────────────────────────────────────────────────────────────────
# REPRODUCIBILITY
# ─────────────────────────────────────────────────────────────────
def seed_everything(seed: int = 42) -> None:
    """
    Seed all common random-number sources for reproducibility.

    NOTE: Full bit-for-bit reproducibility also depends on hardware,
    CUDA settings, and framework versions. Setting seeds is necessary
    but not always sufficient for perfect determinism.

    Parameters
    ----------
    seed : int
        The seed value to apply everywhere.
    """
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    # PyTorch (optional — only if installed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            # Deterministic behaviour at cost of some speed
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        logger.debug("PyTorch seeds set (seed=%d).", seed)
    except ImportError:
        pass

    logger.info("Seeded everything with seed=%d.", seed)


# ─────────────────────────────────────────────────────────────────
# DEVICE DETECTION
# ─────────────────────────────────────────────────────────────────
def get_device() -> str:
    """
    Detect the best available compute device.

    Returns
    -------
    str
        One of: "cuda", "mps", "cpu"
    """
    try:
        import torch

        if torch.cuda.is_available():
            device = "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
    except ImportError:
        device = "cpu"

    logger.info("Compute device: %s", device)
    return device


# ─────────────────────────────────────────────────────────────────
# TIMING
# ─────────────────────────────────────────────────────────────────
class Timer:
    """Simple wall-clock timer."""

    def __init__(self, name: str = "block") -> None:
        self.name = name
        self._start: float = 0.0
        self.elapsed: float = 0.0

    def start(self) -> "Timer":
        self._start = time.perf_counter()
        return self

    def stop(self) -> float:
        self.elapsed = time.perf_counter() - self._start
        return self.elapsed

    def __enter__(self) -> "Timer":
        return self.start()

    def __exit__(self, *_: Any) -> None:
        self.stop()
        logger.info("⏱  [%s] took %.2f s", self.name, self.elapsed)


@contextmanager
def timed(name: str = "block") -> Generator[Timer, None, None]:
    """
    Context manager that logs wall-clock time.

    Usage
    -----
    with timed("feature engineering"):
        features = create_features(df)
    """
    t = Timer(name)
    t.start()
    try:
        yield t
    finally:
        t.stop()
        logger.info("⏱  [%s] took %.2f s", name, t.elapsed)


# ─────────────────────────────────────────────────────────────────
# MEMORY
# ─────────────────────────────────────────────────────────────────
def memory_usage_mb() -> float:
    """Return current process RSS memory in MB."""
    try:
        import psutil

        process = psutil.Process(os.getpid())
        return process.memory_info().rss / 1024 ** 2
    except ImportError:
        return float("nan")


def reduce_mem_usage(df: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
    """
    Downcast numeric columns to reduce DataFrame memory footprint.

    Warning: this changes dtypes in-place — only apply before serialization,
    not before validation where dtype precision matters.

    Parameters
    ----------
    df : pd.DataFrame
    verbose : bool
        If True, log before/after memory usage.

    Returns
    -------
    pd.DataFrame
        DataFrame with downcasted dtypes.
    """
    start_mem = df.memory_usage(deep=True).sum() / 1024 ** 2
    for col in df.columns:
        col_type = df[col].dtype
        if col_type in (object, "category"):
            continue
        c_min = df[col].min()
        c_max = df[col].max()
        if str(col_type).startswith("int"):
            if c_min >= np.iinfo(np.int8).min and c_max <= np.iinfo(np.int8).max:
                df[col] = df[col].astype(np.int8)
            elif c_min >= np.iinfo(np.int16).min and c_max <= np.iinfo(np.int16).max:
                df[col] = df[col].astype(np.int16)
            elif c_min >= np.iinfo(np.int32).min and c_max <= np.iinfo(np.int32).max:
                df[col] = df[col].astype(np.int32)
            else:
                df[col] = df[col].astype(np.int64)
        else:
            if c_min >= np.finfo(np.float32).min and c_max <= np.finfo(np.float32).max:
                df[col] = df[col].astype(np.float32)
            else:
                df[col] = df[col].astype(np.float64)

    end_mem = df.memory_usage(deep=True).sum() / 1024 ** 2
    if verbose:
        saved = 100 * (start_mem - end_mem) / (start_mem + 1e-9)
        logger.info(
            "Memory reduced: %.1f MB → %.1f MB (%.1f%% saved)",
            start_mem, end_mem, saved,
        )
    gc.collect()
    return df


# ─────────────────────────────────────────────────────────────────
# FILE HELPERS
# ─────────────────────────────────────────────────────────────────
def ensure_dir(path: Union[str, Path]) -> Path:
    """Create directory (and parents) if it does not exist."""
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def now_str(fmt: str = "%Y%m%d_%H%M%S") -> str:
    """Return current timestamp as a string safe for filenames."""
    return datetime.now().strftime(fmt)


def save_json(obj: Any, path: Union[str, Path], indent: int = 2) -> None:
    """Serialize obj to JSON at path, creating parent dirs as needed."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=indent, default=str)
    logger.debug("Saved JSON → %s", path)


def load_json_file(path: Union[str, Path]) -> Any:
    """Load and return a JSON file."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"JSON file not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# ─────────────────────────────────────────────────────────────────
# FINAL PACKAGE CHECK
# ─────────────────────────────────────────────────────────────────
def check_final_package(verbose: bool = True) -> bool:
    """
    Run a pre-submission sanity check on the project package.

    Verifies:
      - Core source modules import cleanly
      - Configuration file exists
      - Submission file exists and passes basic schema checks
      - No NaN / Inf predictions
      - Row count is non-zero
      - README and requirements exist

    Returns
    -------
    bool
        True if all checks pass; False otherwise.
    """
    checks: List[tuple[str, bool, str]] = []  # (name, passed, detail)

    # ── 1. Core imports ─────────────────────────────────────────
    core_modules = [
        "src.config", "src.data", "src.schema", "src.features",
        "src.metrics", "src.split", "src.models", "src.train",
        "src.predict", "src.ensemble", "src.submission",
    ]
    for mod in core_modules:
        try:
            __import__(mod)
            checks.append((f"import {mod}", True, ""))
        except Exception as exc:
            checks.append((f"import {mod}", False, str(exc)))

    # ── 2. Configuration & TaskSpec ──────────────────────────────
    config_path = PROJECT_ROOT / "configs" / "config.yaml"
    checks.append((
        "configs/config.yaml exists",
        config_path.exists(),
        "" if config_path.exists() else str(config_path),
    ))
    try:
        from src.config import load_config
        cfg = load_config()
        _ = cfg.task.direction
        checks.append(("TaskSpec validates (direction set)", True, ""))
    except Exception as exc:
        checks.append(("TaskSpec validates", False, str(exc)))

    # ── 3. README & requirements ─────────────────────────────────
    readme = PROJECT_ROOT / "README.md"
    checks.append(("README.md exists", readme.exists(), ""))
    req = PROJECT_ROOT / "requirements.txt"
    checks.append(("requirements.txt exists", req.exists(), ""))

    # ── 4. Artifacts ─────────────────────────────────────────────
    models_dir = PROJECT_ROOT / "artifacts" / "models"
    has_models = models_dir.exists() and any(models_dir.glob("**/*.pkl"))
    checks.append(("Final models exist", has_models, "No .pkl found in artifacts/models/"))

    # ── 5. Submission file ───────────────────────────────────────
    sub_dir = PROJECT_ROOT / "submission"
    sub_files = list(sub_dir.glob("*.csv")) if sub_dir.exists() else []
    has_sub = len(sub_files) > 0
    checks.append(("submission CSV exists", has_sub, "" if has_sub else "No CSV in submission/"))

    if has_sub:
        sub_path = sub_files[-1]
        try:
            sub_df = pd.read_csv(sub_path)
            has_rows = len(sub_df) > 0
            checks.append(("submission has rows", has_rows, ""))
            no_nan = not sub_df.isnull().values.any()
            checks.append(("submission has no NaN", no_nan, ""))
            
            pred_cols = [c for c in sub_df.columns if 'cfg' not in locals() or c != cfg.task.id_column]
            if len(pred_cols) > 0:
                pred_col = pred_cols[0]
                is_numeric = pd.api.types.is_numeric_dtype(sub_df[pred_col])
                checks.append(("prediction dtype is numeric", is_numeric, str(sub_df[pred_col].dtype)))
                if is_numeric:
                    no_inf = not np.isinf(sub_df[pred_col].values).any()
                    checks.append(("submission has no Inf", no_inf, ""))
        except Exception as exc:
            checks.append(("submission readable", False, str(exc)))

    # ── 6. Zip Secrets Check ─────────────────────────────────────
    zip_files = list(sub_dir.glob("*.zip")) if sub_dir.exists() else []
    if zip_files:
        import zipfile
        zip_path = zip_files[-1]
        try:
            with zipfile.ZipFile(zip_path, 'r') as zf:
                names = zf.namelist()
                has_secrets = any(n.endswith(".env") or "secrets" in n.lower() or "credentials" in n.lower() for n in names)
                checks.append(("no secrets in submission zip", not has_secrets, "Found potential secrets!"))
        except Exception as e:
            checks.append(("submission zip readable", False, str(e)))

    # ── 7. Tests ─────────────────────────────────────────────────
    tests_dir = PROJECT_ROOT / "tests"
    checks.append(("tests/ directory exists", tests_dir.exists(), ""))
    
    import subprocess
    try:
        res = subprocess.run(["pytest", str(tests_dir)], capture_output=True, text=True)
        passed = res.returncode == 0
        checks.append(("pytest run check", passed, "Tests failed" if not passed else ""))
    except Exception as e:
        checks.append(("pytest run check", False, str(e)))

    # ── REPORT ───────────────────────────────────────────────────
    all_pass = all(passed for _, passed, _ in checks)
    status = "PASS ✅" if all_pass else "FAIL ❌"

    if verbose:
        print("\n" + "=" * 60)
        print(f"  FINAL PACKAGE STATUS: {status}")
        print("=" * 60)
        for name, passed, detail in checks:
            icon = "✅" if passed else "❌"
            line = f"  {icon}  {name}"
            if not passed and detail:
                line += f"  →  {detail}"
            print(line)
        print("=" * 60 + "\n")

    return all_pass


# ─────────────────────────────────────────────────────────────────
# MISC
# ─────────────────────────────────────────────────────────────────
def flatten_dict(d: Dict[str, Any], sep: str = ".", prefix: str = "") -> Dict[str, Any]:
    """Flatten a nested dict to a single-level dict with dot-separated keys."""
    result: Dict[str, Any] = {}
    for k, v in d.items():
        new_key = f"{prefix}{sep}{k}" if prefix else k
        if isinstance(v, dict):
            result.update(flatten_dict(v, sep=sep, prefix=new_key))
        else:
            result[new_key] = v
    return result


def safe_divide(numerator: float, denominator: float, default: float = 0.0) -> float:
    """Division that returns default instead of ZeroDivisionError."""
    return numerator / denominator if denominator != 0 else default
