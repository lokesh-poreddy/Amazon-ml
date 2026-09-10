"""
src/logging_utils.py
Amazon ML Challenge 2026 — Structured logging setup.

Provides a consistent logger factory so every module uses the same
format and respects the configured log level. Avoids random print()
calls scattered throughout the codebase.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Optional


# ─────────────────────────────────────────────────────────────────
# FORMATS
# ─────────────────────────────────────────────────────────────────
_CONSOLE_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
_FILE_FORMAT    = "%(asctime)s | %(levelname)-8s | %(name)s | %(funcName)s:%(lineno)d | %(message)s"
_DATE_FORMAT    = "%Y-%m-%d %H:%M:%S"

_initialized: bool = False


# ─────────────────────────────────────────────────────────────────
# ROOT LOGGER SETUP
# ─────────────────────────────────────────────────────────────────
def setup_logging(
    level: str = "INFO",
    log_dir: Optional[str] = None,
    log_file: Optional[str] = None,
) -> None:
    """
    Configure the root logger once.

    Parameters
    ----------
    level : str
        Logging level: DEBUG | INFO | WARNING | ERROR
    log_dir : str | None
        Directory for log files. If None, no file handler is added.
    log_file : str | None
        Explicit log file name inside log_dir. Defaults to 'run.log'.
    """
    global _initialized
    if _initialized:
        return

    numeric_level = getattr(logging, level.upper(), logging.INFO)

    root = logging.getLogger()
    root.setLevel(numeric_level)

    # Remove any existing handlers (e.g. from Jupyter)
    root.handlers.clear()

    # ── Console handler ──────────────────────────────────────────
    console = logging.StreamHandler(sys.stdout)
    console.setLevel(numeric_level)
    console.setFormatter(logging.Formatter(_CONSOLE_FORMAT, datefmt=_DATE_FORMAT))
    root.addHandler(console)

    # ── File handler (optional) ──────────────────────────────────
    if log_dir is not None:
        log_dir_path = Path(log_dir)
        log_dir_path.mkdir(parents=True, exist_ok=True)
        fname = log_file or "run.log"
        file_handler = logging.FileHandler(log_dir_path / fname, encoding="utf-8")
        file_handler.setLevel(numeric_level)
        file_handler.setFormatter(
            logging.Formatter(_FILE_FORMAT, datefmt=_DATE_FORMAT)
        )
        root.addHandler(file_handler)

    _initialized = True


def get_logger(name: str) -> logging.Logger:
    """
    Return a named logger.

    This is the preferred way to get a logger in any module:

        logger = get_logger(__name__)
        logger.info("Loading data ...")

    Parameters
    ----------
    name : str
        Logger name, typically ``__name__`` of the calling module.

    Returns
    -------
    logging.Logger
    """
    if not _initialized:
        # Lazy init with sensible defaults if setup_logging() was never called
        setup_logging()
    return logging.getLogger(name)


def set_level(level: str) -> None:
    """Dynamically adjust the global log level at runtime."""
    numeric_level = getattr(logging, level.upper(), logging.INFO)
    logging.getLogger().setLevel(numeric_level)
    for handler in logging.getLogger().handlers:
        handler.setLevel(numeric_level)
