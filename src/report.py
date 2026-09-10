"""
src/report.py
Amazon ML Challenge 2026 — Report generation system.

Automates the creation of experiment summaries, model comparisons, and final reports.
"""

import pandas as pd
from pathlib import Path
from typing import List, Dict, Any
from src.logging_utils import get_logger
from src.utils import ensure_dir

logger = get_logger(__name__)


def generate_experiment_summary(
    experiment_data: List[Dict[str, Any]],
    output_dir: str = "artifacts/reports",
) -> Path:
    """Generate a CSV summary of all experiments."""
    out_dir = Path(output_dir)
    ensure_dir(out_dir)
    out_path = out_dir / "experiment_summary.csv"
    
    if not experiment_data:
        logger.warning("No experiment data provided. Skipping summary.")
        return out_path

    df = pd.DataFrame(experiment_data)
    df.to_csv(out_path, index=False)
    logger.info("Experiment summary written to %s", out_path)
    return out_path


def generate_model_comparison(
    models_data: List[Dict[str, Any]],
    output_dir: str = "artifacts/reports",
) -> Path:
    """Generate a CSV comparing different models."""
    out_dir = Path(output_dir)
    ensure_dir(out_dir)
    out_path = out_dir / "model_comparison.csv"
    
    if not models_data:
        logger.warning("No model data provided. Skipping comparison.")
        return out_path

    df = pd.DataFrame(models_data)
    df.to_csv(out_path, index=False)
    logger.info("Model comparison written to %s", out_path)
    return out_path


def generate_final_report(
    summary_data: Dict[str, Any],
    output_dir: str = "artifacts/reports",
) -> Path:
    """Generate a final Markdown report."""
    out_dir = Path(output_dir)
    ensure_dir(out_dir)
    out_path = out_dir / "final_report.md"
    
    lines = [
        "# Amazon ML Challenge 2026 — Final Report",
        "",
        "## Overview",
        f"**Best Model**: {summary_data.get('best_model', 'N/A')}",
        f"**Best Score**: {summary_data.get('best_score', 'N/A')}",
        "",
        "## Details",
        f"**Total Experiments**: {summary_data.get('total_experiments', 0)}",
        f"**Features Used**: {summary_data.get('features_used', 0)}",
        "",
        "## Notes",
        summary_data.get('notes', "No additional notes."),
    ]
    
    out_path.write_text("\\n".join(lines), encoding="utf-8")
    logger.info("Final report written to %s", out_path)
    return out_path
