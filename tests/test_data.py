"""
tests/test_data.py
Amazon ML Challenge 2026 — Data loading and discovery tests.
"""

import io
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.data import (
    DiscoveredFiles,
    debug_sample,
    discover_files,
    load_csv,
    load_jsonl,
    url_cache_key,
)


# ─────────────────────────────────────────────────────────────────
# FIXTURES
# ─────────────────────────────────────────────────────────────────
@pytest.fixture
def sample_csv(tmp_path: Path) -> Path:
    df = pd.DataFrame({"id": range(10), "value": range(10, 20), "name": [f"item_{i}" for i in range(10)]})
    p = tmp_path / "train.csv"
    df.to_csv(p, index=False)
    return p


@pytest.fixture
def sample_jsonl(tmp_path: Path) -> Path:
    import json
    p = tmp_path / "data.jsonl"
    with open(p, "w") as f:
        for i in range(5):
            f.write(json.dumps({"id": i, "text": f"record {i}"}) + "\n")
    return p


@pytest.fixture
def data_dir(tmp_path: Path, sample_csv: Path) -> Path:
    # Create a realistic data directory
    (tmp_path / "train.csv").write_text("id,value\n1,10\n2,20\n")
    (tmp_path / "test.csv").write_text("id\n1\n2\n")
    (tmp_path / "images").mkdir()
    (tmp_path / "images" / "img_001.jpg").write_bytes(b"fake")
    return tmp_path


# ─────────────────────────────────────────────────────────────────
# LOAD CSV
# ─────────────────────────────────────────────────────────────────
def test_load_csv_basic(sample_csv: Path) -> None:
    df = load_csv(sample_csv)
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 10
    assert "id" in df.columns
    assert "value" in df.columns


def test_load_csv_missing_file(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_csv(tmp_path / "nonexistent.csv")


# ─────────────────────────────────────────────────────────────────
# LOAD JSONL
# ─────────────────────────────────────────────────────────────────
def test_load_jsonl_basic(sample_jsonl: Path) -> None:
    df = load_jsonl(sample_jsonl)
    assert len(df) == 5
    assert "id" in df.columns


def test_load_jsonl_max_lines(sample_jsonl: Path) -> None:
    df = load_jsonl(sample_jsonl, max_lines=2)
    assert len(df) == 2


# ─────────────────────────────────────────────────────────────────
# FILE DISCOVERY
# ─────────────────────────────────────────────────────────────────
def test_discover_files_finds_csv(data_dir: Path) -> None:
    found = discover_files(data_dir)
    assert len(found.csv) >= 2


def test_discover_files_finds_images(data_dir: Path) -> None:
    found = discover_files(data_dir)
    assert len(found.images) >= 1


def test_discover_files_guesses_train(data_dir: Path) -> None:
    found = discover_files(data_dir)
    train_names = [p.name for p in found.probable_train]
    assert any("train" in name.lower() for name in train_names)


def test_discover_files_nonexistent(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        discover_files(tmp_path / "does_not_exist")


# ─────────────────────────────────────────────────────────────────
# DEBUG SAMPLE
# ─────────────────────────────────────────────────────────────────
def test_debug_sample_reduces_size() -> None:
    df = pd.DataFrame({"x": range(1000)})
    sampled = debug_sample(df, n=100)
    assert len(sampled) == 100


def test_debug_sample_no_op_when_smaller() -> None:
    df = pd.DataFrame({"x": range(50)})
    sampled = debug_sample(df, n=200)
    assert len(sampled) == 50


def test_debug_sample_final_mode_returns_full() -> None:
    df = pd.DataFrame({"x": range(1000)})
    sampled = debug_sample(df, n=100, mode="final")
    assert len(sampled) == 1000


# ─────────────────────────────────────────────────────────────────
# URL CACHE KEY
# ─────────────────────────────────────────────────────────────────
def test_url_cache_key_deterministic() -> None:
    url = "https://example.com/image.jpg"
    assert url_cache_key(url) == url_cache_key(url)


def test_url_cache_key_different_urls() -> None:
    k1 = url_cache_key("https://a.com/img1.jpg")
    k2 = url_cache_key("https://a.com/img2.jpg")
    assert k1 != k2


def test_url_cache_key_length() -> None:
    key = url_cache_key("https://example.com/image.jpg", prefix_len=16)
    assert len(key) == 16
