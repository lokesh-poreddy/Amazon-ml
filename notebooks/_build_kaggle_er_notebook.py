#!/usr/bin/env python3
"""Build the production Kaggle ER notebook (nbformat 4)."""
from __future__ import annotations

import json
from pathlib import Path

OUT = Path(__file__).with_name("Kaggle_Amazon_ML_ER_Production.ipynb")


def md(source: str) -> dict:
    text = source.strip("\n") + "\n"
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": [line + "\n" for line in text.split("\n")[:-1]] + [text.split("\n")[-1] + "\n"]
        if text
        else ["\n"],
    }


def code(source: str) -> dict:
    text = source.strip("\n") + "\n"
    lines = text.split("\n")
    src = [ln + "\n" for ln in lines[:-1]]
    if lines[-1] != "":
        src.append(lines[-1] + "\n")
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": src,
    }


def cells():
    c = []

    c.append(md("""# Amazon ML Challenge 2026 — Business Entity Resolution

Production Kaggle notebook: **train → validate → test → `matching_results.tsv`**.

| Contract | Rule |
|---|---|
| Metric | Macro entity-level \\(F_{0.5}\\) (precision weighted 2× recall) |
| Scored file | `matching_results.tsv` only |
| Package file | `candidate_pairs.tsv` = **final** matcher input |
| Country | Open-set strings (train: US/India; test also **France**) |
| Fair play | No registries, geocoders, APIs, or internet identity lookup |
| Hardware | ~30 GB RAM, T4×2 — this notebook is sized for that, not a laptop Cartesian join |

Do **not** use the GitHub default `match_threshold: 0.50`. Thresholds are fit on held-out S1 entities.

**Scale reality (official files):** ~2.2M train S1, ~10.3M train S2+S3, ~1.7M test S1, ~10.0M test S2+S3. We therefore:

1. Index **all** S2/S3 targets (recall ceiling).
2. Train/validate the matcher on a **stratified S1 sample** (fits RAM).
3. Stream test S1 in chunks and write TSV rows immediately.
"""))

    c.append(md("## Cell 1 — Runtime configuration"))
    c.append(code("""# ============================================================
# AMAZON ML CHALLENGE 2026
# BUSINESS ENTITY RESOLUTION
# Production: train -> validate -> test -> matching_results.tsv
# Hardware target: ~30 GB RAM, T4x2 (GPU optional for LightGBM)
# ============================================================

from pathlib import Path
import os
import sys
import gc
import re
import json
import time
import math
import random
import shutil
import subprocess
from collections import defaultdict, Counter

SEED = 42
random.seed(SEED)
os.environ["PYTHONHASHSEED"] = str(SEED)
os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")

import numpy as np

np.random.seed(SEED)

print("Python:", sys.version.replace("\\n", " "))
print("NumPy:", np.__version__)
print("Seed:", SEED)

try:
    import psutil
    vm = psutil.virtual_memory()
    print(f"RAM total={vm.total/1e9:.1f} GB  available={vm.available/1e9:.1f} GB")
except Exception as exc:
    print("psutil unavailable:", exc)


def ram_gb():
    try:
        import psutil as _ps
        return _ps.virtual_memory().used / 1e9
    except Exception:
        return float("nan")


print(f"RAM used now: {ram_gb():.2f} GB")
"""))

    c.append(md("""## Cell 2 — Install ML/string libraries only

Fair-play: RapidFuzz / LightGBM / anyascii only. **No** geocoding, registries, or business APIs.
"""))
    c.append(code("""import importlib.util

REQUIRED = {
    "rapidfuzz": "rapidfuzz",
    "lightgbm": "lightgbm",
    "anyascii": "anyascii",
    "joblib": "joblib",
    "sklearn": "scikit-learn",
    "pyarrow": "pyarrow",
}

for module_name, pip_name in REQUIRED.items():
    if importlib.util.find_spec(module_name) is None:
        print("Installing:", pip_name)
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", pip_name])

print("Dependency check complete.")
"""))

    c.append(md("## Cell 3 — Imports"))
    c.append(code("""import pandas as pd
from rapidfuzz import fuzz
import lightgbm as lgb
from lightgbm import LGBMClassifier
from anyascii import anyascii
from sklearn.model_selection import train_test_split
from sklearn.metrics import average_precision_score, roc_auc_score
import joblib

print("Pandas:", pd.__version__)
print("LightGBM:", lgb.__version__)

USE_GPU = False
try:
    import torch
    USE_GPU = bool(torch.cuda.is_available())
    if USE_GPU:
        print("CUDA devices:", torch.cuda.device_count(), torch.cuda.get_device_name(0))
except Exception:
    pass
print("LightGBM GPU requested:", USE_GPU)
"""))

    c.append(md("""## Cell 4 — Discover the seven official TSVs

Looks under `/kaggle/input` first, then common local layouts (`student_resource/dataset`, `dataset/`).
"""))
    c.append(code("""# ============================================================
# PATHS + RAM-SAFE PIPELINE KNOBS
# ============================================================

KAGGLE_INPUT = Path("/kaggle/input")
WORK_DIR = Path("/kaggle/working") if Path("/kaggle/working").exists() else Path.cwd()

OUTPUT_DIR = WORK_DIR / "output"
ARTIFACT_DIR = WORK_DIR / "artifacts"
MODEL_DIR = ARTIFACT_DIR / "models"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)

EXPECTED_FILES = [
    "train_source1.tsv",
    "train_source2.tsv",
    "train_source3.tsv",
    "train_ground_truth.tsv",
    "test_source1.tsv",
    "test_source2.tsv",
    "test_source3.tsv",
]

SEARCH_ROOTS = [
    KAGGLE_INPUT,
    Path("/kaggle/input").resolve() if KAGGLE_INPUT.exists() else None,
    Path.cwd() / "student_resource" / "dataset",
    Path.cwd() / "dataset",
    Path.cwd(),
]
SEARCH_ROOTS = [p for p in SEARCH_ROOTS if p is not None and Path(p).exists()]


def discover_file(filename: str) -> Path:
    matches = []
    for root in SEARCH_ROOTS:
        matches.extend(Path(root).rglob(filename))
    matches = sorted(set(matches), key=lambda p: (len(p.parts), str(p)))
    if not matches:
        raise FileNotFoundError(f"Could not find {filename}. Attach the Kaggle dataset or place TSVs under dataset/.")
    if len(matches) > 1:
        print(f"[WARN] Multiple matches for {filename}; using {matches[0]}")
    return matches[0]


PATHS = {name: discover_file(name) for name in EXPECTED_FILES}

print("\\nDiscovered dataset:")
for name, path in PATHS.items():
    print(f"{name:25s} -> {path}  ({path.stat().st_size/1e6:.1f} MB)")

# --- scale knobs (30 GB RAM) ---
# Full S1×S2/S3 Cartesian matching is impossible. We index ALL targets,
# train on a stratified S1 sample, and stream test S1.
CFG = {
    "train_s1_n": 55_000,      # labelled S1 used to fit the pair model
    "val_s1_n": 18_000,        # held-out S1 for F0.5 policy search
    "max_candidates": 36,      # final matcher input per S1 (also written to candidate_pairs)
    "max_evidence": 160,       # pre-rerank cap
    "token_max_df": 35,
    "char_max_df": 120,
    "postal_max_df": 350,
    "exact_name_max_df": 80,
    "address_max_df": 40,
    "rare_token_take": 6,
    "rare_gram_take": 5,
    "test_chunk": 4_000,
    "lgbm_trees_cap": 900,
}

print("\\nPipeline knobs:")
print(json.dumps(CFG, indent=2))
"""))

    c.append(md("## Cell 5 — TSV loader + schema contract (`sep='\\t'`)"))
    c.append(code("""REQUIRED_RECORD_COLUMNS = [
    "entity_id",
    "business_name",
    "business_address",
    "country",
]
GROUND_TRUTH_COLUMNS = ["source1_entity_id", "matched_entity_ids"]


def read_record_tsv(path: Path, usecols=None) -> pd.DataFrame:
    df = pd.read_csv(
        path,
        sep="\\t",
        dtype="string",
        keep_default_na=False,
        na_filter=False,
        usecols=usecols or REQUIRED_RECORD_COLUMNS,
        low_memory=False,
        encoding="utf-8",
    )
    missing = [c for c in (usecols or REQUIRED_RECORD_COLUMNS) if c not in df.columns]
    if missing:
        raise ValueError(f"{path.name}: missing columns {missing}")
    return df


def read_ground_truth(path: Path) -> pd.DataFrame:
    df = pd.read_csv(
        path,
        sep="\\t",
        dtype="string",
        keep_default_na=False,
        na_filter=False,
        usecols=GROUND_TRUTH_COLUMNS,
        low_memory=False,
        encoding="utf-8",
    )
    missing = [c for c in GROUND_TRUTH_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"{path.name}: missing columns {missing}")
    return df


print("Loaders ready. Test S1 is streamed later (Cell 34).")
"""))

    c.append(md("## Cell 6 — Data integrity audit (streaming, low extra RAM)"))
    c.append(code("""def audit_source_stream(path: Path, expected_prefix: str, name: str, max_countries: int = 12):
    print(f"\\n===== {name} =====")
    print("file:", path)
    n = 0
    dup = 0
    seen_chunk_warn = 0
    wrong_prefix = 0
    empty = Counter()
    countries = Counter()
    last_ids = set()
    # Single pass; duplicate detection is exact only within a rolling window
    # plus a global unique count via hashing sample. Full unique-ID set of 10M
    # IDs is several GB; we count rows and prefix errors exactly.
    with open(path, "r", encoding="utf-8") as f:
        header = f.readline().rstrip("\\n").split("\\t")
        col = {c: i for i, c in enumerate(header)}
        for req in REQUIRED_RECORD_COLUMNS:
            if req not in col:
                raise ValueError(f"{path.name} missing {req}")
        for line in f:
            if not line.strip():
                continue
            parts = line.rstrip("\\n").split("\\t")
            n += 1
            eid = parts[col["entity_id"]] if col["entity_id"] < len(parts) else ""
            if not eid.startswith(expected_prefix):
                wrong_prefix += 1
            for field in ("business_name", "business_address", "country"):
                val = parts[col[field]] if col[field] < len(parts) else ""
                if not val.strip():
                    empty[field] += 1
            ctry = parts[col["country"]] if col["country"] < len(parts) else ""
            countries[ctry] += 1
            if n <= 2_000_000:
                if eid in last_ids:
                    dup += 1
                last_ids.add(eid)
                if len(last_ids) > 1_500_000:
                    last_ids.clear()
                    seen_chunk_warn += 1
    print("Rows:", n)
    print("Wrong-prefix IDs:", wrong_prefix)
    print("Duplicate IDs (partial window):", dup, "window_resets:", seen_chunk_warn)
    for field in ("business_name", "business_address", "country"):
        print(f"{field:18s} empty:", empty[field])
    print("Countries:")
    for k, v in countries.most_common(max_countries):
        print(f"  {k!r}: {v}")
    return n


train_s1_n = audit_source_stream(PATHS["train_source1.tsv"], "S1-", "TRAIN SOURCE 1")
train_s2_n = audit_source_stream(PATHS["train_source2.tsv"], "S2-", "TRAIN SOURCE 2")
train_s3_n = audit_source_stream(PATHS["train_source3.tsv"], "S3-", "TRAIN SOURCE 3")
_ = audit_source_stream(PATHS["test_source2.tsv"], "S2-", "TEST SOURCE 2")
_ = audit_source_stream(PATHS["test_source3.tsv"], "S3-", "TEST SOURCE 3")
print("\\nTrain S1 rows (exact):", train_s1_n)
print("Train targets ~", train_s2_n + train_s3_n)
print(f"RAM used: {ram_gb():.2f} GB")
"""))

    c.append(md("""## Cell 7 — Ground truth parser

Singletons (empty `matched_entity_ids`) are **in** the macro average: correct empty list scores 1.0; any predicted match scores 0.0.
"""))
    c.append(code("""ground_truth = read_ground_truth(PATHS["train_ground_truth.tsv"])
print("Ground-truth rows:", len(ground_truth))
print("Duplicate S1 IDs:", int(ground_truth["source1_entity_id"].duplicated().sum()))


def build_truth_map(gt: pd.DataFrame):
    truth = {}
    for row in gt.itertuples(index=False):
        s1_id = str(row.source1_entity_id).strip()
        raw = str(row.matched_entity_ids).strip()
        if not raw:
            truth[s1_id] = set()
        else:
            truth[s1_id] = {x.strip() for x in raw.split(",") if x.strip()}
    return truth


truth_map = build_truth_map(ground_truth)
del ground_truth
gc.collect()

singleton_count = sum(1 for v in truth_map.values() if not v)
print("Ground-truth S1 entities:", len(truth_map))
print("Singletons:", singleton_count, f"({singleton_count/max(1,len(truth_map)):.4f})")
print("Non-singletons:", len(truth_map) - singleton_count)
print("Total true S1 -> S2/S3 links:", sum(len(v) for v in truth_map.values()))
"""))

    c.append(md("""## Cell 8 — Multi-view normalization

Raw strings are kept until features are built. Legal-form stripping is **trailing tokens only**. Address expansions include US/India/**French** abbreviations without hard-coding allowed countries.
"""))
    c.append(code("""SPACE_RE = re.compile(r"\\s+")
NON_ALNUM_RE = re.compile(r"[^a-z0-9\\s]")
COMPACT_RE = re.compile(r"[^a-z0-9]")
NON_DIGIT_RE = re.compile(r"[^0-9]")
POSTAL_RE = re.compile(r"(?<!\\d)(\\d{4,6})(?!\\d)")

LEGAL_SUFFIXES = {
    "inc", "incorporated", "corp", "corporation", "co", "company",
    "llc", "ltd", "limited", "plc", "pvt", "private", "llp",
    "sarl", "sas", "gmbh", "sa", "sasu", "eurl", "snc", "sci",
    "bv", "nv", "ag", "oy", "ab", "srl", "spa", "ltda",
}

ADDRESS_MAP = {
    "rd": "road", "rd.": "road",
    "st": "street", "st.": "street", "str": "street",
    "ave": "avenue", "ave.": "avenue", "av": "avenue",
    "blvd": "boulevard", "blvd.": "boulevard", "bd": "boulevard", "bvd": "boulevard",
    "dr": "drive", "dr.": "drive",
    "ln": "lane", "ln.": "lane",
    "hwy": "highway", "nh": "highway",
    "pkwy": "parkway", "pky": "parkway",
    "ct": "court",
    "apt": "apartment", "ste": "suite", "fl": "floor",
    "no": "number", "nr": "near",
    "rue": "rue", "av.": "avenue", "pl": "place", "pl.": "place",
    "ch": "chemin", "imp": "impasse", "all": "allee", "allee": "allee",
    "bat": "batiment", "batiment": "batiment",
    "res": "residence", "cedex": "cedex",
    "sec": "sector", "blk": "block", "bldg": "building",
}

ADDRESS_MAP_RE = re.compile(
    r"\\b(" + "|".join(re.escape(k) for k in sorted(ADDRESS_MAP, key=len, reverse=True)) + r")\\b",
    flags=re.IGNORECASE,
)


def transliterate_ascii(text: str) -> str:
    text = "" if text is None else str(text)
    return text if text.isascii() else anyascii(text)


def base_text(text: str) -> str:
    text = transliterate_ascii(text).lower()
    text = text.replace("&", " and ").replace("/", " ")
    text = SPACE_RE.sub(" ", text)
    return text.strip()


def normalize_name(text: str) -> str:
    text = NON_ALNUM_RE.sub(" ", base_text(text))
    text = SPACE_RE.sub(" ", text).strip()
    tokens = text.split()
    while tokens and tokens[-1] in LEGAL_SUFFIXES:
        tokens.pop()
    return " ".join(tokens)


def normalize_address(text: str) -> str:
    text = base_text(text)
    text = ADDRESS_MAP_RE.sub(lambda m: ADDRESS_MAP[m.group(0).lower()], text)
    text = NON_ALNUM_RE.sub(" ", text)
    return SPACE_RE.sub(" ", text).strip()


def compact(text: str) -> str:
    return COMPACT_RE.sub("", text or "")


def extract_postal(text: str) -> str:
    matches = POSTAL_RE.findall(text or "")
    return matches[-1] if matches else ""


def tokens_of(text: str):
    if not text:
        return ()
    return tuple(t for t in text.split() if t)


def grams_of(text: str, n: int = 4):
    if not text or len(text) < n:
        return ()
    return tuple(text[i:i + n] for i in range(len(text) - n + 1))
"""))

    c.append(md("## Cell 9 — Target store (numpy arrays, pandas dropped)"))
    c.append(code("""class TargetStore:
    __slots__ = (
        "entity_id", "source", "name_norm", "name_compact",
        "address_norm", "address_compact", "country_norm", "postal",
        "n",
    )

    def __init__(self, df: pd.DataFrame):
        self.entity_id = df["entity_id"].astype(str).to_numpy()
        self.source = self.entity_id.astype("U2")
        names = df["business_name"].astype(str).map(normalize_name)
        addrs = df["business_address"].astype(str).map(normalize_address)
        self.name_norm = names.to_numpy()
        self.address_norm = addrs.to_numpy()
        self.name_compact = np.array([compact(x) for x in self.name_norm], dtype=object)
        self.address_compact = np.array([compact(x) for x in self.address_norm], dtype=object)
        self.country_norm = (
            df["country"].astype(str).str.strip().str.casefold().to_numpy()
        )
        self.postal = np.array([extract_postal(x) for x in self.address_norm], dtype=object)
        self.n = len(self.entity_id)

    def __len__(self):
        return self.n


def load_target_store(paths):
    frames = [read_record_tsv(p) for p in paths]
    df = pd.concat(frames, ignore_index=True)
    del frames
    gc.collect()
    store = TargetStore(df)
    del df
    gc.collect()
    return store


def normalize_query_frame(df: pd.DataFrame) -> pd.DataFrame:
    out = df.loc[:, REQUIRED_RECORD_COLUMNS].copy()
    out["entity_id"] = out["entity_id"].astype(str)
    out["country_norm"] = out["country"].astype(str).str.strip().str.casefold()
    out["name_norm"] = out["business_name"].astype(str).map(normalize_name)
    out["address_norm"] = out["business_address"].astype(str).map(normalize_address)
    out["name_compact"] = out["name_norm"].map(compact)
    out["address_compact"] = out["address_norm"].map(compact)
    out["postal"] = out["address_norm"].map(extract_postal)
    return out


print("Normalizers + TargetStore ready.")
"""))

    c.append(md("""## Cell 10 — Entity-aware S1 sample (not an 80/20 split of 2.2M)

Splitting **pairs** leaks. Splitting all 2.2M S1 for candidate generation against 10M targets will exhaust 30 GB. We stratify on singleton vs non-singleton, then sample.
"""))
    c.append(code("""t0 = time.time()
train_s1_all = read_record_tsv(PATHS["train_source1.tsv"])
print("Loaded train S1:", train_s1_all.shape, f"in {time.time()-t0:.1f}s")

all_s1_ids = train_s1_all["entity_id"].astype(str).to_numpy()
strata = np.array([0 if len(truth_map.get(sid, set())) == 0 else 1 for sid in all_s1_ids], dtype=np.int8)

holdout_n = CFG["train_s1_n"] + CFG["val_s1_n"]
holdout_n = min(holdout_n, len(all_s1_ids))
val_frac = CFG["val_s1_n"] / max(1, holdout_n)

# Two-stage: first draw a modelling pool, then split that pool.
pool_ids, _, pool_strata, _ = train_test_split(
    all_s1_ids,
    strata,
    train_size=holdout_n,
    random_state=SEED,
    stratify=strata if len(np.unique(strata)) > 1 else None,
)

train_ids, val_ids = train_test_split(
    pool_ids,
    test_size=val_frac,
    random_state=SEED,
    stratify=pool_strata if len(np.unique(pool_strata)) > 1 else None,
)
train_ids = np.asarray(train_ids)
val_ids = np.asarray(val_ids)
train_id_set = set(train_ids)
val_id_set = set(val_ids)

eid_series = train_s1_all["entity_id"].astype(str)
train_s1_split = normalize_query_frame(
    train_s1_all.loc[eid_series.isin(train_id_set)].reset_index(drop=True)
)
val_s1_split = normalize_query_frame(
    train_s1_all.loc[eid_series.isin(val_id_set)].reset_index(drop=True)
)

del train_s1_all, eid_series, all_s1_ids, strata, pool_ids, pool_strata
gc.collect()

print("Train S1 sample:", len(train_s1_split))
print("Validation S1 sample:", len(val_s1_split))
print(
    "Train singleton rate:",
    float(np.mean([len(truth_map.get(x, set())) == 0 for x in train_s1_split["entity_id"]])),
)
print(
    "Val singleton rate:",
    float(np.mean([len(truth_map.get(x, set())) == 0 for x in val_s1_split["entity_id"]])),
)
print(f"RAM used: {ram_gb():.2f} GB")
"""))

    c.append(md("## Cell 11 — Load + normalize **all** train S2/S3 targets"))
    c.append(code("""t0 = time.time()
print("Loading train S2 + S3 (this is the heavy RAM step)...")
train_target = load_target_store([PATHS["train_source2.tsv"], PATHS["train_source3.tsv"]])
print("Train targets:", len(train_target), f"in {time.time()-t0:.1f}s")
print("Source mix:", dict(zip(*np.unique(train_target.source, return_counts=True))))
print(f"RAM used: {ram_gb():.2f} GB")
"""))

    c.append(md("## Cell 12 — Blocking indexes (no Cartesian product)"))
    c.append(code("""def build_exact_index(values, max_df):
    buckets = defaultdict(list)
    for i, value in enumerate(values):
        if not value:
            continue
        buckets[value].append(i)
    index = {}
    for key, rows in buckets.items():
        if 0 < len(rows) <= max_df:
            index[key] = np.asarray(rows, dtype=np.int32)
    return index


def build_rare_token_index(values, max_df):
    token_df = Counter()
    token_cache = []
    for value in values:
        if not value:
            token_cache.append(())
            continue
        toks = tuple({t for t in value.split() if len(t) >= 3})
        token_cache.append(toks)
        token_df.update(toks)
    buckets = defaultdict(list)
    for i, toks in enumerate(token_cache):
        for token in toks:
            if token_df[token] <= max_df:
                buckets[token].append(i)
    return {k: np.asarray(v, dtype=np.int32) for k, v in buckets.items() if v}


def build_rare_chargram_index(values, max_df, n=4):
    gram_df = Counter()
    gram_cache = []
    for value in values:
        g = grams_of(value, n)
        # unique grams per record
        ug = tuple(set(g))
        gram_cache.append(ug)
        gram_df.update(ug)
    buckets = defaultdict(list)
    for i, grams in enumerate(gram_cache):
        for gram in grams:
            if gram_df[gram] <= max_df:
                buckets[gram].append(i)
    return {k: np.asarray(v, dtype=np.int32) for k, v in buckets.items() if v}


class ERBlockIndex:
    def __init__(self, target: TargetStore):
        self.target = target
        t0 = time.time()
        names = target.name_norm
        compact_names = target.name_compact
        addresses = target.address_norm
        compact_addresses = target.address_compact
        countries = target.country_norm
        postals = target.postal

        self.names = names
        self.compact_names = compact_names
        self.addresses = addresses
        self.compact_addresses = compact_addresses
        self.countries = countries
        self.postals = postals
        self.ids = target.entity_id
        self.sources = target.source

        print("Building exact-name index...")
        self.name_idx = build_exact_index(names, CFG["exact_name_max_df"])
        print("Building compact-name index...")
        self.compact_name_idx = build_exact_index(compact_names, CFG["exact_name_max_df"])

        print("Building name+country index...")
        name_country_keys = np.empty(len(names), dtype=object)
        for i, (nm, cc) in enumerate(zip(names, countries)):
            name_country_keys[i] = f"{nm}\\x1f{cc}" if nm and cc else ""
        self.name_country_idx = build_exact_index(name_country_keys, 100)

        print("Building compact-address index...")
        self.address_idx = build_exact_index(compact_addresses, CFG["address_max_df"])
        print("Building postal index...")
        self.postal_idx = build_exact_index(postals, CFG["postal_max_df"])
        print("Building rare name-token index...")
        self.name_token_idx = build_rare_token_index(names, CFG["token_max_df"])
        print("Building rare address-token index...")
        self.address_token_idx = build_rare_token_index(addresses, CFG["token_max_df"])
        print("Building rare name-chargram index...")
        self.name_char_idx = build_rare_chargram_index(names, CFG["char_max_df"], n=4)

        print("Index sizes:")
        print("  exact name:", len(self.name_idx))
        print("  compact name:", len(self.compact_name_idx))
        print("  name+country:", len(self.name_country_idx))
        print("  exact address:", len(self.address_idx))
        print("  postal:", len(self.postal_idx))
        print("  rare name tokens:", len(self.name_token_idx))
        print("  rare addr tokens:", len(self.address_token_idx))
        print("  name chargrams:", len(self.name_char_idx))
        print(f"Index time {time.time()-t0:.1f}s | RAM {ram_gb():.2f} GB")
"""))

    c.append(md("## Cell 13 — Build training target index"))
    c.append(code("""t0 = time.time()
train_index = ERBlockIndex(train_target)
print("Train index wall time:", round(time.time() - t0, 2), "sec")
"""))

    c.append(md("""## Cell 14 — Candidate generator (final matcher input)

This **is** `candidate_pairs.tsv`: last retrieval + cheap rerank, then at most `max_candidates` IDs. No later filter except the ML decision.
"""))
    c.append(code("""BIT_EXACT_NAME = 1 << 0
BIT_COMPACT_NAME = 1 << 1
BIT_NAME_COUNTRY = 1 << 2
BIT_POSTAL = 1 << 3
BIT_EXACT_ADDRESS = 1 << 4
BIT_NAME_TOKEN = 1 << 5
BIT_ADDRESS_TOKEN = 1 << 6
BIT_NAME_CHAR = 1 << 7


def _add_hits(evidence, hits, bit, weight, max_evidence):
    if hits is None:
        return
    for tidx in hits:
        tidx = int(tidx)
        cur = evidence.get(tidx)
        if cur is None:
            if len(evidence) >= max_evidence and bit in (BIT_NAME_CHAR, BIT_ADDRESS_TOKEN, BIT_POSTAL):
                continue
            evidence[tidx] = [bit, float(weight)]
        else:
            cur[0] |= bit
            cur[1] += float(weight)


def generate_candidates_for_dataframe(
    qdf: pd.DataFrame,
    index: ERBlockIndex,
    max_candidates=None,
    progress_every=5000,
):
    max_candidates = CFG["max_candidates"] if max_candidates is None else max_candidates
    max_evidence = CFG["max_evidence"]
    q_ids = qdf["entity_id"].astype(str).to_numpy()
    q_names = qdf["name_norm"].astype(str).to_numpy()
    q_compact_names = qdf["name_compact"].astype(str).to_numpy()
    q_addresses = qdf["address_norm"].astype(str).to_numpy()
    q_compact_addresses = qdf["address_compact"].astype(str).to_numpy()
    q_countries = qdf["country_norm"].astype(str).to_numpy()
    q_postals = qdf["postal"].astype(str).to_numpy()
    target_ids = index.ids

    rows = []
    candidate_map = {}
    start = time.time()

    for i in range(len(qdf)):
        q_name = q_names[i]
        q_compact_name = q_compact_names[i]
        q_address = q_addresses[i]
        q_compact_address = q_compact_addresses[i]
        q_country = q_countries[i]
        q_postal = q_postals[i]
        evidence = {}

        if q_name:
            _add_hits(evidence, index.name_idx.get(q_name), BIT_EXACT_NAME, 20.0, max_evidence)
        if q_compact_name:
            _add_hits(evidence, index.compact_name_idx.get(q_compact_name), BIT_COMPACT_NAME, 18.0, max_evidence)
        if q_name and q_country:
            _add_hits(
                evidence,
                index.name_country_idx.get(f"{q_name}\\x1f{q_country}"),
                BIT_NAME_COUNTRY,
                15.0,
                max_evidence,
            )
        if q_compact_address:
            _add_hits(evidence, index.address_idx.get(q_compact_address), BIT_EXACT_ADDRESS, 16.0, max_evidence)
        if q_postal:
            _add_hits(evidence, index.postal_idx.get(q_postal), BIT_POSTAL, 8.0, max_evidence)

        if q_name:
            tokens = {t for t in q_name.split() if len(t) >= 3}
            ordered = sorted(tokens, key=lambda tok: len(index.name_token_idx.get(tok, ())))
            for token in ordered[: CFG["rare_token_take"]]:
                _add_hits(evidence, index.name_token_idx.get(token), BIT_NAME_TOKEN, 3.0, max_evidence)

        if q_address:
            tokens = {t for t in q_address.split() if len(t) >= 3}
            ordered = sorted(tokens, key=lambda tok: len(index.address_token_idx.get(tok, ())))
            for token in ordered[: CFG["rare_token_take"]]:
                _add_hits(evidence, index.address_token_idx.get(token), BIT_ADDRESS_TOKEN, 2.5, max_evidence)

        if q_name and len(q_name) >= 4:
            grams = set(grams_of(q_name, 4))
            ordered = sorted(grams, key=lambda g: len(index.name_char_idx.get(g, ())))
            for gram in ordered[: CFG["rare_gram_take"]]:
                _add_hits(evidence, index.name_char_idx.get(gram), BIT_NAME_CHAR, 1.4, max_evidence)

        if not evidence:
            candidate_map[q_ids[i]] = set()
            if progress_every and i % progress_every == 0:
                print(f"{i:,}/{len(qdf):,} | elapsed={time.time()-start:.1f}s")
            continue

        ranked = []
        for tidx, (mask, block_score) in evidence.items():
            t_name = index.names[tidx]
            t_addr = index.addresses[tidx]
            t_country = index.countries[tidx]
            t_postal = index.postals[tidx]
            nr = fuzz.ratio(q_name, t_name) / 100.0 if q_name and t_name else 0.0
            ar = fuzz.ratio(q_address, t_addr) / 100.0 if q_address and t_addr else 0.0
            same_country = 1.0 if q_country and t_country and q_country == t_country else 0.0
            same_postal = 1.0 if q_postal and t_postal and q_postal == t_postal else 0.0
            cheap = block_score + 25.0 * nr + 12.0 * ar + 6.0 * same_country + 8.0 * same_postal
            ranked.append((cheap, block_score, int(tidx), int(mask)))

        ranked.sort(key=lambda x: (-x[0], -x[1], str(target_ids[x[2]])))
        ranked = ranked[:max_candidates]
        selected = set()
        for cheap, block_score, tidx, mask in ranked:
            rows.append((i, tidx, mask, block_score))
            selected.add(str(target_ids[tidx]))
        candidate_map[q_ids[i]] = selected

        if progress_every and i and i % progress_every == 0:
            print(
                f"{i:,}/{len(qdf):,} | avg candidates={len(rows)/i:.2f} | elapsed={time.time()-start:.1f}s"
            )

    print("Candidate generation completed in", round(time.time() - start, 2), "sec")
    return rows, candidate_map
"""))

    c.append(md("## Cell 15 — Blocking quality (recall ceiling)"))
    c.append(code("""def blocking_report(candidate_map, truth, target_count):
    true_pairs = recovered_pairs = 0
    entity_total = entity_all_recovered = 0
    candidate_total = 0
    for s1_id, cand in candidate_map.items():
        true_ids = set(truth.get(s1_id, set()))
        candidate_total += len(cand)
        if true_ids:
            entity_total += 1
            if true_ids.issubset(cand):
                entity_all_recovered += 1
        true_pairs += len(true_ids)
        recovered_pairs += len(true_ids & cand)
    possible = max(1, len(candidate_map) * target_count)
    return {
        "s1_entities": len(candidate_map),
        "candidate_pairs": candidate_total,
        "true_pairs": true_pairs,
        "recovered_true_pairs": recovered_pairs,
        "pair_blocking_recall": recovered_pairs / true_pairs if true_pairs else 1.0,
        "non_singleton_entity_coverage": entity_all_recovered / entity_total if entity_total else 1.0,
        "reduction_ratio": 1.0 - candidate_total / possible,
        "mean_candidates_per_s1": candidate_total / max(1, len(candidate_map)),
    }
"""))

    c.append(md("## Cell 16 — Generate train/validation candidate sets"))
    c.append(code("""t0 = time.time()
train_candidate_rows, train_candidate_map = generate_candidates_for_dataframe(
    train_s1_split, train_index, progress_every=5000
)
val_candidate_rows, val_candidate_map = generate_candidates_for_dataframe(
    val_s1_split, train_index, progress_every=3000
)

print("\\nTRAIN BLOCKING")
train_block_report = blocking_report(
    train_candidate_map,
    {sid: truth_map.get(sid, set()) for sid in train_s1_split["entity_id"]},
    len(train_target),
)
print(json.dumps(train_block_report, indent=2))

print("\\nVALIDATION BLOCKING")
val_block_report = blocking_report(
    val_candidate_map,
    {sid: truth_map.get(sid, set()) for sid in val_s1_split["entity_id"]},
    len(train_target),
)
print(json.dumps(val_block_report, indent=2))
print("Candidate-generation wall time:", round(time.time() - t0, 2), "sec")
print(f"RAM used: {ram_gb():.2f} GB")
"""))

    c.append(md("""## Cell 17 — Pair-feature engine

String similarities (ratio, token-set, token-sort, partial), Jaccard/overlap, postal/country agreement, length, blocking provenance. **True** token/char stats are computed on the pair — not the broken incremental counters from blocking.
"""))
    c.append(code("""FEATURE_COLS = [
    "name_ratio", "name_token_set_ratio", "name_token_sort_ratio", "name_partial_ratio",
    "address_ratio", "address_token_set_ratio", "address_token_sort_ratio", "address_partial_ratio",
    "name_exact", "name_compact_exact", "address_exact",
    "postal_exact", "country_exact",
    "name_prefix3", "address_prefix3",
    "name_contains", "address_contains",
    "name_length_ratio", "address_length_ratio",
    "name_token_jaccard", "name_token_overlap",
    "address_token_jaccard", "address_token_overlap",
    "name_char4_jaccard", "address_char3_jaccard",
    "address_digit_ratio",
    "name_addr_mean", "name_addr_geom", "name_minus_address",
    "block_exact_name", "block_compact_name", "block_name_country", "block_postal",
    "block_exact_address", "block_name_token", "block_address_token", "block_name_char",
    "block_count", "block_score",
    "target_is_s2",
    "query_name_missing", "target_name_missing",
    "query_address_missing", "target_address_missing",
    "both_name_present", "both_address_present",
]


def _jaccard(a_set, b_set):
    if not a_set or not b_set:
        return 0.0
    inter = len(a_set & b_set)
    union = len(a_set) + len(b_set) - inter
    return inter / union if union else 0.0


def _overlap(a_set, b_set):
    if not a_set or not b_set:
        return 0.0
    return len(a_set & b_set) / min(len(a_set), len(b_set))


def _safe_len_ratio(a, b):
    if not a or not b:
        return 0.0
    return min(len(a), len(b)) / max(len(a), len(b))


def _char_jaccard(a, b, n):
    if not a or not b or len(a) < n or len(b) < n:
        return 0.0
    sa = {a[i:i + n] for i in range(len(a) - n + 1)}
    sb = {b[i:i + n] for i in range(len(b) - n + 1)}
    return _jaccard(sa, sb)


def build_pair_features(qdf: pd.DataFrame, index: ERBlockIndex, rows):
    q_names = qdf["name_norm"].astype(str).to_numpy()
    q_compact = qdf["name_compact"].astype(str).to_numpy()
    q_addr = qdf["address_norm"].astype(str).to_numpy()
    q_acomp = qdf["address_compact"].astype(str).to_numpy()
    q_ctry = qdf["country_norm"].astype(str).to_numpy()
    q_post = qdf["postal"].astype(str).to_numpy()
    q_ids = qdf["entity_id"].astype(str).to_numpy()

    n = len(rows)
    data = {c: np.zeros(n, dtype=np.float32) for c in FEATURE_COLS}
    s1_idx = np.zeros(n, dtype=np.int32)
    t_idx = np.zeros(n, dtype=np.int32)
    s1_id = np.empty(n, dtype=object)
    target_id = np.empty(n, dtype=object)
    target_source = np.empty(n, dtype=object)

    for r, row in enumerate(rows):
        qi, ti, mask, block_score = row
        qn, tn = q_names[qi], index.names[ti]
        qa, ta = q_addr[qi], index.addresses[ti]
        qcn, tcn = q_compact[qi], index.compact_names[ti]
        qca, tca = q_acomp[qi], index.compact_addresses[ti]
        qc, tc = q_ctry[qi], index.countries[ti]
        qp, tp = q_post[qi], index.postals[ti]

        nr = fuzz.ratio(qn, tn) / 100.0 if qn and tn else 0.0
        nts = fuzz.token_set_ratio(qn, tn) / 100.0 if qn and tn else 0.0
        nsort = fuzz.token_sort_ratio(qn, tn) / 100.0 if qn and tn else 0.0
        npr = fuzz.partial_ratio(qn, tn) / 100.0 if qn and tn else 0.0
        ar = fuzz.ratio(qa, ta) / 100.0 if qa and ta else 0.0
        ats = fuzz.token_set_ratio(qa, ta) / 100.0 if qa and ta else 0.0
        asort = fuzz.token_sort_ratio(qa, ta) / 100.0 if qa and ta else 0.0
        apr = fuzz.partial_ratio(qa, ta) / 100.0 if qa and ta else 0.0

        qn_tok, tn_tok = set(qn.split()) if qn else set(), set(tn.split()) if tn else set()
        qa_tok, ta_tok = set(qa.split()) if qa else set(), set(ta.split()) if ta else set()

        data["name_ratio"][r] = nr
        data["name_token_set_ratio"][r] = nts
        data["name_token_sort_ratio"][r] = nsort
        data["name_partial_ratio"][r] = npr
        data["address_ratio"][r] = ar
        data["address_token_set_ratio"][r] = ats
        data["address_token_sort_ratio"][r] = asort
        data["address_partial_ratio"][r] = apr
        data["name_exact"][r] = 1.0 if qn and qn == tn else 0.0
        data["name_compact_exact"][r] = 1.0 if qcn and qcn == tcn else 0.0
        data["address_exact"][r] = 1.0 if qca and qca == tca else 0.0
        data["postal_exact"][r] = 1.0 if qp and tp and qp == tp else 0.0
        data["country_exact"][r] = 1.0 if qc and tc and qc == tc else 0.0
        data["name_prefix3"][r] = 1.0 if qn and tn and qn[:3] == tn[:3] else 0.0
        data["address_prefix3"][r] = 1.0 if qa and ta and qa[:3] == ta[:3] else 0.0
        data["name_contains"][r] = 1.0 if qn and tn and (qn in tn or tn in qn) else 0.0
        data["address_contains"][r] = 1.0 if qa and ta and (qa in ta or ta in qa) else 0.0
        data["name_length_ratio"][r] = _safe_len_ratio(qn, tn)
        data["address_length_ratio"][r] = _safe_len_ratio(qa, ta)
        data["name_token_jaccard"][r] = _jaccard(qn_tok, tn_tok)
        data["name_token_overlap"][r] = _overlap(qn_tok, tn_tok)
        data["address_token_jaccard"][r] = _jaccard(qa_tok, ta_tok)
        data["address_token_overlap"][r] = _overlap(qa_tok, ta_tok)
        data["name_char4_jaccard"][r] = _char_jaccard(qn, tn, 4)
        data["address_char3_jaccard"][r] = _char_jaccard(qa, ta, 3)
        qd, td = NON_DIGIT_RE.sub("", qa), NON_DIGIT_RE.sub("", ta)
        data["address_digit_ratio"][r] = fuzz.ratio(qd, td) / 100.0 if qd and td else 0.0
        data["name_addr_mean"][r] = 0.5 * (nr + ar)
        data["name_addr_geom"][r] = math.sqrt(max(0.0, nr * ar))
        data["name_minus_address"][r] = nr - ar
        data["block_exact_name"][r] = 1.0 if mask & BIT_EXACT_NAME else 0.0
        data["block_compact_name"][r] = 1.0 if mask & BIT_COMPACT_NAME else 0.0
        data["block_name_country"][r] = 1.0 if mask & BIT_NAME_COUNTRY else 0.0
        data["block_postal"][r] = 1.0 if mask & BIT_POSTAL else 0.0
        data["block_exact_address"][r] = 1.0 if mask & BIT_EXACT_ADDRESS else 0.0
        data["block_name_token"][r] = 1.0 if mask & BIT_NAME_TOKEN else 0.0
        data["block_address_token"][r] = 1.0 if mask & BIT_ADDRESS_TOKEN else 0.0
        data["block_name_char"][r] = 1.0 if mask & BIT_NAME_CHAR else 0.0
        data["block_count"][r] = float(int(mask).bit_count())
        data["block_score"][r] = float(block_score)
        data["target_is_s2"][r] = 1.0 if index.sources[ti] == "S2" else 0.0
        data["query_name_missing"][r] = 0.0 if qn else 1.0
        data["target_name_missing"][r] = 0.0 if tn else 1.0
        data["query_address_missing"][r] = 0.0 if qa else 1.0
        data["target_address_missing"][r] = 0.0 if ta else 1.0
        data["both_name_present"][r] = 1.0 if qn and tn else 0.0
        data["both_address_present"][r] = 1.0 if qa and ta else 0.0

        s1_idx[r] = qi
        t_idx[r] = ti
        s1_id[r] = q_ids[qi]
        target_id[r] = index.ids[ti]
        target_source[r] = index.sources[ti]

    feat = pd.DataFrame(data)
    feat.insert(0, "s1_idx", s1_idx)
    feat.insert(1, "target_idx", t_idx)
    feat.insert(2, "s1_id", s1_id)
    feat.insert(3, "target_id", target_id)
    feat.insert(4, "target_source", target_source)
    return feat
"""))

    c.append(md("## Cell 18 — Labels on the **final** candidate set"))
    c.append(code("""target_id_to_idx = {str(eid): i for i, eid in enumerate(train_target.entity_id)}

train_features_full = build_pair_features(train_s1_split, train_index, train_candidate_rows)
val_features_full = build_pair_features(val_s1_split, train_index, val_candidate_rows)

train_features_full["label"] = [
    int(tid in truth_map.get(s1, set()))
    for s1, tid in zip(train_features_full["s1_id"], train_features_full["target_id"])
]
val_features_full["label"] = [
    int(tid in truth_map.get(s1, set()))
    for s1, tid in zip(val_features_full["s1_id"], val_features_full["target_id"])
]

print("Train candidate pairs:", len(train_features_full))
print(train_features_full["label"].value_counts())
print("Validation candidate pairs:", len(val_features_full))
print(val_features_full["label"].value_counts())
"""))

    c.append(md("""## Cell 19 — Inject blocked-out positives (training only)

Missed true pairs cannot be recovered at inference. They are still useful as labelled positives for the scorer. Validation metrics stay on the real candidate set.
"""))
    c.append(code("""def build_missing_positive_rows(qdf, truth, candidate_map, id_to_idx):
    rows = []
    ids = qdf["entity_id"].astype(str).to_numpy()
    for s1_idx, sid in enumerate(ids):
        missing = truth.get(sid, set()) - candidate_map.get(sid, set())
        for tid in missing:
            tidx = id_to_idx.get(tid)
            if tidx is not None:
                rows.append((s1_idx, tidx, 0, 0.0))
    return rows


train_missing_rows = build_missing_positive_rows(
    train_s1_split, truth_map, train_candidate_map, target_id_to_idx
)
print("Training positives missing from blocking:", len(train_missing_rows))

if train_missing_rows:
    train_missing_features = build_pair_features(train_s1_split, train_index, train_missing_rows)
    train_missing_features["label"] = 1
    train_features_augmented = pd.concat([train_features_full, train_missing_features], ignore_index=True)
    del train_missing_features
else:
    train_features_augmented = train_features_full
print("Augmented train rows:", len(train_features_augmented))
print(train_features_augmented["label"].value_counts())
"""))

    c.append(md("## Cell 20 — Hard-negative mining"))
    c.append(code("""def sample_training_pairs(df, seed=42, hard_ratio=6, singleton_negatives=4):
    rng = np.random.RandomState(seed)
    parts = []
    for s1_id, group in df.groupby("s1_id", sort=False):
        pos = group[group["label"] == 1]
        neg = group[group["label"] == 0]
        parts.append(pos)
        if len(neg) == 0:
            continue
        desired = max(6, len(pos) * hard_ratio) if len(pos) else singleton_negatives
        desired = min(desired, len(neg))
        score = (
            0.65 * neg["name_ratio"].to_numpy()
            + 0.25 * neg["address_ratio"].to_numpy()
            + 0.10 * (neg["block_score"].to_numpy() / max(1.0, float(neg["block_score"].max())))
        )
        order = np.argsort(-score)
        hard_n = max(1, int(desired * 0.75))
        hard_idx = order[:hard_n]
        remain = order[hard_n:]
        rand_n = min(desired - hard_n, len(remain))
        if rand_n > 0:
            pick = rng.choice(remain, size=rand_n, replace=False)
            take = np.concatenate([hard_idx, pick])
        else:
            take = hard_idx[:desired]
        parts.append(neg.iloc[take])
    return pd.concat(parts, ignore_index=True)


train_model_df = sample_training_pairs(train_features_augmented, seed=SEED)
print("Final pair-training rows:", len(train_model_df))
print(train_model_df["label"].value_counts())
print("Positive rate:", round(float(train_model_df["label"].mean()), 6))
"""))

    c.append(md("## Cell 21 — Train two LightGBM pair scorers (GPU if the wheel supports it)"))
    c.append(code("""X_train = train_model_df[FEATURE_COLS].to_numpy(np.float32)
y_train = train_model_df["label"].to_numpy(np.int8)
X_val = val_features_full[FEATURE_COLS].to_numpy(np.float32)
y_val = val_features_full["label"].to_numpy(np.int8)

positive_count = max(1, int(y_train.sum()))
negative_count = max(1, int((y_train == 0).sum()))
scale_pos_weight = min(8.0, negative_count / positive_count)
print("pos", positive_count, "neg", negative_count, "spw", scale_pos_weight)


def lgbm_params(seed, num_leaves, extra=None):
    p = {
        "objective": "binary",
        "n_estimators": CFG["lgbm_trees_cap"],
        "learning_rate": 0.04,
        "num_leaves": num_leaves,
        "min_child_samples": 50,
        "subsample": 0.88,
        "subsample_freq": 1,
        "colsample_bytree": 0.88,
        "reg_alpha": 0.15,
        "reg_lambda": 2.0,
        "max_bin": 255,
        "verbosity": -1,
        "random_state": seed,
        "n_jobs": -1,
        "scale_pos_weight": scale_pos_weight,
    }
    if extra:
        p.update(extra)
    return p


def fit_lgbm(params):
    tried = dict(params)
    if USE_GPU:
        tried["device"] = "gpu"
    model = LGBMClassifier(**tried)
    try:
        model.fit(
            X_train,
            y_train,
            eval_set=[(X_val, y_val)],
            eval_metric=["binary_logloss", "auc"],
            callbacks=[lgb.early_stopping(stopping_rounds=80, verbose=False), lgb.log_evaluation(0)],
        )
        return model
    except Exception as exc:
        print("GPU/fit fallback:", exc)
        tried.pop("device", None)
        tried.pop("device_type", None)
        model = LGBMClassifier(**tried)
        model.fit(
            X_train,
            y_train,
            eval_set=[(X_val, y_val)],
            eval_metric=["binary_logloss", "auc"],
            callbacks=[lgb.early_stopping(stopping_rounds=80, verbose=False), lgb.log_evaluation(0)],
        )
        return model


t0 = time.time()
model_a = fit_lgbm(lgbm_params(42, 96))
print("Model A best_iteration:", model_a.best_iteration_)
model_b = fit_lgbm(lgbm_params(2026, 48, extra={"max_depth": 12}))
print("Model B best_iteration:", model_b.best_iteration_)
print("Training time:", round(time.time() - t0, 2), "sec")
"""))

    c.append(md("## Cell 22 — Pair-level diagnostics (not the leaderboard metric)"))
    c.append(code("""val_score_a = model_a.predict_proba(X_val)[:, 1]
val_score_b = model_b.predict_proba(X_val)[:, 1]
val_score_ens = 0.5 * val_score_a + 0.5 * val_score_b

for name, score in [("Model A", val_score_a), ("Model B", val_score_b), ("Ensemble", val_score_ens)]:
    print(
        name,
        "ROC-AUC", round(roc_auc_score(y_val, score), 6),
        "PR-AUC", round(average_precision_score(y_val, score), 6),
    )
"""))

    c.append(md("## Cell 23 — Official-style entity F0.5 + resolver"))
    c.append(code("""BETA2 = 0.25  # F0.5 uses beta^2 = 0.25


def entity_f05(truth, predictions):
    scores = []
    tp_total = fp_total = fn_total = 0
    singleton_correct = singleton_false = 0
    for s1_id, true_set in truth.items():
        true_set = set(true_set)
        pred_set = set(predictions.get(s1_id, set()))
        tp = len(true_set & pred_set)
        fp = len(pred_set - true_set)
        fn = len(true_set - pred_set)
        tp_total += tp
        fp_total += fp
        fn_total += fn
        if not true_set and not pred_set:
            scores.append(1.0)
            singleton_correct += 1
        elif not true_set and pred_set:
            scores.append(0.0)
            singleton_false += 1
        else:
            precision = tp / (tp + fp) if tp + fp else 0.0
            recall = tp / (tp + fn) if tp + fn else 0.0
            if precision == 0.0 and recall == 0.0:
                scores.append(0.0)
            else:
                scores.append((1 + BETA2) * precision * recall / (BETA2 * precision + recall))
    return {
        "macro_f05": float(np.mean(scores)) if scores else 0.0,
        "micro_precision": tp_total / (tp_total + fp_total) if tp_total + fp_total else 0.0,
        "micro_recall": tp_total / (tp_total + fn_total) if tp_total + fn_total else 0.0,
        "tp": tp_total,
        "fp": fp_total,
        "fn": fn_total,
        "singleton_correct": singleton_correct,
        "singleton_false_merge": singleton_false,
        "n_entities": len(truth),
    }


def group_scores(feature_df, scores):
    # Pre-group per S1 for fast policy search.
    s1_ids = feature_df["s1_id"].astype(str).to_numpy()
    tgt = feature_df["target_id"].astype(str).to_numpy()
    src = feature_df["target_source"].astype(str).to_numpy()
    grouped = defaultdict(list)
    for i in range(len(feature_df)):
        grouped[s1_ids[i]].append((float(scores[i]), src[i], tgt[i]))
    for sid in grouped:
        grouped[sid].sort(key=lambda x: -x[0])
    return grouped


def resolve_grouped(grouped, all_s1, threshold_s2, threshold_s3, multi_delta, margin):
    predictions = {sid: set() for sid in all_s1}
    for sid, rows in grouped.items():
        if not rows:
            continue
        best = rows[0][0]
        matches = set()
        for rank, (score, source, tid) in enumerate(rows):
            thresh = threshold_s2 if source == "S2" else threshold_s3
            if score < thresh:
                break
            if rank == 0:
                matches.add(tid)
                continue
            if score >= thresh + multi_delta and (best - score) <= margin:
                matches.add(tid)
            else:
                break
        predictions[sid] = matches
    return predictions
"""))

    c.append(md("""## Cell 24 — Policy search on validation S1

Searches absolute threshold, S2 vs S3, multi-match floor, and best-vs-second margin. **Not** a hardcoded 0.50.
"""))
    c.append(code("""def search_policy(feature_df, scores, truth):
    grouped = group_scores(feature_df, scores)
    all_s1 = list(truth.keys())
    best = None

    thresholds = np.round(np.arange(0.52, 0.931, 0.03), 4)
    multi_deltas = [0.00, 0.03, 0.06, 0.10]
    margins = [0.04, 0.08, 0.12, 0.16]

    for t in thresholds:
        for delta in multi_deltas:
            for margin in margins:
                pred = resolve_grouped(grouped, all_s1, float(t), float(t), float(delta), float(margin))
                metric = entity_f05(truth, pred)
                if best is None or metric["macro_f05"] > best["macro_f05"]:
                    best = {
                        "threshold_s2": float(t),
                        "threshold_s3": float(t),
                        "multi_delta": float(delta),
                        "margin": float(margin),
                        **metric,
                    }

    center2, center3 = best["threshold_s2"], best["threshold_s3"]
    grid2 = np.round(np.arange(max(0.45, center2 - 0.08), min(0.96, center2 + 0.081), 0.02), 4)
    grid3 = np.round(np.arange(max(0.45, center3 - 0.08), min(0.96, center3 + 0.081), 0.02), 4)
    for t2 in grid2:
        for t3 in grid3:
            pred = resolve_grouped(
                grouped, all_s1, float(t2), float(t3), best["multi_delta"], best["margin"]
            )
            metric = entity_f05(truth, pred)
            if metric["macro_f05"] > best["macro_f05"]:
                best = {
                    "threshold_s2": float(t2),
                    "threshold_s3": float(t3),
                    "multi_delta": best["multi_delta"],
                    "margin": best["margin"],
                    **metric,
                }

    for delta in [0.00, 0.02, 0.04, 0.06, 0.08, 0.10, 0.14]:
        for margin in [0.03, 0.06, 0.10, 0.14, 0.18]:
            pred = resolve_grouped(
                grouped, all_s1, best["threshold_s2"], best["threshold_s3"], float(delta), float(margin)
            )
            metric = entity_f05(truth, pred)
            if metric["macro_f05"] > best["macro_f05"]:
                best = {
                    "threshold_s2": best["threshold_s2"],
                    "threshold_s3": best["threshold_s3"],
                    "multi_delta": float(delta),
                    "margin": float(margin),
                    **metric,
                }
    return best, grouped
"""))

    c.append(md("## Cell 25 — Select model by entity F0.5"))
    c.append(code("""val_truth_map = {sid: truth_map.get(sid, set()) for sid in val_s1_split["entity_id"].astype(str)}
POLICIES = {}
GROUPED = {}

for model_name, scores in [("model_a", val_score_a), ("model_b", val_score_b), ("ensemble", val_score_ens)]:
    policy, grouped = search_policy(val_features_full, scores, val_truth_map)
    POLICIES[model_name] = policy
    GROUPED[model_name] = grouped
    print("\\n====", model_name, "====")
    print(json.dumps(policy, indent=2))

best_model_name = max(POLICIES, key=lambda k: POLICIES[k]["macro_f05"])
best_policy = POLICIES[best_model_name]
print("\\nSELECTED MODEL:", best_model_name)
print("Validation macro F0.5:", best_policy["macro_f05"])
print("This number is computed on the held-out S1 sample — do not replace it with a guess.")
"""))

    c.append(md("## Cell 26 — Validation error analysis"))
    c.append(code("""selected_val_scores = {"model_a": val_score_a, "model_b": val_score_b, "ensemble": val_score_ens}[best_model_name]
val_predictions = resolve_grouped(
    GROUPED[best_model_name],
    list(val_truth_map),
    best_policy["threshold_s2"],
    best_policy["threshold_s3"],
    best_policy["multi_delta"],
    best_policy["margin"],
)
val_metric = entity_f05(val_truth_map, val_predictions)
print(json.dumps(val_metric, indent=2))

rows = []
for s1_id, truth_ids in val_truth_map.items():
    pred_ids = val_predictions.get(s1_id, set())
    rows.append({
        "source1_entity_id": s1_id,
        "true_count": len(truth_ids),
        "pred_count": len(pred_ids),
        "tp": len(truth_ids & pred_ids),
        "fp": len(pred_ids - truth_ids),
        "fn": len(truth_ids - pred_ids),
        "singleton_truth": int(len(truth_ids) == 0),
    })
diagnostics_df = pd.DataFrame(rows)
print("\\nFalse-positive-heavy entities:")
print(diagnostics_df.query("fp > 0").sort_values(["fp", "true_count"], ascending=False).head(15))
print("\\nMiss-heavy entities:")
print(diagnostics_df.query("fn > 0").sort_values(["fn", "true_count"], ascending=False).head(15))
print("\\nSingleton false-merge rate:",
      float(diagnostics_df.query("singleton_truth == 1 and pred_count > 0").shape[0]
            / max(1, diagnostics_df.query("singleton_truth == 1").shape[0])))
"""))

    c.append(md("## Cell 27 — Exact name+country baseline"))
    c.append(code("""exact_baseline = {sid: set() for sid in val_truth_map}
mask = (val_features_full["name_exact"] == 1) & (val_features_full["country_exact"] == 1)
for row in val_features_full.loc[mask, ["s1_id", "target_id"]].itertuples(index=False):
    exact_baseline[row.s1_id].add(row.target_id)
print("Exact normalized name + country baseline:")
print(json.dumps(entity_f05(val_truth_map, exact_baseline), indent=2))
print("Selected model F0.5:", best_policy["macro_f05"])
"""))

    c.append(md("## Cell 28 — Retrain on train+val sampled S1 (policy frozen)"))
    c.append(code("""val_missing_rows = build_missing_positive_rows(
    val_s1_split, truth_map, val_candidate_map, target_id_to_idx
)
if val_missing_rows:
    val_missing_features = build_pair_features(val_s1_split, train_index, val_missing_rows)
    val_missing_features["label"] = 1
    val_features_augmented = pd.concat([val_features_full, val_missing_features], ignore_index=True)
    del val_missing_features
else:
    val_features_augmented = val_features_full

full_pair_features = pd.concat([train_features_augmented, val_features_augmented], ignore_index=True)
full_model_df = sample_training_pairs(full_pair_features, seed=SEED)
print("Full sampled training rows:", len(full_model_df))
print(full_model_df["label"].value_counts())

X_full = full_model_df[FEATURE_COLS].to_numpy(np.float32)
y_full = full_model_df["label"].to_numpy(np.int8)
pos_full = max(1, int(y_full.sum()))
neg_full = max(1, int((y_full == 0).sum()))
scale_full = min(8.0, neg_full / pos_full)

best_iter_a = int(max(250, min(model_a.best_iteration_ or 600, CFG["lgbm_trees_cap"])))
best_iter_b = int(max(250, min(model_b.best_iteration_ or 600, CFG["lgbm_trees_cap"])))


def fit_final(base_params, n_estimators, spw):
    params = dict(base_params)
    params["n_estimators"] = n_estimators
    params["scale_pos_weight"] = spw
    params.pop("device_type", None)
    params.pop("device", None)
    if USE_GPU:
        params["device"] = "gpu"
    model = LGBMClassifier(**params)
    try:
        model.fit(X_full, y_full, callbacks=[lgb.log_evaluation(0)])
        return model
    except Exception as exc:
        print("final fit fallback:", exc)
        params.pop("device", None)
        params.pop("device_type", None)
        model = LGBMClassifier(**params)
        model.fit(X_full, y_full, callbacks=[lgb.log_evaluation(0)])
        return model


print("Retraining final models...")
final_model_a = fit_final(lgbm_params(42, 96), best_iter_a, scale_full)
final_model_b = fit_final(lgbm_params(2026, 48, extra={"max_depth": 12}), best_iter_b, scale_full)
final_models = ["model_a", "model_b"] if best_model_name == "ensemble" else [best_model_name]
print("Final inference models:", final_models)
"""))

    c.append(md("## Cell 29 — Save models + locked policy"))
    c.append(code("""if "model_a" in final_models:
    joblib.dump(final_model_a, MODEL_DIR / "lightgbm_model_a.joblib")
if "model_b" in final_models:
    joblib.dump(final_model_b, MODEL_DIR / "lightgbm_model_b.joblib")

with open(ARTIFACT_DIR / "decision_policy.json", "w", encoding="utf-8") as f:
    json.dump(
        {
            "selected_model": best_model_name,
            "policy": best_policy,
            "validation_metrics": val_metric,
            "blocking_metrics": val_block_report,
            "cfg": CFG,
            "fair_play": "no_external_lookup",
        },
        f,
        indent=2,
    )
print("Wrote", ARTIFACT_DIR / "decision_policy.json")
print(open(ARTIFACT_DIR / "decision_policy.json", encoding="utf-8").read())
"""))

    c.append(md("## Cell 30 — Free training-side memory"))
    c.append(code("""del train_features_full, train_features_augmented, val_features_full, val_features_augmented
del full_pair_features, full_model_df, train_model_df
del X_train, y_train, X_val, y_val, X_full, y_full
del train_candidate_rows, val_candidate_rows, train_candidate_map, val_candidate_map
del train_index, train_target, target_id_to_idx
gc.collect()
print("Training-side large objects released. RAM:", round(ram_gb(), 2), "GB")
"""))

    c.append(md("## Cell 31 — Test S2/S3 index"))
    c.append(code("""t0 = time.time()
print("Loading test S2 + S3...")
test_target = load_target_store([PATHS["test_source2.tsv"], PATHS["test_source3.tsv"]])
print("Test targets:", len(test_target), f"in {time.time()-t0:.1f}s")
test_index = ERBlockIndex(test_target)
print("Test index + load RAM:", round(ram_gb(), 2), "GB")
"""))

    c.append(md("""## Cell 32 — Stream test S1 → write portal files

Every test S1 gets one row. Matches are a subset of that entity's **final** candidates. France is just another `country` string — it is not dropped.
"""))
    c.append(code("""MATCHING_PATH = OUTPUT_DIR / "matching_results.tsv"
CANDIDATE_PATH = OUTPUT_DIR / "candidate_pairs.tsv"

threshold_s2 = best_policy["threshold_s2"]
threshold_s3 = best_policy["threshold_s3"]
multi_delta = best_policy["multi_delta"]
margin = best_policy["margin"]


def predict_scores(feature_df):
    X = feature_df[FEATURE_COLS].to_numpy(np.float32)
    if best_model_name == "model_a":
        return final_model_a.predict_proba(X)[:, 1]
    if best_model_name == "model_b":
        return final_model_b.predict_proba(X)[:, 1]
    return 0.5 * final_model_a.predict_proba(X)[:, 1] + 0.5 * final_model_b.predict_proba(X)[:, 1]


total_s1 = total_candidates = total_matches = 0
zero_candidate_entities = matched_entities = 0
max_candidates_seen = 0
start_total = time.time()

with open(MATCHING_PATH, "w", encoding="utf-8", buffering=1024 * 1024) as matching_f, open(
    CANDIDATE_PATH, "w", encoding="utf-8", buffering=1024 * 1024
) as candidate_f:
    matching_f.write("source1_entity_id\\tmatched_entity_ids\\n")
    candidate_f.write("source1_entity_id\\tcandidate_entity_ids\\n")

    for chunk_no, qchunk_raw in enumerate(
        pd.read_csv(
            PATHS["test_source1.tsv"],
            sep="\\t",
            dtype="string",
            keep_default_na=False,
            na_filter=False,
            usecols=REQUIRED_RECORD_COLUMNS,
            chunksize=CFG["test_chunk"],
            low_memory=False,
            encoding="utf-8",
        ),
        start=1,
    ):
        qchunk = normalize_query_frame(qchunk_raw)
        candidate_rows, candidate_map = generate_candidates_for_dataframe(
            qchunk, test_index, progress_every=None
        )
        total_s1 += len(qchunk)

        ordered_s1 = qchunk["entity_id"].astype(str).tolist()
        for sid in ordered_s1:
            cands = sorted(candidate_map.get(sid, set()))
            if cands:
                total_candidates += len(cands)
            else:
                zero_candidate_entities += 1
            max_candidates_seen = max(max_candidates_seen, len(cands))
            candidate_f.write(sid + "\\t" + ",".join(cands) + "\\n")

        if candidate_rows:
            feature_df = build_pair_features(qchunk, test_index, candidate_rows)
            scores = predict_scores(feature_df)
            grouped = group_scores(feature_df, scores)
            predictions = resolve_grouped(
                grouped, ordered_s1, threshold_s2, threshold_s3, multi_delta, margin
            )
            del feature_df, scores, grouped
        else:
            predictions = {sid: set() for sid in ordered_s1}

        for sid in ordered_s1:
            matches = sorted(predictions.get(sid, set()))
            cands = candidate_map.get(sid, set())
            if not set(matches).issubset(cands):
                raise RuntimeError(f"Invariant failed for {sid}: match not in final candidates")
            if matches:
                matched_entities += 1
                total_matches += len(matches)
            matching_f.write(sid + "\\t" + ",".join(matches) + "\\n")

        del qchunk, qchunk_raw, candidate_rows, candidate_map, predictions
        if chunk_no % 5 == 0:
            gc.collect()
            elapsed = time.time() - start_total
            print(
                f"Chunk {chunk_no} | S1={total_s1:,} | candidates={total_candidates:,} | "
                f"matches={total_matches:,} | RAM={ram_gb():.1f}GB | {elapsed/60:.2f} min"
            )

print("\\nFINAL TEST INFERENCE COMPLETE")
print("S1 entities:", total_s1)
print("Candidate pairs:", total_candidates)
print("Predicted matches:", total_matches)
print("Matched S1 entities:", matched_entities)
print("No-candidate S1:", zero_candidate_entities)
print("Maximum candidates/entity:", max_candidates_seen)
print("Runtime minutes:", round((time.time() - start_total) / 60, 2))
print("Matching file:", MATCHING_PATH)
print("Candidate file:", CANDIDATE_PATH)
"""))

    c.append(md("## Cell 33 — Inspect generated TSVs"))
    c.append(code("""print("MATCHING RESULTS")
print(pd.read_csv(MATCHING_PATH, sep="\\t", dtype="string", nrows=12).to_string(index=False))
print("\\nCANDIDATE PAIRS")
print(pd.read_csv(CANDIDATE_PATH, sep="\\t", dtype="string", nrows=12).to_string(index=False))
print("\\nFile sizes (MB):")
print("matching_results.tsv", MATCHING_PATH.stat().st_size / 1e6)
print("candidate_pairs.tsv", CANDIDATE_PATH.stat().st_size / 1e6)
"""))

    c.append(md("""## Cell 34 — Official `validate_submission.py` (stdlib)

Writes the challenge validator next to the outputs and runs it. Default mode skips the memory-heavy ID-existence set; Cell 35 can enable `--check-ids` if RAM allows.
"""))
    c.append(code("""VALIDATOR_DST = WORK_DIR / "validate_submission.py"
validator_copied = False
explicit = [
    Path.cwd() / "student_resource" / "utils" / "validate_submission.py",
    Path("/kaggle/working") / "validate_submission.py",
]
for cand in explicit:
    if cand.exists() and cand.resolve() != VALIDATOR_DST.resolve():
        shutil.copy(cand, VALIDATOR_DST)
        validator_copied = True
        print("Using official validator at", cand)
        break
if not validator_copied and Path("/kaggle/input").exists():
    for cand in Path("/kaggle/input").rglob("validate_submission.py"):
        shutil.copy(cand, VALIDATOR_DST)
        validator_copied = True
        print("Using official validator at", cand)
        break

if not validator_copied:
    # Embedded official-style validator (stdlib). Same rules as student_resource/utils/validate_submission.py
    VALIDATOR_DST.write_text(r'''#!/usr/bin/env python3
import argparse, os, sys
DELIM = "\\t"
MAX_EXAMPLES = 5
MATCHING_HEADER = ["source1_entity_id", "matched_entity_ids"]
CANDIDATE_HEADER = ["source1_entity_id", "candidate_entity_ids"]

def read_ids(path):
    with open(path, encoding="utf-8") as f:
        next(f, None)
        return {line.split(DELIM, 1)[0].strip() for line in f if line.strip()}

def examples(items):
    items = sorted(items)
    shown = ", ".join(items[:MAX_EXAMPLES])
    if len(items) > MAX_EXAMPLES:
        return f"{len(items)} total, e.g. {shown}, ..."
    return shown

def load_match_targets(test_dir, warnings):
    targets = set()
    for name in ("test_source2.tsv", "test_source3.tsv"):
        path = os.path.join(test_dir, name)
        if not os.path.isfile(path):
            warnings.append(f"{path} not found — skipping ID existence check.")
            return None
        targets |= read_ids(path)
    return targets

def validate_id_list_file(path, expected_header, col_label, required, valid_ids, errors):
    if not os.path.isfile(path):
        errors.append(f"File not found: {path}")
        return None
    name = os.path.basename(path)
    mapping, seen, dup_rows, intra_dupes = {}, set(), set(), set()
    self_matches, wrong_prefix, unknown = set(), set(), set()
    n_rows = empties = 0
    with open(path, encoding="utf-8") as f:
        header = f.readline()
        if not header:
            errors.append(f"{name} is empty.")
            return None
        if DELIM not in header and "," in header:
            errors.append(f"{name}: looks COMMA-separated; must be TAB-separated.")
            return None
        cols = [c.strip().lower() for c in header.rstrip("\\n").split(DELIM)]
        if cols != expected_header:
            errors.append(f"{name}: unexpected header {cols}. Expected {expected_header}.")
            return None
        for line_num, line in enumerate(f, start=2):
            s1, tab, rest = line.partition(DELIM)
            if not tab:
                if s1.strip():
                    errors.append(f"{name}: malformed row at line {line_num}")
                continue
            n_rows += 1
            if s1 in seen:
                dup_rows.add(s1)
            seen.add(s1)
            ids = rest.rstrip("\\n").split(",") if rest.strip() else []
            if not ids:
                empties += 1
                mapping[s1] = set()
                continue
            if len(ids) != len(set(ids)):
                intra_dupes.add(s1)
            id_set = set(ids)
            mapping[s1] = id_set
            for mid in id_set:
                if mid.startswith("S1-"):
                    self_matches.add(mid)
                elif not mid.startswith(("S2-", "S3-")):
                    wrong_prefix.add(mid)
                elif valid_ids is not None and mid not in valid_ids:
                    unknown.add(mid)
    findings = [
        (dup_rows, "{name}: duplicate source1_entity_id row(s): {ex}."),
        (intra_dupes, "{name}: repeated ID inside a {col} list for: {ex}."),
        (self_matches, "{name}: {col} contains Source-1 IDs: {ex}."),
        (wrong_prefix, "{name}: {col} contains IDs without S2-/S3- prefix: {ex}."),
        (unknown, "{name}: {col} references IDs not in test S2/S3: {ex}."),
        (required - seen, "{name}: required S1 entity(ies) missing: {ex}."),
        (seen - required, "{name}: S1 ID not in the test set: {ex}."),
    ]
    for offenders, message in findings:
        if offenders:
            errors.append(message.format(name=name, ex=examples(offenders), col=col_label))
    print(f"  {name}: {n_rows} rows ({empties} empty, {n_rows - empties} non-empty).")
    return mapping

def validate(matching_path, candidate_path, test_dir, check_ids=False):
    errors, warnings = [], []
    source1 = os.path.join(test_dir, "test_source1.tsv")
    if not os.path.isfile(source1):
        errors.append(f"Test source1 file not found: {source1}")
        return errors, warnings
    required = read_ids(source1)
    print(f"  required S1 entities: {len(required)}")
    if check_ids:
        valid_ids = load_match_targets(test_dir, warnings)
        if valid_ids is not None:
            print(f"  valid S2/S3 match IDs: {len(valid_ids)}")
    else:
        valid_ids = None
        warnings.append("ID-existence check is OFF.")
    matched = validate_id_list_file(matching_path, MATCHING_HEADER, "matched_entity_ids", required, valid_ids, errors)
    candidate = None
    if candidate_path and os.path.isfile(candidate_path):
        candidate = validate_id_list_file(candidate_path, CANDIDATE_HEADER, "candidate_entity_ids", required, valid_ids, errors)
    elif candidate_path:
        warnings.append(f"{candidate_path} not found — skipping candidate checks.")
    if matched is not None and candidate is not None:
        offenders = {s1 for s1, mids in matched.items() if mids - candidate.get(s1, set())}
        if offenders:
            warnings.append(f"{len(offenders)} S1 have matches absent from candidate_pairs.tsv, e.g. {examples(offenders)}.")
    return errors, warnings

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--matching", "-m", required=True)
    p.add_argument("--candidate", "-c", default=None)
    p.add_argument("--test-dir", "-t", required=True)
    p.add_argument("--check-ids", action="store_true")
    args = p.parse_args()
    candidate_path = args.candidate
    print("ML Challenge 2026 — submission validator")
    errors, warnings = validate(args.matching, candidate_path, args.test_dir, check_ids=args.check_ids)
    print()
    for warning in warnings:
        print(f"WARNING: {warning}")
    if errors:
        print(f"FAIL — {len(errors)} issue(s):")
        for i, error in enumerate(errors, 1):
            print(f"  {i}. {error}")
        return 1
    print("PASS — no blocking issues found. Safe to submit.")
    return 0

if __name__ == "__main__":
    sys.exit(main())
''')

TEST_DIR = PATHS["test_source1.tsv"].parent
cmd = [
    sys.executable,
    str(VALIDATOR_DST),
    "--matching", str(MATCHING_PATH),
    "--candidate", str(CANDIDATE_PATH),
    "--test-dir", str(TEST_DIR),
]
print("Running:", " ".join(cmd))
proc = subprocess.run(cmd, capture_output=True, text=True)
print(proc.stdout)
if proc.stderr:
    print(proc.stderr)
print("validator exit:", proc.returncode)
if proc.returncode != 0:
    raise SystemExit("Validator FAILED — do not upload matching_results.tsv")
print("CELL 34 GATE: PASS")
"""))

    c.append(md("""## Cell 35 — Optional `--check-ids`

Loads all test S2/S3 IDs (a few GB). Skip if RAM is tight; a missing ID hurts score rather than portal rejection.
"""))
    c.append(code("""RUN_CHECK_IDS = False  # set True only if you still have several GB free after the test index
print("Attempting --check-ids:", RUN_CHECK_IDS)
if RUN_CHECK_IDS:
    cmd = [
        sys.executable,
        str(VALIDATOR_DST),
        "--matching", str(MATCHING_PATH),
        "--candidate", str(CANDIDATE_PATH),
        "--test-dir", str(TEST_DIR),
        "--check-ids",
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    print(proc.stdout)
    if proc.stderr:
        print(proc.stderr)
    print("check-ids exit:", proc.returncode)
else:
    print("Skipped --check-ids to protect RAM.")
"""))

    c.append(md("## Cell 36 — Streaming coverage: one matching row per test S1, in order"))
    c.append(code("""def assert_aligned(test_s1_path, matching_path, candidate_path):
    n = 0
    with open(test_s1_path, encoding="utf-8") as sf, open(matching_path, encoding="utf-8") as mf, open(
        candidate_path, encoding="utf-8"
    ) as cf:
        sh, mh, ch = sf.readline(), mf.readline(), cf.readline()
        if not mh.startswith("source1_entity_id\\tmatched_entity_ids"):
            raise ValueError("bad matching header")
        if not ch.startswith("source1_entity_id\\tcandidate_entity_ids"):
            raise ValueError("bad candidate header")
        while True:
            s, m, c = sf.readline(), mf.readline(), cf.readline()
            if not s and not m and not c:
                break
            if not (s and m and c):
                raise ValueError("row-count mismatch among test S1 / matching / candidates")
            s1 = s.split("\\t", 1)[0].strip()
            m1 = m.split("\\t", 1)[0].strip()
            c1 = c.split("\\t", 1)[0].strip()
            if not (s1 == m1 == c1):
                raise ValueError(f"ID order mismatch at row {n+1}: {s1} / {m1} / {c1}")
            n += 1
    return n

n_aligned = assert_aligned(PATHS["test_source1.tsv"], MATCHING_PATH, CANDIDATE_PATH)
print("Aligned S1 rows:", n_aligned)
print("CELL 36 GATE: PASS")
"""))

    c.append(md("## Cell 37 — France / open-set country sanity (test S1 vs outputs)"))
    c.append(code("""# Confirm every test S1 country, including France, still has a submission row.
# We do not filter on country anywhere in inference.
france_n = 0
total_n = 0
with open(PATHS["test_source1.tsv"], encoding="utf-8") as f:
    header = f.readline().rstrip("\\n").split("\\t")
    ci = header.index("country")
    for line in f:
        if not line.strip():
            continue
        total_n += 1
        parts = line.rstrip("\\n").split("\\t")
        if len(parts) > ci and parts[ci].strip().casefold() in {"fr", "france"}:
            france_n += 1
print("Test S1 rows:", total_n, "France-labelled (heuristic):", france_n)
print("Output S1 rows:", n_aligned)
assert n_aligned == total_n, "Output does not cover every test S1 row"
print("CELL 37 GATE: PASS")
"""))

    c.append(md("## Cell 38 — Final summary (what to upload)"))
    c.append(code("""summary = {
    "selected_model": best_model_name,
    "policy": {
        "threshold_s2": best_policy["threshold_s2"],
        "threshold_s3": best_policy["threshold_s3"],
        "multi_delta": best_policy["multi_delta"],
        "margin": best_policy["margin"],
    },
    "validation_macro_f05": best_policy["macro_f05"],
    "validation_blocking_pair_recall": val_block_report["pair_blocking_recall"],
    "validation_blocking_entity_coverage": val_block_report["non_singleton_entity_coverage"],
    "test_s1": total_s1,
    "test_candidate_pairs": total_candidates,
    "test_predicted_matches": total_matches,
    "matching_results": str(MATCHING_PATH),
    "candidate_pairs": str(CANDIDATE_PATH),
    "leaderboard_upload": str(MATCHING_PATH),
}
print(json.dumps(summary, indent=2))
print("\\nUPLOAD TO PORTAL:", MATCHING_PATH)
print("PACKAGE ALSO NEEDS:", CANDIDATE_PATH)
print("Do not submit until Cell 34 printed PASS.")
"""))

    c.append(md("""## Cell 39 — Optional output zip for the final package `output/` folder

The live leaderboard wants **only** `matching_results.tsv`. The zip below is for the final team archive.
"""))
    c.append(code("""import zipfile

zip_path = WORK_DIR / "er_output_files.zip"
with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
    zf.write(MATCHING_PATH, arcname="output/matching_results.tsv")
    zf.write(CANDIDATE_PATH, arcname="output/candidate_pairs.tsv")
print("Wrote", zip_path, zip_path.stat().st_size / 1e6, "MB")
"""))

    c.append(md("""## Cell 40 — Operator checklist

1. Cell 16 pair-blocking recall is the **recall ceiling**. If it is low, raise `max_candidates` / token df — do not lower the match threshold first.
2. Cell 25 `macro_f05` is the number you trust before test inference. It is **not** the public leaderboard.
3. Cell 34 must print `PASS` before portal upload.
4. Upload `/kaggle/working/output/matching_results.tsv`.
5. Keep `candidate_pairs.tsv` for the final zip (`output/` + `code/` + `Documentation_template.md`).
6. No external business lookup was used: normalization is local string ops only (`anyascii`, abbreviation maps).
"""))
    c.append(code("""print("Checklist file pointers")
print(" matching:", MATCHING_PATH.exists(), MATCHING_PATH)
print(" candidate:", CANDIDATE_PATH.exists(), CANDIDATE_PATH)
print(" policy:", (ARTIFACT_DIR / "decision_policy.json").exists())
print(" validator exit was required PASS in Cell 34")
print("Done.")
"""))

    return c


def main():
    nb = {
        "nbformat": 4,
        "nbformat_minor": 5,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "pygments_lexer": "ipython3"},
            "kaggle": {"accelerator": "nvidiaTeslaT4", "isInternetEnabled": True, "isGpuEnabled": True},
        },
        "cells": cells(),
    }
    OUT.write_text(json.dumps(nb, indent=1), encoding="utf-8")
    print("Wrote", OUT, "cells", len(nb["cells"]))


if __name__ == "__main__":
    main()
