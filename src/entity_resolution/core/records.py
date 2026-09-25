"""
core/records.py — Canonical record representation.

Every record from S1/S2/S3 is loaded into a normalized columnar
representation. Normalization happens ONCE at ingestion.

The RecordStore holds all records from one or more sources as
a DataFrame with pre-computed normalized fields.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Set, Tuple

import numpy as np
import pandas as pd

from src.entity_resolution.core import EntityID, Source


# ──────────────────────────────────────────────────────────────
# NORMALIZATION FUNCTIONS (stateless, composable)
# ──────────────────────────────────────────────────────────────

def normalize_unicode(text: str) -> str:
    """NFKD normalize, strip accents, keep ASCII + digits + basic punct."""
    if not text:
        return ""
    text = unicodedata.normalize("NFKD", text)
    # Remove combining marks (accents/diacritics)
    text = "".join(c for c in text if unicodedata.category(c) != "Mn")
    return text


def normalize_whitespace(text: str) -> str:
    """Collapse all whitespace to single spaces, strip edges."""
    return " ".join(text.split())


# Legal suffix canonical forms
_LEGAL_SUFFIXES: Dict[str, str] = {
    "pvt": "private",
    "priv": "private",
    "ltd": "limited",
    "ltda": "limited",
    "inc": "incorporated",
    "incorp": "incorporated",
    "corp": "corporation",
    "llc": "llc",
    "llp": "llp",
    "plc": "plc",
    "co": "company",
    "gmbh": "gmbh",
    "ag": "ag",
    "sa": "sa",
    "sarl": "sarl",
    "srl": "srl",
    "bv": "bv",
    "nv": "nv",
    "pty": "proprietary",
    "ste": "societe",
    "assn": "association",
    "assoc": "association",
    "grp": "group",
    "intl": "international",
    "natl": "national",
    "mfg": "manufacturing",
    "mgmt": "management",
    "svcs": "services",
    "svc": "service",
    "tech": "technology",
    "techs": "technologies",
    "soln": "solution",
    "solns": "solutions",
    "engg": "engineering",
    "engr": "engineering",
    "infra": "infrastructure",
    "govt": "government",
    "edu": "education",
    "hosp": "hospital",
    "pharm": "pharmacy",
    "pharma": "pharmaceuticals",
    "labs": "laboratories",
    "lab": "laboratory",
    "mktg": "marketing",
    "advt": "advertising",
    "advtg": "advertising",
    "ind": "industries",
    "inds": "industries",
    "est": "establishment",
    "estb": "establishment",
    "dept": "department",
    "hq": "headquarters",
    "hdqrs": "headquarters",
    "fdn": "foundation",
    "fndn": "foundation",
}

# Address abbreviation canonical forms
_ADDRESS_ABBREVS: Dict[str, str] = {
    "st": "street",
    "str": "street",
    "ave": "avenue",
    "av": "avenue",
    "blvd": "boulevard",
    "rd": "road",
    "dr": "drive",
    "ln": "lane",
    "ct": "court",
    "pl": "place",
    "sq": "square",
    "hwy": "highway",
    "pkwy": "parkway",
    "cir": "circle",
    "apt": "apartment",
    "ste": "suite",
    "fl": "floor",
    "bldg": "building",
    "ofc": "office",
    "dept": "department",
    "rm": "room",
    "no": "number",
    "nr": "near",
    "opp": "opposite",
    "dist": "district",
    "sec": "sector",
    "blk": "block",
    "extn": "extension",
    "ext": "extension",
    "nagar": "nagar",
    "ngr": "nagar",
    "marg": "marg",
    "mg": "mahatma gandhi",
    "rly": "railway",
    "stn": "station",
    "jn": "junction",
    "jct": "junction",
    "xrd": "crossroad",
    "mkt": "market",
    "indl": "industrial",
    "est": "estate",
    "twp": "township",
    "vlg": "village",
    "vill": "village",
    "po": "post office",
    "ph": "phase",
    "pkg": "parking",
    "rte": "route",
    "rt": "route",
    "n": "north",
    "s": "south",
    "e": "east",
    "w": "west",
    "ne": "northeast",
    "nw": "northwest",
    "se": "southeast",
    "sw": "southwest",
}

# Country normalization
_COUNTRY_MAP: Dict[str, str] = {
    "us": "us",
    "usa": "us",
    "united states": "us",
    "united states of america": "us",
    "u.s.": "us",
    "u.s.a.": "us",
    "in": "in",
    "ind": "in",
    "india": "in",
    "fr": "fr",
    "fra": "fr",
    "france": "fr",
    "uk": "gb",
    "gb": "gb",
    "united kingdom": "gb",
    "de": "de",
    "deu": "de",
    "germany": "de",
    "cn": "cn",
    "chn": "cn",
    "china": "cn",
    "jp": "jp",
    "jpn": "jp",
    "japan": "jp",
    "au": "au",
    "aus": "au",
    "australia": "au",
    "ca": "ca",
    "can": "ca",
    "canada": "ca",
    "br": "br",
    "bra": "br",
    "brazil": "br",
    "sg": "sg",
    "sgp": "sg",
    "singapore": "sg",
}


def normalize_name(raw: str) -> str:
    """
    Normalize a business name.

    Steps:
        1. Unicode normalize
        2. Lowercase
        3. Replace '&' with 'and'
        4. Remove punctuation except hyphens
        5. Expand legal suffixes
        6. Collapse whitespace
    """
    if not raw or pd.isna(raw):
        return ""
    text = normalize_unicode(str(raw))
    text = text.lower()
    # & → and
    text = text.replace("&", " and ")
    # Remove punctuation except hyphen and apostrophe
    text = re.sub(r"[^\w\s\-']", " ", text)
    # Remove standalone dots
    text = re.sub(r"(?<!\w)\.(?!\w)", " ", text)
    # Normalize whitespace
    text = normalize_whitespace(text)
    # Expand legal suffixes
    tokens = text.split()
    expanded = [_LEGAL_SUFFIXES.get(t, t) for t in tokens]
    return " ".join(expanded)


def normalize_address(raw: str) -> str:
    """
    Normalize a business address.

    Steps:
        1. Unicode normalize
        2. Lowercase
        3. Remove punctuation except hyphens and slashes
        4. Expand address abbreviations
        5. Collapse whitespace
    """
    if not raw or pd.isna(raw):
        return ""
    text = normalize_unicode(str(raw))
    text = text.lower()
    text = text.replace("&", " and ")
    # Keep hyphens, slashes, hash (address components)
    text = re.sub(r"[^\w\s\-/#]", " ", text)
    text = normalize_whitespace(text)
    # Expand abbreviations
    tokens = text.split()
    expanded = [_ADDRESS_ABBREVS.get(t, t) for t in tokens]
    return " ".join(expanded)


def normalize_country(raw: str) -> str:
    """
    Normalize country to ISO-like 2-letter code.
    Unknown countries are lowercased and stripped.
    Open-set: never discard unknown countries.
    """
    if not raw or pd.isna(raw):
        return ""
    text = str(raw).strip().lower()
    text = re.sub(r"[^\w\s]", "", text)
    text = normalize_whitespace(text)
    return _COUNTRY_MAP.get(text, text)


def extract_tokens(text: str) -> Tuple[str, ...]:
    """Split normalized text into token tuple."""
    if not text:
        return ()
    return tuple(text.split())


def extract_numeric_tokens(text: str) -> Tuple[str, ...]:
    """Extract sequences of digits from text."""
    if not text:
        return ()
    return tuple(re.findall(r"\d+", text))


def extract_postal(address: str) -> str:
    """
    Extract postal/ZIP/PIN code from address text.
    Handles US (5-digit, 5+4), India (6-digit), France (5-digit).
    """
    if not address:
        return ""
    # US ZIP: 5 digits or 5+4
    m = re.search(r"\b(\d{5})(?:-\d{4})?\b", address)
    if m:
        return m.group(1)
    # India PIN: 6 digits
    m = re.search(r"\b(\d{6})\b", address)
    if m:
        return m.group(1)
    return ""


# ──────────────────────────────────────────────────────────────
# RECORD STORE — columnar representation
# ──────────────────────────────────────────────────────────────

# Standard column names after normalization
NORM_COLS = [
    "entity_id",
    "source",
    "name_raw",
    "name_norm",
    "name_tokens",
    "name_compact",  # no spaces, for exact dedup
    "address_raw",
    "address_norm",
    "address_tokens",
    "postal",
    "numeric_tokens",
    "country_raw",
    "country_norm",
]


def load_source_tsv(path: str, expected_source: Optional[Source] = None) -> pd.DataFrame:
    """
    Load a source TSV file and normalize all fields.

    Returns a DataFrame with NORM_COLS columns.
    """
    df = pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False)

    # Validate columns
    required = {"entity_id", "business_name", "business_address", "country"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in {path}: {missing}")

    # Build normalized DataFrame
    result = pd.DataFrame()
    result["entity_id"] = df["entity_id"].astype(str).str.strip()
    result["source"] = result["entity_id"].apply(
        lambda x: Source.from_id(x).value
    ).astype(np.int8)

    # Validate source if expected
    if expected_source is not None:
        bad = result[result["source"] != expected_source.value]
        if len(bad) > 0:
            raise ValueError(
                f"Found {len(bad)} records with wrong source in {path}. "
                f"Expected S{expected_source.value}, got prefixes: "
                f"{bad['entity_id'].head(5).tolist()}"
            )

    # Raw fields
    result["name_raw"] = df["business_name"].fillna("").astype(str)
    result["address_raw"] = df["business_address"].fillna("").astype(str)
    result["country_raw"] = df["country"].fillna("").astype(str)

    # Normalized fields
    result["name_norm"] = result["name_raw"].apply(normalize_name)
    result["name_tokens"] = result["name_norm"].apply(extract_tokens)
    result["name_compact"] = result["name_norm"].str.replace(
        r"\s+", "", regex=True
    )

    result["address_norm"] = result["address_raw"].apply(normalize_address)
    result["address_tokens"] = result["address_norm"].apply(extract_tokens)

    result["postal"] = result["address_raw"].apply(
        lambda x: extract_postal(str(x).lower())
    )
    result["numeric_tokens"] = result["address_raw"].apply(
        lambda x: extract_numeric_tokens(str(x).lower())
    )

    result["country_norm"] = result["country_raw"].apply(normalize_country)

    return result


def load_ground_truth(path: str) -> Dict[EntityID, Set[EntityID]]:
    """
    Load ground truth TSV.

    Format:
        source1_entity_id \\t matched_entity_ids

    matched_entity_ids is comma-separated, empty for singletons.

    Returns dict: s1_id -> set of matched IDs (empty set for singletons).
    """
    df = pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False)

    # Handle column names
    if "source1_entity_id" not in df.columns:
        # Try common alternatives
        possible_id_cols = [c for c in df.columns if "source1" in c.lower() or "s1" in c.lower()]
        if possible_id_cols:
            df = df.rename(columns={possible_id_cols[0]: "source1_entity_id"})
        else:
            raise ValueError(
                f"Cannot find source1 entity ID column. Columns: {list(df.columns)}"
            )

    if "matched_entity_ids" not in df.columns:
        possible_match_cols = [c for c in df.columns if "match" in c.lower()]
        if possible_match_cols:
            df = df.rename(columns={possible_match_cols[0]: "matched_entity_ids"})
        else:
            raise ValueError(
                f"Cannot find matched entity IDs column. Columns: {list(df.columns)}"
            )

    gt: Dict[EntityID, Set[EntityID]] = {}
    for _, row in df.iterrows():
        s1_id = str(row["source1_entity_id"]).strip()
        raw_matches = str(row["matched_entity_ids"]).strip()
        if raw_matches == "" or raw_matches.lower() == "nan":
            gt[s1_id] = set()
        else:
            matches = {m.strip() for m in raw_matches.split(",") if m.strip()}
            gt[s1_id] = matches

    return gt


class RecordStore:
    """
    Holds normalized records from one or more sources.

    Provides O(1) lookup by entity_id and efficient source filtering.
    """

    def __init__(self, df: pd.DataFrame):
        self._df = df
        # Build ID → row index lookup
        self._id_to_idx: Dict[EntityID, int] = {}
        for idx, eid in enumerate(df["entity_id"].values):
            self._id_to_idx[str(eid)] = idx

    @property
    def df(self) -> pd.DataFrame:
        return self._df

    def __len__(self) -> int:
        return len(self._df)

    def __contains__(self, entity_id: EntityID) -> bool:
        return entity_id in self._id_to_idx

    def get_row(self, entity_id: EntityID) -> pd.Series:
        idx = self._id_to_idx[entity_id]
        return self._df.iloc[idx]

    def get_field(self, entity_id: EntityID, field: str) -> str:
        idx = self._id_to_idx[entity_id]
        return str(self._df.at[self._df.index[idx], field])

    def source_ids(self, source: Source) -> List[EntityID]:
        mask = self._df["source"] == source.value
        return self._df.loc[mask, "entity_id"].tolist()

    def source_df(self, source: Source) -> pd.DataFrame:
        mask = self._df["source"] == source.value
        return self._df[mask].reset_index(drop=True)

    def all_ids(self) -> List[EntityID]:
        return self._df["entity_id"].tolist()

    @classmethod
    def from_sources(cls, *dfs: pd.DataFrame) -> "RecordStore":
        combined = pd.concat(dfs, ignore_index=True)
        # Verify no duplicate entity_ids
        dups = combined["entity_id"].duplicated()
        if dups.any():
            dup_ids = combined.loc[dups, "entity_id"].head(10).tolist()
            raise ValueError(f"Duplicate entity_ids found: {dup_ids}")
        return cls(combined)
