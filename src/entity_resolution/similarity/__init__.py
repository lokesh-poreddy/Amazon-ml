"""
similarity/functions.py — All string/token/numeric similarity functions.

Stateless, composable functions for computing pairwise similarities.
These are used by the feature builder to construct pair feature vectors.

Organized by family:
    - Edit distance (Levenshtein, normalized)
    - Jaro / Jaro-Winkler
    - Token overlap (Jaccard, overlap coefficient, Dice)
    - Character n-gram similarity
    - TF-IDF cosine (pre-fitted vectorizer)
    - Numeric token overlap
    - Exact / containment
"""

from __future__ import annotations

import math
from collections import Counter
from typing import Optional, Sequence, Tuple


# ──────────────────────────────────────────────────────────────
# EDIT DISTANCE
# ──────────────────────────────────────────────────────────────

def levenshtein_distance(s1: str, s2: str) -> int:
    """Compute Levenshtein edit distance between two strings."""
    if not s1:
        return len(s2)
    if not s2:
        return len(s1)
    if s1 == s2:
        return 0

    try:
        from rapidfuzz.distance import Levenshtein
        return Levenshtein.distance(s1, s2)
    except ImportError:
        pass

    # Pure Python fallback (Wagner-Fischer)
    m, n = len(s1), len(s2)
    if m < n:
        s1, s2, m, n = s2, s1, n, m
    prev = list(range(n + 1))
    for i in range(1, m + 1):
        curr = [i] + [0] * n
        for j in range(1, n + 1):
            cost = 0 if s1[i - 1] == s2[j - 1] else 1
            curr[j] = min(
                curr[j - 1] + 1,
                prev[j] + 1,
                prev[j - 1] + cost,
            )
        prev = curr
    return prev[n]


def normalized_levenshtein(s1: str, s2: str) -> float:
    """Levenshtein similarity normalized to [0, 1]."""
    if not s1 and not s2:
        return 1.0
    max_len = max(len(s1), len(s2))
    if max_len == 0:
        return 1.0
    return 1.0 - levenshtein_distance(s1, s2) / max_len


def edit_similarity(s1: str, s2: str) -> float:
    """Alias for normalized_levenshtein."""
    return normalized_levenshtein(s1, s2)


# ──────────────────────────────────────────────────────────────
# JARO / JARO-WINKLER
# ──────────────────────────────────────────────────────────────

def jaro_similarity(s1: str, s2: str) -> float:
    """Compute Jaro similarity between two strings."""
    if not s1 and not s2:
        return 1.0
    if not s1 or not s2:
        return 0.0
    if s1 == s2:
        return 1.0

    try:
        from rapidfuzz.distance import Jaro
        return Jaro.similarity(s1, s2)
    except ImportError:
        pass

    # Pure Python Jaro
    len1, len2 = len(s1), len(s2)
    match_distance = max(len1, len2) // 2 - 1
    if match_distance < 0:
        match_distance = 0

    s1_matches = [False] * len1
    s2_matches = [False] * len2
    matches = 0
    transpositions = 0

    for i in range(len1):
        start = max(0, i - match_distance)
        end = min(i + match_distance + 1, len2)
        for j in range(start, end):
            if s2_matches[j] or s1[i] != s2[j]:
                continue
            s1_matches[i] = True
            s2_matches[j] = True
            matches += 1
            break

    if matches == 0:
        return 0.0

    k = 0
    for i in range(len1):
        if not s1_matches[i]:
            continue
        while not s2_matches[k]:
            k += 1
        if s1[i] != s2[k]:
            transpositions += 1
        k += 1

    return (
        matches / len1 + matches / len2 +
        (matches - transpositions / 2) / matches
    ) / 3.0


def jaro_winkler_similarity(s1: str, s2: str, p: float = 0.1) -> float:
    """Compute Jaro-Winkler similarity."""
    if s1 == s2:
        return 1.0

    try:
        from rapidfuzz.distance import JaroWinkler
        return JaroWinkler.similarity(s1, s2, prefix_weight=p)
    except ImportError:
        pass

    jaro = jaro_similarity(s1, s2)
    # Common prefix length (max 4)
    prefix_len = 0
    for i in range(min(len(s1), len(s2), 4)):
        if s1[i] == s2[i]:
            prefix_len += 1
        else:
            break

    return jaro + prefix_len * p * (1.0 - jaro)


# ──────────────────────────────────────────────────────────────
# TOKEN OVERLAP
# ──────────────────────────────────────────────────────────────

def token_jaccard(tokens1: Sequence[str], tokens2: Sequence[str]) -> float:
    """Jaccard similarity on token sets."""
    if not tokens1 and not tokens2:
        return 1.0
    s1 = set(tokens1)
    s2 = set(tokens2)
    if not s1 and not s2:
        return 1.0
    intersection = s1 & s2
    union = s1 | s2
    if not union:
        return 0.0
    return len(intersection) / len(union)


def token_overlap_coeff(tokens1: Sequence[str],
                        tokens2: Sequence[str]) -> float:
    """Overlap coefficient: |intersection| / min(|A|, |B|)."""
    if not tokens1 and not tokens2:
        return 1.0
    s1 = set(tokens1)
    s2 = set(tokens2)
    if not s1 or not s2:
        return 0.0
    return len(s1 & s2) / min(len(s1), len(s2))


def token_dice(tokens1: Sequence[str], tokens2: Sequence[str]) -> float:
    """Dice coefficient on token sets."""
    if not tokens1 and not tokens2:
        return 1.0
    s1 = set(tokens1)
    s2 = set(tokens2)
    denom = len(s1) + len(s2)
    if denom == 0:
        return 0.0
    return 2 * len(s1 & s2) / denom


def token_containment(tokens1: Sequence[str],
                      tokens2: Sequence[str]) -> float:
    """Fraction of tokens1 that appear in tokens2."""
    if not tokens1:
        return 1.0 if not tokens2 else 0.0
    s2 = set(tokens2)
    return sum(1 for t in tokens1 if t in s2) / len(tokens1)


def weighted_token_jaccard(
    tokens1: Sequence[str],
    tokens2: Sequence[str],
    idf: Optional[dict] = None,
) -> float:
    """
    IDF-weighted Jaccard similarity.

    If idf dict is provided, each token contributes its IDF weight
    rather than unit weight.
    """
    if not tokens1 and not tokens2:
        return 1.0
    s1 = set(tokens1)
    s2 = set(tokens2)
    if not s1 and not s2:
        return 1.0
    if not s1 or not s2:
        return 0.0

    if idf is None:
        return token_jaccard(tokens1, tokens2)

    intersection = s1 & s2
    union = s1 | s2

    w_inter = sum(idf.get(t, 1.0) for t in intersection)
    w_union = sum(idf.get(t, 1.0) for t in union)

    if w_union == 0:
        return 0.0
    return w_inter / w_union


# ──────────────────────────────────────────────────────────────
# CHARACTER N-GRAM SIMILARITY
# ──────────────────────────────────────────────────────────────

def char_ngram_similarity(
    s1: str, s2: str, n: int = 3
) -> float:
    """Jaccard similarity on character n-gram sets."""
    if not s1 and not s2:
        return 1.0
    if not s1 or not s2:
        return 0.0

    ngrams1 = set(s1[i:i+n] for i in range(len(s1) - n + 1))
    ngrams2 = set(s2[i:i+n] for i in range(len(s2) - n + 1))

    if not ngrams1 and not ngrams2:
        return 1.0
    if not ngrams1 or not ngrams2:
        return 0.0

    return len(ngrams1 & ngrams2) / len(ngrams1 | ngrams2)


# ──────────────────────────────────────────────────────────────
# NUMERIC OVERLAP
# ──────────────────────────────────────────────────────────────

def numeric_token_overlap(
    nums1: Sequence[str], nums2: Sequence[str]
) -> float:
    """Jaccard on numeric token sets."""
    if not nums1 and not nums2:
        return 1.0
    s1 = set(nums1)
    s2 = set(nums2)
    if not s1 and not s2:
        return 1.0
    if not s1 or not s2:
        return 0.0
    return len(s1 & s2) / len(s1 | s2)


def numeric_containment(nums1: Sequence[str],
                        nums2: Sequence[str]) -> float:
    """Fraction of numeric tokens in nums1 found in nums2."""
    if not nums1:
        return 1.0 if not nums2 else 0.0
    s2 = set(nums2)
    return sum(1 for n in nums1 if n in s2) / len(nums1)


# ──────────────────────────────────────────────────────────────
# EXACT / CONTAINMENT
# ──────────────────────────────────────────────────────────────

def exact_match(s1: str, s2: str) -> float:
    """1.0 if strings are identical, 0.0 otherwise."""
    return 1.0 if s1 == s2 else 0.0


def prefix_match(s1: str, s2: str, min_len: int = 3) -> float:
    """1.0 if one string is a prefix of the other (min length)."""
    if not s1 or not s2:
        return 0.0
    shorter = min(len(s1), len(s2))
    if shorter < min_len:
        return 0.0
    if s1.startswith(s2) or s2.startswith(s1):
        return 1.0
    return 0.0


def containment_match(s1: str, s2: str) -> float:
    """1.0 if one string contains the other."""
    if not s1 or not s2:
        return 0.0
    if s1 in s2 or s2 in s1:
        return 1.0
    return 0.0


def length_ratio(s1: str, s2: str) -> float:
    """Ratio of shorter to longer string length."""
    if not s1 and not s2:
        return 1.0
    l1, l2 = len(s1), len(s2)
    if l1 == 0 or l2 == 0:
        return 0.0
    return min(l1, l2) / max(l1, l2)


def token_count_diff(tokens1: Sequence[str],
                     tokens2: Sequence[str]) -> int:
    """Absolute difference in token counts."""
    return abs(len(tokens1) - len(tokens2))


def length_diff(s1: str, s2: str) -> int:
    """Absolute character length difference."""
    return abs(len(s1) - len(s2))
