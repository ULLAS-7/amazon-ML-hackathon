"""
features.py — Pairwise feature computation for entity matching.

Each function takes two record dicts (each with keys: entity_id,
business_name, business_address, country) and returns a feature dict
suitable for passing to a classifier.
"""

import re
from typing import Any

from rapidfuzz import fuzz
from rapidfuzz.distance import Levenshtein


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _norm(text: Any) -> str:
    """Lowercase and collapse whitespace; return '' for None/missing."""
    if not text:
        return ""
    return re.sub(r"\s+", " ", str(text).lower().strip())


def _tokens(text: str) -> set[str]:
    """Split on non-alphanumeric chars, drop empty tokens."""
    return set(t for t in re.split(r"[^a-z0-9]+", text) if t)


def _jaccard(a: str, b: str) -> float:
    """Token-level Jaccard similarity between two normalised strings."""
    ta, tb = _tokens(a), _tokens(b)
    if not ta and not tb:
        return 1.0
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / len(ta | tb)


def _levenshtein_ratio(a: str, b: str) -> float:
    """Normalised Levenshtein similarity in [0, 1]."""
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    # rapidfuzz normalised_similarity returns a value in [0, 1]
    return Levenshtein.normalized_similarity(a, b)


def _extract_house_number(address: str) -> str:
    """Return the leading digit sequence from an address string (e.g. '1795')."""
    if not address:
        return ""
    m = re.match(r"(\d+)", address.strip())
    return m.group(1) if m else ""


def _extract_city_region(address: str) -> str:
    """
    Heuristic: take the second-to-last comma-separated segment as city/region.
    Falls back to the last segment if there is only one.
    """
    if not address:
        return ""
    parts = [p.strip() for p in address.split(",") if p.strip()]
    if len(parts) >= 2:
        return parts[-2].lower()
    return parts[-1].lower() if parts else ""


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def compute_pair_features(s1_row: dict, candidate_row: dict) -> dict:
    """Compute matching features for a (source-1 entity, candidate) pair.

    Parameters
    ----------
    s1_row        : dict with keys entity_id, business_name, business_address,
                    country — the Source 1 anchor record.
    candidate_row : dict with the same keys — a candidate record from Source
                    2 or Source 3 (or Source 1 for dedup).

    Returns
    -------
    dict mapping feature name → numeric value (all floats/ints).

    Features
    --------
    name_jaccard          : Token Jaccard similarity of business names.
    name_lev_ratio        : Normalised Levenshtein ratio of business names.
    name_token_sort_ratio : rapidfuzz token_sort_ratio (0-1 scaled) of names.
    addr_jaccard          : Token Jaccard similarity of addresses.
    addr_lev_ratio        : Normalised Levenshtein ratio of addresses.
    house_num_match       : 1 if leading house numbers are identical, else 0.
    city_region_match     : 1 if extracted city/region segments match, else 0.
    country_match         : 1 if country strings match exactly, else 0.
    name_len_diff         : |len(name_a) - len(name_b)| (character count).
    addr_len_diff         : |len(addr_a) - len(addr_b)| (character count).
    """
    name_a = _norm(s1_row.get("business_name", ""))
    name_b = _norm(candidate_row.get("business_name", ""))

    addr_a = _norm(s1_row.get("business_address", ""))
    addr_b = _norm(candidate_row.get("business_address", ""))

    country_a = _norm(s1_row.get("country", ""))
    country_b = _norm(candidate_row.get("country", ""))

    # Name features
    name_jaccard = _jaccard(name_a, name_b)
    name_lev_ratio = _levenshtein_ratio(name_a, name_b)
    # token_sort_ratio returns 0-100; scale to 0-1
    name_token_sort_ratio = fuzz.token_sort_ratio(name_a, name_b) / 100.0

    # Address features
    addr_jaccard = _jaccard(addr_a, addr_b)
    addr_lev_ratio = _levenshtein_ratio(addr_a, addr_b)

    # Structural address features
    house_a = _extract_house_number(addr_a)
    house_b = _extract_house_number(addr_b)
    house_num_match = int(bool(house_a) and house_a == house_b)

    city_a = _extract_city_region(addr_a)
    city_b = _extract_city_region(addr_b)
    city_region_match = int(bool(city_a) and city_a == city_b)

    country_match = int(bool(country_a) and country_a == country_b)

    # Length difference features
    name_len_diff = abs(len(name_a) - len(name_b))
    addr_len_diff = abs(len(addr_a) - len(addr_b))

    return {
        "name_jaccard": name_jaccard,
        "name_lev_ratio": name_lev_ratio,
        "name_token_sort_ratio": name_token_sort_ratio,
        "addr_jaccard": addr_jaccard,
        "addr_lev_ratio": addr_lev_ratio,
        "house_num_match": house_num_match,
        "city_region_match": city_region_match,
        "country_match": country_match,
        "name_len_diff": name_len_diff,
        "addr_len_diff": addr_len_diff,
    }
