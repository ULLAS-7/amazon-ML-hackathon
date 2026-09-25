"""
blocking.py — Multi-probe blocking for Business Entity Resolution.

Public API
----------
get_aligned_sample(gt_path, s1_path, s2_path, s3_path, n=2000)
    -> tuple[list[dict], list[dict], list[dict], list[dict]]
    Build a properly-aligned sample for recall measurement:
      - Streams GT to collect the first N source1 entity IDs that have at
        least one match (skips pure singletons).
      - Collects the union of all matched_entity_ids for those N entities.
      - Streams each source file and pulls only the rows whose entity_id
        appears in the relevant collected ID sets.
    Returns (s1_rows, s2_rows, s3_rows, gt_rows) — all four aligned.

build_frequency_stopwords(rows, threshold=50) -> frozenset[str]
    Scan a list of record dicts and return tokens that appear as the first
    significant word in more than `threshold` records.  These are too common
    to be useful discriminators (e.g. "new", "mumbai", "delhi").

generate_blocking_keys(business_name, business_address, country,
                       freq_stopwords=frozenset()) -> list[str]
    Return the list of probe keys for a single record (multi-probe blocking).
    A record belongs to the union of all buckets keyed by any of these probes.

    Probe 1  : country + first-3-chars of normalized name (forward)
                 (after stripping legal suffixes AND freq_stopwords)
    Probe 2  : country + first-3-chars of reversed-token normalized name
                 (handles "Acme Robotics" vs "Robotics Acme")
    Probe 3  : country + lowercased last comma-segment of address (city-ish)
    Probe 4  : country + first contiguous digit sequence in address
                 (house/building number; skipped when no digits found)

build_blocking_index(rows, freq_stopwords=frozenset()) -> dict[str, list[str]]
    Build a {key -> [entity_id, ...]} index from a list of record dicts.
    Each record is inserted under ALL of its probe keys.

get_candidates(entity_id, business_name, business_address, country,
               index, freq_stopwords=frozenset(), city_cap=80) -> set[str]
    Return the set of candidate entity_ids that share at least one probe key
    with the given record (excluding the record itself).
    City-probe buckets larger than city_cap are filtered: only records that
    also share a name/nrev/housenum probe are included from oversized buckets.

load_sample(filepath, n=5000) -> list[dict]
    Read the first n data rows from a TSV file without loading it fully.
"""

from __future__ import annotations

import collections
import csv
import re
import unicodedata
from typing import Optional

# ---------------------------------------------------------------------------
# Static stopwords
# ---------------------------------------------------------------------------

# Legal / structural words that almost never distinguish a business.
# Lower-case; comparison happens after lowercasing.
LEGAL_STOPWORDS: frozenset[str] = frozenset(
    [
        "inc", "incorporated",
        "ltd", "limited",
        "pvt", "private",
        "llc", "llp",
        "corp", "corporation",
        "co", "company",
        "sarl", "sas", "sa", "srl", "gmbh", "bv", "nv",
        "plc", "pllc",
        "the", "and", "&",
        "of", "for", "in",
    ]
)

# ---------------------------------------------------------------------------
# Regex helpers
# ---------------------------------------------------------------------------

_LEADING_NOISE_RE  = re.compile(
    r"^[\s\-\<\>\!\@\#\$\%\^\*\(\)\[\]\{\}\|\\\/\+\=\~\`\'\"\.,:;]+"
)
_TRAILING_NOISE_RE = re.compile(
    r"[\s\-\<\>\!\@\#\$\%\^\*\(\)\[\]\{\}\|\\\/\+\=\~\`\'\"\.,:;]+$"
)
_WHITESPACE_RE = re.compile(r"\s+")
_TOKEN_SPLIT_RE = re.compile(r"\s+")

# ---------------------------------------------------------------------------
# Low-level text helpers
# ---------------------------------------------------------------------------

def _strip_noise(text: str) -> str:
    """Strip leading/trailing punctuation/noise and collapse internal spaces."""
    text = _LEADING_NOISE_RE.sub("", text)
    text = _TRAILING_NOISE_RE.sub("", text)
    text = _WHITESPACE_RE.sub(" ", text)
    return text.strip()


def _is_latin(token: str) -> bool:
    """True if at least one character in *token* belongs to the Latin script."""
    for ch in token:
        if unicodedata.category(ch).startswith("L"):
            if "LATIN" in unicodedata.name(ch, ""):
                return True
    return False


def _clean_token(raw: str) -> str:
    """Strip per-token leading/trailing noise; return empty string if nothing survives.

    Interior hyphens are NOT removed here — that is handled at the full-name
    level in _normalize_name before the string is split into tokens, so that
    "QS-Automobiles" splits into ["QS", "Automobiles"] matching "QS Automobiles".
    """
    tok = _LEADING_NOISE_RE.sub("", raw)
    tok = _TRAILING_NOISE_RE.sub("", tok)
    return tok


def _normalize_name(name: str) -> str:
    """Top-level noise strip + hyphen-to-space + space collapse (does NOT lowercase).

    Hyphens inside the name are replaced with a space so that hyphenated
    variants like "QS-Automobiles" split into the same tokens as the
    unhyphenated "QS Automobiles".  Leading/trailing noise is stripped first
    so that a name starting with "--" loses the leading hyphens before this
    interior replacement runs.
    """
    name = _strip_noise(name)             # drop leading/trailing noise
    name = name.replace("-", " ")         # interior hyphens → space
    name = _WHITESPACE_RE.sub(" ", name)  # collapse any new multi-spaces
    return name.strip()


def _significant_tokens(
    name: str,
    extra_stopwords: frozenset[str] = frozenset(),
) -> list[str]:
    """Return all non-stopword tokens from a business name, lower-cased if Latin.

    Tokens in LEGAL_STOPWORDS or extra_stopwords (both lower-case sets) are
    omitted.  Non-Latin tokens are kept as-is (no lowercase).
    Falls back to a list containing the very first non-empty token when every
    token is a stopword, so callers always get something to work with.
    """
    name = _normalize_name(name)
    if not name:
        return []

    combined_stops = LEGAL_STOPWORDS | extra_stopwords
    result: list[str] = []
    first_tok: Optional[str] = None

    for raw in _TOKEN_SPLIT_RE.split(name):
        tok = _clean_token(raw)
        if not tok:
            continue
        tok_out = tok.lower() if _is_latin(tok) else tok
        if first_tok is None:
            first_tok = tok_out
        if tok.lower() not in combined_stops:
            result.append(tok_out)

    if not result and first_tok is not None:
        result = [first_tok]
    return result


def _name_prefix3(
    name: str,
    extra_stopwords: frozenset[str] = frozenset(),
    reversed_order: bool = False,
) -> str:
    """Return the first 3 characters of the first significant token of *name*.

    If reversed_order=True, tokens are considered in reverse order (last
    significant token first) — this gives a second probe that survives
    word-order transpositions like "Acme Robotics" vs "Robotics Acme".

    Non-Latin scripts: the raw Unicode characters are used; len() counts
    Unicode code points, which is fine for prefix blocking.
    Returns "" when name is empty or all tokens are stripped away.
    """
    tokens = _significant_tokens(name, extra_stopwords)
    if not tokens:
        return ""
    tok = tokens[-1] if reversed_order else tokens[0]
    return tok[:3]  # first 3 Unicode code-points


def _last_address_segment(address: str) -> str:
    """Lowest-level last-comma-segment extractor (city-ish proxy).

    Returns empty string for missing/blank address.
    Lowercases Latin-script output; passes non-Latin through unchanged.
    """
    if not address or not address.strip():
        return ""
    segs = address.split(",")
    # Walk backward to find a non-empty segment
    for seg in reversed(segs):
        seg = _strip_noise(seg.strip())
        if seg:
            return seg.lower() if _is_latin(seg) else seg
    return ""


# Pre-compiled patterns for house/building number extraction.
#
# Strategy: look for a run of digits that is NOT immediately preceded by
# a bare "No" / "No." / "No " label, which merely introduces the number
# rather than being part of it.  We do this by first trying to match the
# canonical "No[. ]+<digits>" pattern and capturing the digits; if that
# doesn't match we fall back to the first bare digit run in the address.
#
# This fixes the "No.90" → "9" truncation: the old _FIRST_DIGITS_RE hit the
# "9" in "No." before reaching "90".  Now we explicitly look past the label.
_NO_PREFIX_RE  = re.compile(r"\bNo\.?\s*(\d+)", re.IGNORECASE)
_FIRST_DIGITS_RE = re.compile(r"\d+")


def _first_address_digits(address: str) -> str:
    """Extract the primary house/building number from *address*.

    Two-pass strategy:
    1. Look for a "No. <digits>" or "No <digits>" pattern (case-insensitive)
       and return the captured digit group — this skips the "No" label that
       was causing premature truncation ("No.90" → "90" not "9").
    2. Fall back to the first bare contiguous digit run anywhere in the
       address string (original behaviour for addresses without a "No" label).

    Returns empty string when the address is blank or digit-free.

    Examples
    --------
    "No.90, Pantheon Road, Chennai"          -> "90"   (was "9" before fix)
    "Door No 183, 41St Cross, Bengaluru"     -> "183"
    "KH NO. -570/13, NEW DELHI"              -> "570"
    "1795 Westchester Drive, High Point, NC" -> "1795" (no "No" label)
    "6905 Darmstadt Road, Evansville, IN"    -> "6905"
    "6905-C Darmstadt Rd, Evansville, IN"    -> "6905"
    "Connaught Place, New Delhi, Delhi"      -> ""     (no digits)
    ""                                       -> ""
    """
    if not address:
        return ""
    # Pass 1: "No[.] <digits>" pattern — captures the number after the label
    m = _NO_PREFIX_RE.search(address)
    if m:
        return m.group(1)
    # Pass 2: first bare digit run
    m = _FIRST_DIGITS_RE.search(address)
    return m.group() if m else ""


# ---------------------------------------------------------------------------
# Frequency-based stopword detection
# ---------------------------------------------------------------------------

def build_frequency_stopwords(
    rows: list[dict],
    threshold: int = 50,
) -> frozenset[str]:
    """Return tokens that appear as the first significant name-word in > threshold records.

    These tokens are too common to be useful discriminators in a 5000-row
    sample (e.g. "new", "mumbai", "delhi" from business names like
    "New Delhi Traders").  They are added to the dynamic stopword set so
    _name_prefix3 can skip them.

    Only the FIRST significant token (using LEGAL_STOPWORDS only, not yet the
    dynamic set) is counted — we want to find tokens that are common *in the
    first position*, which is where they cause over-broad buckets.

    Parameters
    ----------
    rows      : list of record dicts (from load_sample).
    threshold : int — tokens seen in > threshold records are stopwords.

    Returns
    -------
    frozenset[str] — lower-case dynamic stopwords.
    """
    counter: collections.Counter[str] = collections.Counter()
    for row in rows:
        name = row.get("business_name", "") or ""
        toks = _significant_tokens(name, frozenset())  # use only static stops
        if toks:
            counter[toks[0]] += 1
    return frozenset(tok for tok, cnt in counter.items() if cnt > threshold)


# ---------------------------------------------------------------------------
# Multi-probe key generation
# ---------------------------------------------------------------------------

def generate_blocking_keys(
    business_name: str,
    business_address: str,
    country: str,
    freq_stopwords: frozenset[str] = frozenset(),
) -> list[str]:
    """Return the list of probe keys for one record (multi-probe blocking).

    Three probes are generated:

    Probe 1 — country + name-prefix-forward
        ``<country>::name:<first-3-chars-of-first-significant-token>``
        Primary discriminator.  Legal + freq stopwords are skipped.

    Probe 2 — country + name-prefix-reversed
        ``<country>::nrev:<first-3-chars-of-last-significant-token>``
        Handles word-order transpositions ("Acme Robotics" ↔ "Robotics Acme").

    Probe 3 — country + address city segment
        ``<country>::city:<last-comma-segment-of-address>``
        Coarser geographic probe; useful when name is highly varied but
        address is consistent.

    Keys are formatted with a prefix tag (``name:``, ``nrev:``, ``city:``) so
    probe types don't accidentally collide in the shared index.

    Parameters
    ----------
    business_name    : str — may be empty, non-Latin, noisy.
    business_address : str — may be empty or have no fixed field order.
    country          : str — open set of string labels (US / India / France …).
    freq_stopwords   : frozenset[str] — dynamic high-freq tokens to skip
                       (from build_frequency_stopwords).

    Returns
    -------
    list[str] — 1–3 probe keys (fewer when components are empty).
    """
    business_name    = business_name    or ""
    business_address = business_address or ""
    country          = country          or ""

    ctry = _strip_noise(country).lower()

    keys: list[str] = []

    # Probe 1: forward name prefix
    p1 = _name_prefix3(business_name, freq_stopwords, reversed_order=False)
    if p1:
        keys.append(f"{ctry}::name:{p1}")

    # Probe 2: reversed token order name prefix (only add if different from p1)
    p2 = _name_prefix3(business_name, freq_stopwords, reversed_order=True)
    k2 = f"{ctry}::nrev:{p2}" if p2 else None
    if k2 and k2 != keys[0] if keys else k2:
        keys.append(k2)

    # Probe 3: address city segment
    city = _last_address_segment(business_address)
    if city:
        keys.append(f"{ctry}::city:{city}")

    # Probe 4: first contiguous digit sequence in address (house/building number).
    # This catches DBA/trade-name cases where the name differs entirely but the
    # street number is shared across sources.  Only added when digits exist —
    # never creates a fake empty-string bucket.
    housenum = _first_address_digits(business_address)
    if housenum:
        keys.append(f"{ctry}::housenum:{housenum}")

    # Guarantee at least one key even for empty records
    if not keys:
        keys.append(f"{ctry}::empty")

    return keys


# ---------------------------------------------------------------------------
# Index build + candidate lookup
# ---------------------------------------------------------------------------

def build_blocking_index(
    rows: list[dict],
    freq_stopwords: frozenset[str] = frozenset(),
) -> dict[str, list[str]]:
    """Build a {probe_key -> [entity_id, ...]} inverted index from *rows*.

    Each record is inserted under every probe key it generates.

    Parameters
    ----------
    rows           : list of record dicts — must have 'entity_id',
                     'business_name', 'business_address', 'country'.
    freq_stopwords : frozenset[str] — passed to generate_blocking_keys.

    Returns
    -------
    dict[str, list[str]]
    """
    index: dict[str, list[str]] = collections.defaultdict(list)
    for row in rows:
        eid  = row["entity_id"]
        keys = generate_blocking_keys(
            row.get("business_name", ""),
            row.get("business_address", ""),
            row.get("country", ""),
            freq_stopwords,
        )
        for k in keys:
            index[k].append(eid)
    return dict(index)


def get_candidates(
    entity_id: str,
    business_name: str,
    business_address: str,
    country: str,
    index: dict[str, list[str]],
    freq_stopwords: frozenset[str] = frozenset(),
    city_cap: int = 80,
) -> set[str]:
    """Return the union of all records that share any probe key with this record.

    City-probe cap
    --------------
    The city probe (``::city:``) generates the broadest buckets and is the
    weakest signal per candidate it adds.  To reduce bloat without hurting
    recall, any city bucket with more than ``city_cap`` records is treated
    as a *secondary filter* instead of a *primary source*:

    - If the city bucket size is <= city_cap  → include all members as
      normal (same behaviour as before).
    - If the city bucket size > city_cap → only include a city-bucket member
      if it *also* shares at least one non-city probe key (name, nrev, or
      housenum) with the query record.  Records that match on city alone
      inside an oversized bucket are dropped.

    This turns the city probe into a geographic tie-breaker for large cities
    rather than a standalone recall source for very common city names.

    Parameters
    ----------
    entity_id        : str — the query record's own ID (excluded from output).
    business_name    : str
    business_address : str
    country          : str
    index            : dict returned by build_blocking_index.
    freq_stopwords   : frozenset[str]
    city_cap         : int — max city-bucket size before the filter kicks in
                       (default 80; set to a large number to disable).

    Returns
    -------
    set[str] — candidate entity_ids.
    """
    keys = generate_blocking_keys(business_name, business_address, country,
                                   freq_stopwords)

    # Separate city keys from the other (strong) keys
    city_keys  = [k for k in keys if "::city:" in k]
    other_keys = [k for k in keys if "::city:" not in k]

    # Build the candidate set from strong probes first
    candidates: set[str] = set()
    for k in other_keys:
        candidates.update(index.get(k, []))

    # Apply city probe with optional cap
    for ck in city_keys:
        bucket = index.get(ck, [])
        if len(bucket) <= city_cap:
            # Small bucket — add everyone unconditionally
            candidates.update(bucket)
        else:
            # Large bucket — only add members already in the strong-probe set
            # (i.e. they share at least one name/nrev/housenum key with us)
            for eid in bucket:
                if eid in candidates:
                    # Already captured by a strong probe — keep (no-op, already in)
                    pass
                # else: city-only match in an oversized bucket — skip

    candidates.discard(entity_id)
    return candidates


# ---------------------------------------------------------------------------
# Sample loader (no pandas, no full-file load)
# ---------------------------------------------------------------------------

def load_sample(filepath: str, n: int = 5000) -> list[dict]:
    """Read the first *n* data rows from a TSV file without loading it fully.

    Parameters
    ----------
    filepath : str — path to a tab-separated file with a header row.
    n        : int — maximum data rows to return (default 5000).

    Returns
    -------
    list[dict] — each dict maps column name → value string.
    """
    rows: list[dict] = []
    with open(filepath, encoding="utf-8", errors="replace", newline="") as fh:
        reader = csv.DictReader(fh, delimiter="\t")
        for i, row in enumerate(reader):
            if i >= n:
                break
            rows.append(dict(row))
    return rows


# ---------------------------------------------------------------------------
# Aligned sample builder
# ---------------------------------------------------------------------------

def get_aligned_sample(
    gt_path: str,
    s1_path: str,
    s2_path: str,
    s3_path: str,
    n: int = 2000,
) -> tuple[list[dict], list[dict], list[dict], list[dict]]:
    """Build a recall-testable aligned sample from the training files.

    Pure positional slicing produces almost no overlap between GT entity IDs
    and source-file rows (the files are in different orderings).  This function
    fixes that by collecting IDs from GT first, then pulling only the matching
    rows from each source file.

    Algorithm
    ---------
    1. Stream ``gt_path``: collect the first ``n`` source1_entity_ids that
       have at least one match (skip pure singletons — no true pairs to test
       recall against).  Also collect the union of all their matched_entity_ids.
    2. Stream ``s1_path``: keep rows whose entity_id is in the N selected IDs.
    3. Stream ``s2_path`` and ``s3_path``: keep rows whose entity_id is in
       the matched_entity_ids set.
    4. Also return the GT rows for those N entities (same subset as step 1).

    Parameters
    ----------
    gt_path : str — path to train_ground_truth.tsv
    s1_path : str — path to train_source1.tsv
    s2_path : str — path to train_source2.tsv
    s3_path : str — path to train_source3.tsv
    n       : int — number of S1 entities to sample (default 2000)

    Returns
    -------
    (s1_rows, s2_rows, s3_rows, gt_rows) — four lists of record dicts.
    s1_rows  : records from source1 for the N sampled entities
    s2_rows  : records from source2 that are true matches for sampled entities
    s3_rows  : records from source3 that are true matches for sampled entities
    gt_rows  : ground-truth rows for the N sampled entities (two-column dicts)
    """
    # ---- Pass 1: stream GT to collect target IDs ----
    selected_s1_ids: list[str] = []
    selected_s1_set: set[str] = set()
    matched_ids: set[str] = set()
    gt_subset: list[dict] = []

    with open(gt_path, encoding="utf-8", errors="replace") as fh:
        fh.readline()  # skip header
        for line in fh:
            if len(selected_s1_ids) >= n:
                break
            line = line.rstrip("\n")
            if not line:
                continue
            parts = line.split("\t")
            s1_eid = parts[0]
            matched_str = parts[1] if len(parts) > 1 else ""
            if not matched_str.strip():
                continue  # skip singletons
            ids = set(matched_str.split(","))
            selected_s1_ids.append(s1_eid)
            selected_s1_set.add(s1_eid)
            matched_ids.update(ids)
            gt_subset.append({"source1_entity_id": s1_eid,
                               "matched_entity_ids": matched_str})

    # ---- Pass 2: stream source1 for matching S1 rows ----
    # Use raw readline+split instead of csv.DictReader — DictReader buffers
    # aggressively on very large files and can exhaust memory in subprocesses.
    s1_rows: list[dict] = []
    with open(s1_path, encoding="utf-8", errors="replace") as fh:
        hdr = fh.readline().rstrip("\n").split("\t")
        for line in fh:
            parts = line.rstrip("\n").split("\t")
            if not parts[0]:
                continue
            if parts[0] in selected_s1_set:
                s1_rows.append(dict(zip(hdr, parts)))
                if len(s1_rows) == len(selected_s1_set):
                    break  # found all; no need to scan further

    # ---- Pass 3+4: stream source2/source3 for matching S2/S3 rows ----
    s2_rows: list[dict] = []
    with open(s2_path, encoding="utf-8", errors="replace") as fh:
        hdr = fh.readline().rstrip("\n").split("\t")
        for line in fh:
            parts = line.rstrip("\n").split("\t")
            if not parts[0]:
                continue
            if parts[0] in matched_ids:
                s2_rows.append(dict(zip(hdr, parts)))

    s3_rows: list[dict] = []
    with open(s3_path, encoding="utf-8", errors="replace") as fh:
        hdr = fh.readline().rstrip("\n").split("\t")
        for line in fh:
            parts = line.rstrip("\n").split("\t")
            if not parts[0]:
                continue
            if parts[0] in matched_ids:
                s3_rows.append(dict(zip(hdr, parts)))

    return s1_rows, s2_rows, s3_rows, gt_subset


# ---------------------------------------------------------------------------
# Quick self-test  (python src/blocking.py)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    _cases = [
        ("Orelee's Barbershop",
         "1795 Westchester Drive, High Point, NC", "US"),
        ("B+ Retail Inc",
         "1712 Montebello Avenue, Phoenix, AZ", "US"),
        ("Pvt. EFS Print Ventures Ltd.",
         "Door No 183, 41St Cross, Bengaluru, Karnataka", "India"),
        ("राम मार्केटिंग प्राइवेट लिमिटेड",
         "KH NO. -570/13, NEW DELHI, WEST DELHI, Delhi", "India"),
        ("-- Holloway Peak Inc Seafood",
         "105 ELM ST, MORGANTON, NC", "US"),
        ("<< Team Ecole",
         "175 Boulevard Roosevelt, Bordeaux, Nouvelle-Aquitaine", "France"),
        ("LLC Moncada Learning Center",
         "5780 Fawn Ct, Fort Worth, Texas", "US"),
        ("New Delhi Traders",
         "Connaught Place, New Delhi, Delhi", "India"),
        # Probe 4 test: DBA case — different names, same street number
        ("Colonial Family Practice Group",
         "6905 Darmstadt Road, Evansville, IN", "US"),
        ("Dovakor",
         "6905-C Darmstadt Rd, Evansville, IN", "US"),
        ("", "", ""),
    ]

    freq_sw = frozenset(["new"])   # simulate freq-stopword for "new"

    print(f"{'NAME':<42}  PROBES")
    print("-" * 110)
    for name, addr, ctry in _cases:
        probes = generate_blocking_keys(name, addr, ctry, freq_sw)
        print(f"{name:<42}  {probes}")
