"""
vectorized_blocking.py — Fully vectorized pandas blocking for the test set.

Replaces the row-by-row Python loop with:
  1. Vectorized key generation (pandas str ops, no apply/iterrows)
  2. Hash-join merge per probe type
  3. groupby+agg deduplication
  4. Binary-mode buffered output

Probes (same logic as blocking.py, reimplemented with pandas str ops):
  - name_fwd  : country + first 3 chars of first significant name token (forward)
  - name_rev  : country + first 3 chars of last significant name token (reversed)
  - city      : country + last comma-segment of address (lowercased)
  - housenum  : country + first digit-run in address

Safety caps applied via groupby().head(N) BEFORE merge to keep join fan-out bounded.

Usage:
  python src/vectorized_blocking.py \
      --s1 dataset/test/test_source1.tsv \
      --s2 dataset/test/test_source2.tsv \
      --s3 dataset/test/test_source3.tsv \
      --out submissions/test_candidate_pairs_vectorized.tsv
"""

from __future__ import annotations
import argparse, re, sys, time
from pathlib import Path
import pandas as pd
import numpy as np

# ---------------------------------------------------------------------------
# Caps (matching the NAME_CAP=100 regime)
# ---------------------------------------------------------------------------
NAME_CAP    = 100
NREV_CAP    = 1_000
HOUSENUM_CAP = 100
CITY_CAP    = 80

# Legal suffixes to strip before taking name prefix (lower-case)
LEGAL_STOPS = {
    'inc','incorporated','ltd','limited','pvt','private','llc','llp',
    'corp','corporation','co','company','sarl','sas','sa','srl','gmbh',
    'bv','nv','plc','pllc','the','and','of','for','in',
}
# nrev garbage tokens (same list as scale_blocking.py)
NREV_GARBAGE = {
    'लिम','लि','లిమ','ಲಿಮ','லிம',
    'cen','ser','par','gro','ind','hol','l.l',
    'ass','con','pro','tra','lp','tec','ven',
    'sol','pub','cli','bro','med','exp','hea',
    'com','ent','inf','car','pre',
}

_WRITE_BUF = 64 << 20  # 64 MB

# ---------------------------------------------------------------------------
# Vectorized key builders
# ---------------------------------------------------------------------------

def _clean_country(s: pd.Series) -> pd.Series:
    return s.fillna("").str.strip().str.lower()


def _significant_first_token(name_series: pd.Series) -> pd.Series:
    """Return first significant (non-legal-stop) lowercase token of each name."""
    # Normalize: lowercase, replace hyphens with space, collapse whitespace
    s = (name_series.fillna("")
         .str.lower()
         .str.replace(r"[-]", " ", regex=True)
         .str.replace(r"\s+", " ", regex=True)
         .str.strip())
    # Split on whitespace; iterate columns to find first non-stop token
    # Split into up to 10 tokens
    parts = s.str.split(r"\s+", expand=True, n=9)
    # Build result Series: first non-legal-stop token, prefix 3 chars
    result = pd.Series("", index=s.index, dtype=str)
    for col in parts.columns:
        col_vals = parts[col].fillna("").str.strip()
        # Strip leading/trailing punctuation per token
        col_vals = col_vals.str.replace(r"^[^\w]+|[^\w]+$", "", regex=True)
        is_stop = col_vals.isin(LEGAL_STOPS) | (col_vals == "")
        # Fill result where not yet filled and this col is not a stop
        unfilled = result == ""
        can_fill = unfilled & ~is_stop
        result = result.where(~can_fill, col_vals.str[:3])
    return result


def _significant_last_token(name_series: pd.Series) -> pd.Series:
    """Return first 3 chars of last significant non-legal-stop token."""
    s = (name_series.fillna("")
         .str.lower()
         .str.replace(r"[-]", " ", regex=True)
         .str.replace(r"\s+", " ", regex=True)
         .str.strip())
    parts = s.str.split(r"\s+", expand=True, n=9)
    result = pd.Series("", index=s.index, dtype=str)
    # Iterate columns in reverse to find last non-stop token
    for col in reversed(parts.columns):
        col_vals = parts[col].fillna("").str.strip()
        col_vals = col_vals.str.replace(r"^[^\w]+|[^\w]+$", "", regex=True)
        is_stop = col_vals.isin(LEGAL_STOPS) | (col_vals == "") | col_vals.isin(NREV_GARBAGE)
        unfilled = result == ""
        can_fill = unfilled & ~is_stop
        result = result.where(~can_fill, col_vals.str[:3])
    return result


def _city_key(addr_series: pd.Series) -> pd.Series:
    """Last non-empty comma-segment of address, lowercased and stripped."""
    s = addr_series.fillna("")
    # Extract the substring after the last comma (or whole string if no comma)
    has_comma = s.str.contains(",", regex=False)
    after_last = s.str.rsplit(",", n=1).str[-1].str.strip().str.lower()
    whole      = s.str.strip().str.lower()
    return after_last.where(has_comma, whole)


def _housenum_key(addr_series: pd.Series) -> pd.Series:
    """First digit-run in address (house/building number)."""
    s = addr_series.fillna("")
    # First try "No. <digits>" pattern
    no_match = s.str.extract(r'\bNo\.?\s*(\d+)', flags=re.IGNORECASE)[0]
    # Fall back to first bare digit run
    bare_match = s.str.extract(r'(\d+)')[0]
    return no_match.fillna(bare_match).fillna("")


def build_keys_df(df: pd.DataFrame, entity_col: str = "entity_id") -> pd.DataFrame:
    """Build (entity_id, probe_type, key_value) long-format DataFrame."""
    ctry = _clean_country(df["country"])

    name_fwd = _significant_first_token(df["business_name"])
    name_rev = _significant_last_token(df["business_name"])
    city     = _city_key(df["business_address"])
    hnum     = _housenum_key(df["business_address"])

    eid = df[entity_col].fillna("").str.strip()

    # Build one DF per probe type, concatenate
    parts = []

    # name_fwd
    mask = (name_fwd != "") & (name_fwd != name_rev)
    key_fwd = ctry + "::name:" + name_fwd
    parts.append(pd.DataFrame({
        "entity_id": eid[mask].values,
        "key": key_fwd[mask].values,
        "probe": "name"
    }))
    # Also include rows where fwd==rev (just don't duplicate)
    mask_same = (name_fwd != "") & (name_fwd == name_rev)
    parts.append(pd.DataFrame({
        "entity_id": eid[mask_same].values,
        "key": key_fwd[mask_same].values,
        "probe": "name"
    }))

    # name_rev
    mask_rev = (name_rev != "") & (name_rev != name_fwd)
    key_rev = ctry + "::nrev:" + name_rev
    parts.append(pd.DataFrame({
        "entity_id": eid[mask_rev].values,
        "key": key_rev[mask_rev].values,
        "probe": "nrev"
    }))

    # city
    mask_city = city != ""
    key_city = ctry + "::city:" + city
    parts.append(pd.DataFrame({
        "entity_id": eid[mask_city].values,
        "key": key_city[mask_city].values,
        "probe": "city"
    }))

    # housenum
    mask_hnum = hnum != ""
    key_hnum = ctry + "::housenum:" + hnum
    parts.append(pd.DataFrame({
        "entity_id": eid[mask_hnum].values,
        "key": key_hnum[mask_hnum].values,
        "probe": "housenum"
    }))

    return pd.concat(parts, ignore_index=True)


def apply_caps_to_index(idx_df: pd.DataFrame) -> pd.DataFrame:
    """Drop index keys whose bucket exceeds the per-probe-type cap."""
    bucket_sizes = idx_df.groupby("key").size().rename("bsize")
    idx_with_size = idx_df.join(bucket_sizes, on="key")

    probe = idx_with_size["probe"]
    size  = idx_with_size["bsize"]

    keep = (
        ((probe == "name")     & (size <= NAME_CAP))    |
        ((probe == "nrev")     & (size <= NREV_CAP))    |
        ((probe == "city")     & (size <= CITY_CAP))    |
        ((probe == "housenum") & (size <= HOUSENUM_CAP))
    )
    filtered = idx_df[keep.values].copy()
    dropped = len(idx_df) - len(filtered)
    print(f"  Index: {len(filtered):,} pairs kept, {dropped:,} dropped by caps", flush=True)
    return filtered


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run(s1_path: str, s2_path: str, s3_path: str, out_path: str) -> None:
    t0_total = time.time()
    COLS = ["entity_id", "business_name", "business_address", "country"]

    print("=" * 65, flush=True)
    print(f"Vectorized blocking  NAME_CAP={NAME_CAP} NREV_CAP={NREV_CAP} "
          f"HNUM_CAP={HOUSENUM_CAP} CITY_CAP={CITY_CAP}")
    print(f"S1={s1_path}\nS2={s2_path}\nS3={s3_path}\nOUT={out_path}")
    print("=" * 65, flush=True)

    # ── Load source files ──────────────────────────────────────────────
    print("\n[1/5] Loading source files …", flush=True)
    t = time.time()
    s1 = pd.read_csv(s1_path, sep="\t", usecols=COLS,
                     dtype=str, keep_default_na=False, low_memory=False)
    s2 = pd.read_csv(s2_path, sep="\t", usecols=COLS,
                     dtype=str, keep_default_na=False, low_memory=False)
    s3 = pd.read_csv(s3_path, sep="\t", usecols=COLS,
                     dtype=str, keep_default_na=False, low_memory=False)
    print(f"  S1={len(s1):,}  S2={len(s2):,}  S3={len(s3):,}  [{time.time()-t:.1f}s]",
          flush=True)

    # ── Build key DataFrames ──────────────────────────────────────────
    print("\n[2/5] Building probe keys (vectorized) …", flush=True)
    t = time.time()
    s1_keys = build_keys_df(s1, "entity_id")
    print(f"  S1 keys: {len(s1_keys):,}  [{time.time()-t:.1f}s]", flush=True)

    t = time.time()
    s2_keys = build_keys_df(s2, "entity_id")
    s3_keys = build_keys_df(s3, "entity_id")
    print(f"  S2 keys: {len(s2_keys):,}  S3 keys: {len(s3_keys):,}  [{time.time()-t:.1f}s]",
          flush=True)

    # ── Apply caps to S2/S3 indexes ───────────────────────────────────
    print("\n[3/5] Applying per-probe caps to S2/S3 indexes …", flush=True)
    t = time.time()
    # Rename for merge clarity
    s2_idx = apply_caps_to_index(s2_keys.rename(columns={"entity_id": "candidate_id"}))
    s3_idx = apply_caps_to_index(s3_keys.rename(columns={"entity_id": "candidate_id"}))
    idx_all = pd.concat([s2_idx, s3_idx], ignore_index=True)
    del s2_keys, s3_keys, s2_idx, s3_idx
    print(f"  Combined index: {len(idx_all):,} rows  [{time.time()-t:.1f}s]", flush=True)

    # Safety check: if combined index is huge, cap total size
    MAX_INDEX_ROWS = 50_000_000  # 50M rows hard limit
    if len(idx_all) > MAX_INDEX_ROWS:
        print(f"  WARNING: index too large ({len(idx_all):,}), "
              f"truncating to {MAX_INDEX_ROWS:,}", flush=True)
        idx_all = idx_all.iloc[:MAX_INDEX_ROWS]

    # ── Merge S1 keys with combined index ─────────────────────────────
    print("\n[4/5] Merging S1 keys with S2/S3 index …", flush=True)
    t = time.time()

    # Check merge fan-out estimate before doing it
    s1_key_count = s1_keys["key"].nunique()
    idx_key_count = idx_all["key"].nunique()
    common_keys = len(set(s1_keys["key"].unique()) & set(idx_all["key"].unique()))
    print(f"  S1 unique keys: {s1_key_count:,}  Index unique keys: {idx_key_count:,}  "
          f"Common: {common_keys:,}", flush=True)

    merged = pd.merge(
        s1_keys[["entity_id", "key"]],
        idx_all[["candidate_id", "key"]],
        on="key",
        how="inner"
    )
    del idx_all, s1_keys
    print(f"  Merge result: {len(merged):,} pairs  [{time.time()-t:.1f}s]", flush=True)

    if len(merged) == 0:
        print("  WARNING: no pairs after merge — writing empty output", flush=True)

    # ── Group by S1 entity, aggregate candidates ──────────────────────
    print("\n[5/5] Grouping and writing output …", flush=True)
    t = time.time()

    # Remove self-matches (shouldn't happen across sources, but guard)
    merged = merged[merged["entity_id"] != merged["candidate_id"]]

    # Deduplicate (entity_id, candidate_id) pairs
    merged = merged.drop_duplicates(subset=["entity_id", "candidate_id"])

    # Group: aggregate candidate IDs as comma-joined string
    grouped = (
        merged.groupby("entity_id", sort=False)["candidate_id"]
        .agg(",".join)
        .reset_index()
        .rename(columns={"candidate_id": "candidate_entity_ids"})
    )
    del merged
    print(f"  {len(grouped):,} S1 entities with ≥1 candidate  [{time.time()-t:.1f}s]",
          flush=True)

    # Build full output: every S1 entity must appear (empty string if no candidates)
    all_s1 = pd.DataFrame({"entity_id": s1["entity_id"].str.strip()})
    del s1
    result = all_s1.merge(grouped, on="entity_id", how="left")
    result["candidate_entity_ids"] = result["candidate_entity_ids"].fillna("")
    del grouped, all_s1

    # Write binary, buffered
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    TAB, NL = b"\t", b"\n"
    written = 0
    non_empty = 0
    with open(out_path, "wb", buffering=_WRITE_BUF) as fh:
        batch: list[bytes] = []
        for eid, cands in zip(result["entity_id"], result["candidate_entity_ids"]):
            line = eid.encode() + TAB + (cands.encode() if cands else b"") + NL
            batch.append(line)
            written += 1
            if cands:
                non_empty += 1
            if len(batch) >= 50_000:
                fh.write(b"".join(batch))
                batch.clear()
        if batch:
            fh.write(b"".join(batch))

    out_mb = Path(out_path).stat().st_size / 1_000_000
    total_t = time.time() - t0_total

    print("\n" + "=" * 65)
    print("DONE")
    print(f"  Output        : {out_path}")
    print(f"  File size     : {out_mb:.1f} MB")
    print(f"  Total S1 rows : {written:,}")
    print(f"  Non-empty     : {non_empty:,}")
    print(f"  Total time    : {total_t:.1f}s ({total_t/60:.1f} min)")
    print("=" * 65, flush=True)


def _parse() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--s1", required=True)
    p.add_argument("--s2", required=True)
    p.add_argument("--s3", required=True)
    p.add_argument("--out", required=True)
    return p.parse_args()


if __name__ == "__main__":
    args = _parse()
    run(args.s1, args.s2, args.s3, args.out)
