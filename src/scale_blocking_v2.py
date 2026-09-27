"""
scale_blocking_v2.py — Vectorised pandas-merge blocking for the full dataset.

Strategy
--------
1. Stream S1 in chunks; for each chunk build a DataFrame of
   (s1_entity_id, probe_key) rows.
2. Build a filtered (candidate_entity_id, probe_key) DataFrame from S2+S3 once
   (the "index DF"), applying the same garbage-token drops and per-probe-type
   bucket caps as scale_blocking.py.
3. For each S1 chunk: merge(s1_keys_df, index_df, on='probe_key') to get all
   (s1_id, cand_id) pairs in one vectorised join, then groupby+agg to
   deduplicate candidates per S1 entity.
4. Write results in batches using binary-mode buffered output.

All blocking probe logic is delegated to generate_blocking_keys() from
blocking.py — no logic reimplemented here.

Usage
-----
  # Full train:
  python src/scale_blocking_v2.py \
      --s1 dataset/train/train_source1.tsv \
      --s2 dataset/train/train_source2.tsv \
      --s3 dataset/train/train_source3.tsv \
      --out submissions/candidate_pairs.tsv

  # Speed test (50k S1 rows):
  python src/scale_blocking_v2.py ... --s1-limit 50000
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

import pandas as pd
import numpy as np

_SRC_DIR = Path(__file__).resolve().parent
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from blocking import (
    build_frequency_stopwords,
    generate_blocking_keys,
    load_sample,
)

# ---------------------------------------------------------------------------
# Constants (same tuning as scale_blocking.py NAME_CAP=8000 config)
# ---------------------------------------------------------------------------

S1_CHUNK_SIZE   = 50_000   # S1 rows per merge batch
FREQ_SAMPLE_SIZE = 10_000
CITY_CAP         = 80
NAME_SAFETY_CAP  = 8_000
NREV_SAFETY_CAP  = 3_000
HOUSENUM_CAP     = 200
WRITE_BUF_BYTES  = 64 << 20  # 64 MB output buffer

NREV_GARBAGE_TOKENS: frozenset[str] = frozenset([
    'लिम', 'लि', 'లిమ', 'ಲಿಮ', 'லிம',
    'cen', 'ser', 'par', 'gro', 'ind', 'hol', 'l.l',
    'ass', 'con', 'pro', 'tra', 'lp', 'tec', 'ven',
    'sol', 'pub', 'cli', 'bro', 'med', 'exp', 'hea',
    'com', 'ent', 'inf', 'car', 'pre',
])


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _rows_to_keys_df(rows: list[dict], freq_sw: frozenset[str]) -> pd.DataFrame:
    """Convert a list of entity row dicts to a (entity_id, probe_key) DataFrame."""
    records = []
    for row in rows:
        eid = (row.get("entity_id") or "").strip()
        if not eid:
            continue
        for k in generate_blocking_keys(
            row.get("business_name", "") or "",
            row.get("business_address", "") or "",
            row.get("country", "") or "",
            freq_sw,
        ):
            records.append((eid, k))
    if not records:
        return pd.DataFrame(columns=["entity_id", "probe_key"])
    return pd.DataFrame(records, columns=["entity_id", "probe_key"])


def _build_index_df(
    filepath: str,
    freq_sw: frozenset[str],
    label: str,
    row_limit: int = 0,
) -> pd.DataFrame:
    """Stream *filepath* → filtered (candidate_id, probe_key) DataFrame."""
    records = []
    count = 0
    with open(filepath, encoding="utf-8", errors="replace", newline="") as fh:
        reader = csv.DictReader(fh, delimiter="\t")
        for row in reader:
            if row_limit and count >= row_limit:
                break
            eid = (row.get("entity_id") or "").strip()
            if not eid:
                continue
            for k in generate_blocking_keys(
                row.get("business_name", "") or "",
                row.get("business_address", "") or "",
                row.get("country", "") or "",
                freq_sw,
            ):
                records.append((eid, k))
            count += 1

    print(f"  [{label}] {count:,} rows → {len(records):,} raw key pairs", flush=True)
    if not records:
        return pd.DataFrame(columns=["candidate_id", "probe_key"])

    df = pd.DataFrame(records, columns=["candidate_id", "probe_key"])

    # Apply per-probe-type bucket caps
    bucket_sizes = df.groupby("probe_key").size()

    def _keep_key(key: str, size: int) -> bool:
        if "::nrev:" in key:
            tok = key.split("::nrev:")[1]
            if tok in NREV_GARBAGE_TOKENS:
                return False
            return size <= NREV_SAFETY_CAP
        elif "::name:" in key:
            return size <= NAME_SAFETY_CAP
        elif "::housenum:" in key:
            return size <= HOUSENUM_CAP
        # city probes: keep all; city_cap applied at query time via get_candidates
        # but in v2 we don't call get_candidates — apply city_cap here directly
        elif "::city:" in key:
            return size <= CITY_CAP
        return True

    kept_keys = bucket_sizes[
        [_keep_key(k, s) for k, s in bucket_sizes.items()]
    ].index
    df_filtered = df[df["probe_key"].isin(kept_keys)]

    dropped = len(df) - len(df_filtered)
    print(f"  [{label}] filtered: {len(df_filtered):,} pairs kept "
          f"({dropped:,} dropped by caps)", flush=True)
    return df_filtered.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_blocking_v2(
    s1_path: str,
    s2_path: str,
    s3_path: str,
    out_path: str,
    s1_limit: int = 0,
) -> None:
    t_total = time.time()

    print("=" * 65)
    print(f"S1 : {s1_path}" + (f"  [first {s1_limit:,} rows]" if s1_limit else ""))
    print(f"S2 : {s2_path}")
    print(f"S3 : {s3_path}")
    print(f"OUT: {out_path}")
    print(f"city_cap={CITY_CAP}  name_cap={NAME_SAFETY_CAP}  "
          f"nrev_garbage={len(NREV_GARBAGE_TOKENS)}+cap{NREV_SAFETY_CAP}  "
          f"hnum_cap={HOUSENUM_CAP}  s1_chunk={S1_CHUNK_SIZE:,}")
    print("=" * 65, flush=True)

    # Step 1: freq stopwords
    print(f"\n[1/4] freq_stopwords from first {FREQ_SAMPLE_SIZE:,} S1 rows …", flush=True)
    t0 = time.time()
    freq_sw = build_frequency_stopwords(
        load_sample(s1_path, n=FREQ_SAMPLE_SIZE), threshold=50
    )
    print(f"      {sorted(freq_sw)}  [{time.time()-t0:.1f}s]", flush=True)

    # Step 2: build combined S2+S3 index DataFrame
    print("\n[2/4] Building S2 index DataFrame …", flush=True)
    t0 = time.time()
    idx_s2 = _build_index_df(s2_path, freq_sw, "S2")
    print(f"      S2 done [{time.time()-t0:.1f}s]", flush=True)

    print("\n[3/4] Building S3 index DataFrame …", flush=True)
    t0 = time.time()
    idx_s3 = _build_index_df(s3_path, freq_sw, "S3")
    print(f"      S3 done [{time.time()-t0:.1f}s]", flush=True)

    print("\n      Concatenating S2+S3 index …", end=" ", flush=True)
    t0 = time.time()
    idx_df = pd.concat([idx_s2, idx_s3], ignore_index=True)
    del idx_s2, idx_s3
    # Convert probe_key to category for faster merge
    idx_df["probe_key"] = idx_df["probe_key"].astype("category")
    print(f"{len(idx_df):,} rows  [{time.time()-t0:.1f}s]", flush=True)

    # Step 4: stream S1 in chunks, merge, write
    print(f"\n[4/4] Streaming S1 in chunks of {S1_CHUNK_SIZE:,} …", flush=True)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)

    total_rows  = 0
    total_cands = 0
    TAB = b"\t"
    NL  = b"\n"
    t_stream = time.time()
    PROGRESS_INTERVAL = 200_000
    last_milestone = 0

    with open(s1_path, encoding="utf-8", errors="replace", newline="") as fh_in, \
         open(out_path, "wb", buffering=WRITE_BUF_BYTES) as fh_out:

        reader = csv.DictReader(fh_in, delimiter="\t")
        s1_chunk: list[dict] = []
        rows_seen = 0

        def _process_and_write(chunk: list[dict]) -> tuple[int, int]:
            """Merge chunk against index_df, write results, return (rows, cands)."""
            # Build S1 keys DataFrame for this chunk
            s1_keys = _rows_to_keys_df(chunk, freq_sw)
            if s1_keys.empty:
                return 0, 0

            # Merge on probe_key — vectorised join
            s1_keys["probe_key"] = s1_keys["probe_key"].astype(
                idx_df["probe_key"].dtype  # match category dtype for fast join
            )
            merged = pd.merge(s1_keys, idx_df, on="probe_key", how="inner")

            # Keep only S2-/S3- candidates (entity_ids from S2/S3 always start S2-/S3-)
            mask = (
                merged["candidate_id"].str.startswith("S2-") |
                merged["candidate_id"].str.startswith("S3-")
            )
            merged = merged[mask]

            # Remove self-matches (s1 entity_id == candidate_id — can't happen
            # across sources but guard anyway)
            merged = merged[merged["entity_id"] != merged["candidate_id"]]

            # Deduplicate and group: for each S1 entity get unique candidate set
            # Use groupby + agg for vectorised dedup
            if merged.empty:
                # All S1 IDs in chunk get empty candidate lists
                lines = [
                    (row.get("entity_id","") or "").strip().encode() + TAB + NL
                    for row in chunk
                    if (row.get("entity_id","") or "").strip()
                ]
                fh_out.write(b"".join(lines))
                return len(lines), 0

            grouped = (
                merged.groupby("entity_id", sort=False)["candidate_id"]
                .agg(lambda x: ",".join(x.unique()))
                .reset_index()
            )
            cand_counts = merged.groupby("entity_id")["candidate_id"].nunique()

            # Build a lookup for quick output
            cand_map = dict(zip(grouped["entity_id"], grouped["candidate_id"]))

            rows_w = 0
            cands_w = 0
            line_batch: list[bytes] = []

            for row in chunk:
                eid = (row.get("entity_id") or "").strip()
                if not eid:
                    continue
                cand_str = cand_map.get(eid, "")
                line_batch.append(eid.encode() + TAB + cand_str.encode() + NL)
                rows_w += 1
                cands_w += cand_counts.get(eid, 0)

            fh_out.write(b"".join(line_batch))
            return rows_w, int(cands_w)

        for row in reader:
            rows_seen += 1
            if s1_limit and rows_seen > s1_limit:
                break
            s1_chunk.append(row)

            if len(s1_chunk) >= S1_CHUNK_SIZE:
                r, c = _process_and_write(s1_chunk)
                total_rows  += r
                total_cands += c
                s1_chunk = []

                milestone = total_rows // PROGRESS_INTERVAL
                if milestone > last_milestone:
                    last_milestone = milestone
                    elapsed = time.time() - t_stream
                    rate = rows_seen / elapsed if elapsed > 0 else 0
                    avg  = total_cands / total_rows if total_rows else 0
                    print(
                        f"  … {total_rows:>9,} written ({rows_seen:,} seen) | "
                        f"avg cands={avg:.1f} | "
                        f"{rate:,.0f} rows/s | elapsed={elapsed:.0f}s",
                        flush=True,
                    )

        if s1_chunk:
            r, c = _process_and_write(s1_chunk)
            total_rows  += r
            total_cands += c

    total_elapsed = time.time() - t_total
    stream_elapsed = time.time() - t_stream  # approx (excludes index build)
    avg_cands = total_cands / total_rows if total_rows else 0
    out_size  = Path(out_path).stat().st_size if Path(out_path).exists() else 0
    stream_rate = total_rows / stream_elapsed if stream_elapsed > 0 else 0

    print("\n" + "=" * 65)
    print("DONE")
    print(f"  Output          : {out_path}")
    print(f"  File size       : {out_size/1_000_000:.1f} MB")
    print(f"  S1 rows written : {total_rows:,}")
    print(f"  Avg cands/S1    : {avg_cands:.1f}")
    print(f"  Total time      : {total_elapsed:.1f}s ({total_elapsed/60:.1f} min)")
    print(f"  S1 stream rate  : {stream_rate:,.0f} rows/s")
    print(f"  Projected full  : {2_206_821/stream_rate/60:.0f} min "
          f"(+~5 min index) for 2.2M rows" if stream_rate > 0 else "")
    print("=" * 65, flush=True)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--s1",       required=True)
    p.add_argument("--s2",       required=True)
    p.add_argument("--s3",       required=True)
    p.add_argument("--out",      required=True)
    p.add_argument("--s1-limit", type=int, default=0)
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    run_blocking_v2(
        s1_path  = args.s1,
        s2_path  = args.s2,
        s3_path  = args.s3,
        out_path = args.out,
        s1_limit = args.s1_limit,
    )
