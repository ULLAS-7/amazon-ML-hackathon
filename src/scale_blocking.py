"""
scale_blocking.py — Full-dataset blocking runner (chunked, streaming, no pandas).

Usage
-----
  # Train
  python src/scale_blocking.py \
      --s1 dataset/train/train_source1.tsv \
      --s2 dataset/train/train_source2.tsv \
      --s3 dataset/train/train_source3.tsv \
      --out submissions/candidate_pairs.tsv

  # Test
  python src/scale_blocking.py \
      --s1 dataset/test/test_source1.tsv \
      --s2 dataset/test/test_source2.tsv \
      --s3 dataset/test/test_source3.tsv \
      --out submissions/test_candidate_pairs.tsv

  # Slice test (first 5000 S1 rows only, for validation):
  python src/scale_blocking.py \
      --s1 dataset/train/train_source1.tsv \
      --s2 dataset/train/train_source2.tsv \
      --s3 dataset/train/train_source3.tsv \
      --out submissions/slice_test.tsv \
      --s1-limit 5000

Design
------
This script is a pure scale-up wrapper around the already-validated blocking.py
functions.  It does NOT reimplement any blocking logic — it calls:

  generate_blocking_keys()   from blocking.py  (probe generation)
  get_candidates()           from blocking.py  (city_cap=80 filtering)

The only thing added here is:

  1. Streaming S2/S3 into memory-efficient inverted indexes (entity_id lists only).
  2. Bucket-size cap (MAX_BUCKET_SIZE) applied to the index BEFORE passing it to
     get_candidates.  This cap is the full-scale analogue of the sample cap: at 5k
     rows the largest bucket was ~300 so the sample naturally produced ~77 avg
     candidates.  At 5M rows, buckets like "india::nrev:लिम" grow to 190k — these
     are degenerate keys (legal-suffix reversed tokens, single-digit house numbers)
     that the sample never exposed.  Capping them at MAX_BUCKET_SIZE replicates the
     sample's effective behavior: any bucket so large it is non-discriminating is
     treated as noise and dropped entirely from the lookup.
  3. Chunked S1 streaming (CHUNK_SIZE rows at a time) to avoid full-file RAM.
  4. Progress reporting every PROGRESS_INTERVAL S1 rows.

Output format (TSV, no header):
  source1_entity_id<TAB>candidate_entity_ids
  where candidate_entity_ids is a comma-separated list (S2-/S3- only,
  no duplicates, sorted for determinism) or empty string if none.
"""

from __future__ import annotations

import argparse
import collections
import csv
import sys
import time
from pathlib import Path

# ---------------------------------------------------------------------------
# Path setup — src/ directory (where blocking.py lives) must be on sys.path
# ---------------------------------------------------------------------------
_SRC_DIR = Path(__file__).resolve().parent
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from blocking import (
    build_frequency_stopwords,
    generate_blocking_keys,
    get_candidates,
    load_sample,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

CHUNK_SIZE        = 50_000   # S1 rows per processing batch
PROGRESS_INTERVAL = 200_000  # print progress line every N S1 rows
FREQ_SAMPLE_SIZE  = 10_000   # S1 rows used to build freq_stopwords
CITY_CAP          = 80       # passed directly to get_candidates (unchanged)

# Max entries in any single probe bucket before it is discarded as
# non-discriminating at full dataset scale.  Rationale:
#   - Validated sample: max bucket ~300, avg cands ~77 → ratio ~1:4
#   - Full dataset is ~1000× larger, so we scale the cutoff proportionally:
#     300 × (5_000_000 / 5_000) would be too generous; instead we cap at a
#     level that drops clearly degenerate keys (nrev:लिम @190k, housenum:1
#     @81k) while keeping genuinely specific buckets (nrev:smi, name:aco…).
#   - 500 is a conservative cutoff: keeps all specific buckets, kills only
#     the handful of near-universal keys.
MAX_BUCKET_SIZE = 500


# ---------------------------------------------------------------------------
# Index builder — streams one source file, no full-row storage
# ---------------------------------------------------------------------------

def _build_index_from_file(
    filepath: str,
    freq_stopwords: frozenset[str],
    label: str,
    row_limit: int = 0,
) -> dict[str, list[str]]:
    """Stream *filepath* and build probe_key → [entity_id, …].

    Only entity_ids stored (not full rows), so memory ~ keys × avg_bucket.
    """
    index: dict[str, list[str]] = collections.defaultdict(list)
    row_count = 0

    with open(filepath, encoding="utf-8", errors="replace", newline="") as fh:
        reader = csv.DictReader(fh, delimiter="\t")
        for row in reader:
            if row_limit and row_count >= row_limit:
                break
            eid = row.get("entity_id", "").strip()
            if not eid:
                continue
            keys = generate_blocking_keys(
                row.get("business_name", "") or "",
                row.get("business_address", "") or "",
                row.get("country", "") or "",
                freq_stopwords,
            )
            for k in keys:
                index[k].append(eid)
            row_count += 1

    print(f"  [{label}] indexed {row_count:,} rows → {len(index):,} probe keys",
          flush=True)
    return dict(index)


def _cap_index(
    index: dict[str, list[str]],
    max_size: int,
) -> tuple[dict[str, list[str]], int]:
    """Return a copy of *index* with all buckets larger than *max_size* removed.

    Returns (capped_index, n_dropped_keys).
    """
    capped = {}
    dropped = 0
    for k, v in index.items():
        if len(v) <= max_size:
            capped[k] = v
        else:
            dropped += 1
    return capped, dropped


# ---------------------------------------------------------------------------
# Main runner
# ---------------------------------------------------------------------------

def run_blocking(
    s1_path: str,
    s2_path: str,
    s3_path: str,
    out_path: str,
    s1_limit: int = 0,
) -> tuple[int, float, int]:
    overall_start = time.time()

    print("=" * 65)
    print(f"S1 : {s1_path}" + (f"  [first {s1_limit:,} rows]" if s1_limit else ""))
    print(f"S2 : {s2_path}")
    print(f"S3 : {s3_path}")
    print(f"OUT: {out_path}")
    print(f"city_cap={CITY_CAP}  max_bucket_size={MAX_BUCKET_SIZE}")
    print("=" * 65, flush=True)

    # ------------------------------------------------------------------
    # Step 1: freq stopwords from a sample of S1
    # ------------------------------------------------------------------
    print(f"\n[1/4] Building freq_stopwords from first {FREQ_SAMPLE_SIZE:,} S1 rows …",
          flush=True)
    t0 = time.time()
    sample_rows = load_sample(s1_path, n=FREQ_SAMPLE_SIZE)
    freq_stopwords = build_frequency_stopwords(sample_rows, threshold=50)
    print(f"      freq_stopwords ({len(freq_stopwords)}): {sorted(freq_stopwords)[:20]}",
          flush=True)
    print(f"      done in {time.time()-t0:.1f}s", flush=True)

    # ------------------------------------------------------------------
    # Step 2: build + cap S2 index
    # ------------------------------------------------------------------
    print("\n[2/4] Building inverted index from S2 …", flush=True)
    t0 = time.time()
    raw_s2 = _build_index_from_file(s2_path, freq_stopwords, "S2")
    index_s2, dropped_s2 = _cap_index(raw_s2, MAX_BUCKET_SIZE)
    del raw_s2
    print(f"      S2: {len(index_s2):,} keys kept, {dropped_s2} oversized keys dropped "
          f"(>{MAX_BUCKET_SIZE})  [{time.time()-t0:.1f}s]", flush=True)

    # ------------------------------------------------------------------
    # Step 3: build + cap S3 index
    # ------------------------------------------------------------------
    print("\n[3/4] Building inverted index from S3 …", flush=True)
    t0 = time.time()
    raw_s3 = _build_index_from_file(s3_path, freq_stopwords, "S3")
    index_s3, dropped_s3 = _cap_index(raw_s3, MAX_BUCKET_SIZE)
    del raw_s3
    print(f"      S3: {len(index_s3):,} keys kept, {dropped_s3} oversized keys dropped "
          f"(>{MAX_BUCKET_SIZE})  [{time.time()-t0:.1f}s]", flush=True)

    # Merge capped S2 + S3 into one combined index
    print("\n      Merging capped S2+S3 indexes …", end=" ", flush=True)
    t0 = time.time()
    combined_index: dict[str, list[str]] = collections.defaultdict(list)
    for k, v in index_s2.items():
        combined_index[k].extend(v)
    for k, v in index_s3.items():
        combined_index[k].extend(v)
    # After merging, re-apply cap so cross-source combined buckets stay bounded
    combined_raw = dict(combined_index)
    combined_index, dropped_merged = _cap_index(combined_raw, MAX_BUCKET_SIZE)
    del combined_raw, index_s2, index_s3
    print(f"{len(combined_index):,} keys  ({dropped_merged} additional merged drops)  "
          f"[{time.time()-t0:.1f}s]", flush=True)

    # ------------------------------------------------------------------
    # Step 4: stream S1, query candidates, write output
    # ------------------------------------------------------------------
    print(f"\n[4/4] Streaming S1 in chunks of {CHUNK_SIZE:,} rows …", flush=True)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)

    total_rows  = 0
    total_cands = 0
    error_rows  = 0
    max_cands   = 0
    t_stream    = time.time()
    last_milestone = 0

    with open(s1_path, encoding="utf-8", errors="replace", newline="") as fh_in, \
         open(out_path, "w", encoding="utf-8", newline="") as fh_out:

        writer = csv.writer(fh_out, delimiter="\t", lineterminator="\n")
        reader = csv.DictReader(fh_in, delimiter="\t")

        chunk: list[dict] = []

        def _flush(chunk: list[dict]) -> tuple[int, int, int, int]:
            rows_out = cands_out = errs = max_c = 0
            for row in chunk:
                eid = (row.get("entity_id") or "").strip()
                if not eid:
                    errs += 1
                    continue
                try:
                    cands = get_candidates(
                        eid,
                        row.get("business_name", "") or "",
                        row.get("business_address", "") or "",
                        row.get("country", "") or "",
                        combined_index,
                        freq_stopwords,
                        city_cap=CITY_CAP,
                    )
                except Exception as exc:
                    print(f"\n  ERROR on entity_id={eid}: {exc}", flush=True)
                    errs += 1
                    continue

                filtered = sorted(
                    c for c in cands
                    if c.startswith("S2-") or c.startswith("S3-")
                )
                writer.writerow([eid, ",".join(filtered)])
                n = len(filtered)
                cands_out += n
                rows_out  += 1
                if n > max_c:
                    max_c = n
            return rows_out, cands_out, errs, max_c

        for row in reader:
            if s1_limit and total_rows + len(chunk) >= s1_limit:
                remaining = s1_limit - total_rows - len(chunk)
                if remaining > 0:
                    chunk.append(row)
                if len(chunk) >= min(CHUNK_SIZE, s1_limit - total_rows):
                    r, c, e, m = _flush(chunk)
                    total_rows  += r
                    total_cands += c
                    error_rows  += e
                    max_cands    = max(max_cands, m)
                    chunk = []
                if total_rows >= s1_limit:
                    break
                continue

            chunk.append(row)

            if len(chunk) >= CHUNK_SIZE:
                r, c, e, m = _flush(chunk)
                total_rows  += r
                total_cands += c
                error_rows  += e
                max_cands    = max(max_cands, m)
                chunk = []

                milestone = total_rows // PROGRESS_INTERVAL
                if milestone > last_milestone:
                    last_milestone = milestone
                    elapsed = time.time() - t_stream
                    rate = total_rows / elapsed if elapsed > 0 else 0
                    avg  = total_cands / total_rows if total_rows else 0
                    print(
                        f"  … {total_rows:>9,} S1 rows | "
                        f"avg cands={avg:.1f} | max={max_cands} | "
                        f"{rate:,.0f} rows/s | elapsed={elapsed:.0f}s",
                        flush=True,
                    )

        # flush remainder
        if chunk:
            r, c, e, m = _flush(chunk)
            total_rows  += r
            total_cands += c
            error_rows  += e
            max_cands    = max(max_cands, m)

    total_elapsed = time.time() - overall_start
    avg_cands = total_cands / total_rows if total_rows else 0
    out_size  = Path(out_path).stat().st_size if Path(out_path).exists() else 0

    print("\n" + "=" * 65)
    print("DONE")
    print(f"  Output file     : {out_path}")
    print(f"  File size       : {out_size/1_000_000:.1f} MB")
    print(f"  S1 rows written : {total_rows:,}")
    print(f"  Total candidates: {total_cands:,}")
    print(f"  Avg cands/S1    : {avg_cands:.2f}")
    print(f"  Max cands/S1    : {max_cands}")
    print(f"  Skipped/errors  : {error_rows}")
    print(f"  Total time      : {total_elapsed:.1f}s  ({total_elapsed/60:.1f} min)")
    print("=" * 65, flush=True)

    return total_rows, avg_cands, error_rows


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Full-dataset chunked blocking runner (no pandas, streaming S1)."
    )
    p.add_argument("--s1",       required=True, help="Path to source1 TSV")
    p.add_argument("--s2",       required=True, help="Path to source2 TSV")
    p.add_argument("--s3",       required=True, help="Path to source3 TSV")
    p.add_argument("--out",      required=True, help="Output candidate_pairs.tsv path")
    p.add_argument("--s1-limit", type=int, default=0,
                   help="Process only the first N rows of S1 (0=all; for slice tests)")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    run_blocking(
        s1_path  = args.s1,
        s2_path  = args.s2,
        s3_path  = args.s3,
        out_path = args.out,
        s1_limit = args.s1_limit,
    )
