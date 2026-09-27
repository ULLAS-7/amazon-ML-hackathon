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

  # Slice test (first N S1 rows only):
  python src/scale_blocking.py ... --s1-limit 50000

Design
------
Pure scale-up wrapper around blocking.py.  No blocking logic reimplemented here.
Calls:
  generate_blocking_keys()  (probe generation)
  get_candidates()          (city_cap=80 — unchanged)

Write-path optimisations (all output logic only — zero change to blocking):
  - Output file opened in BINARY mode with a 64 MB OS buffer.
  - Lines accumulated in a list; flushed every WRITE_BATCH lines with a single
    b"".join() + fh.write() call — eliminates per-row Python overhead.
  - Candidate list joined with b",".join() on pre-encoded bytes — avoids
    repeated str→bytes conversion and csv.writer machinery.
  - sorted() dropped from candidate output (order within a row irrelevant for
    blocking); set→list conversion is O(n), sort is O(n log n).

Full-scale bucket control (applied when building the index):
  - city probes   : handled by city_cap=80 inside get_candidates; pass-through here.
  - nrev probes   : drop NREV_GARBAGE_TOKENS; safety cap NREV_SAFETY_CAP=3000.
  - name probes   : safety cap NAME_SAFETY_CAP=8000.
  - housenum probes: cap HOUSENUM_CAP=200.
"""

from __future__ import annotations

import argparse
import collections
import csv
import sys
import time
from pathlib import Path

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
# Tuning constants
# ---------------------------------------------------------------------------

CHUNK_SIZE        = 50_000
PROGRESS_INTERVAL = 200_000
FREQ_SAMPLE_SIZE  = 10_000
CITY_CAP          = 80

# nrev tokens identified as garbage via full-corpus bucket-size analysis.
NREV_GARBAGE_TOKENS: frozenset[str] = frozenset([
    'लिम', 'लि', 'లిమ', 'ಲಿಮ', 'லிம',
    'cen', 'ser', 'par', 'gro', 'ind', 'hol', 'l.l',
    'ass', 'con', 'pro', 'tra', 'lp', 'tec', 'ven',
    'sol', 'pub', 'cli', 'bro', 'med', 'exp', 'hea',
    'com', 'ent', 'inf', 'car', 'pre',
])

NAME_SAFETY_CAP = 100
NREV_SAFETY_CAP = 1_000
HOUSENUM_CAP    = 100

# Write-path tuning
WRITE_BATCH     = 10_000   # lines accumulated before a single fh.write() call
WRITE_BUF_BYTES = 64 << 20  # 64 MB OS-level write buffer


# ---------------------------------------------------------------------------
# Index builder
# ---------------------------------------------------------------------------

def _build_index_from_file(
    filepath: str,
    freq_stopwords: frozenset[str],
    label: str,
    row_limit: int = 0,
) -> dict[str, list[str]]:
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
    print(f"  [{label}] indexed {row_count:,} rows → {len(index):,} raw probe keys",
          flush=True)
    return dict(index)


def _filter_index(
    index: dict[str, list[str]],
    label: str,
) -> dict[str, list[str]]:
    filtered: dict[str, list[str]] = {}
    dropped_nrev_garbage = dropped_nrev_cap = dropped_name_cap = dropped_hnum_cap = 0
    for k, v in index.items():
        if '::nrev:' in k:
            tok = k.split('::nrev:')[1]
            if tok in NREV_GARBAGE_TOKENS:
                dropped_nrev_garbage += 1
                continue
            if len(v) > NREV_SAFETY_CAP:
                dropped_nrev_cap += 1
                continue
            filtered[k] = v
        elif '::name:' in k:
            if len(v) > NAME_SAFETY_CAP:
                dropped_name_cap += 1
                continue
            filtered[k] = v
        elif '::housenum:' in k:
            if len(v) > HOUSENUM_CAP:
                dropped_hnum_cap += 1
                continue
            filtered[k] = v
        else:
            filtered[k] = v
    print(
        f"  [{label}] filtered: {len(filtered):,} keys kept | "
        f"nrev garbage={dropped_nrev_garbage} | nrev cap={dropped_nrev_cap} | "
        f"name cap={dropped_name_cap} | hnum cap={dropped_hnum_cap}",
        flush=True,
    )
    return filtered


# ---------------------------------------------------------------------------
# Main runner
# ---------------------------------------------------------------------------

def run_blocking(
    s1_path: str,
    s2_path: str,
    s3_path: str,
    out_path: str,
    s1_limit: int = 0,
    s1_id_filter: set[str] | None = None,
) -> tuple[int, float, int]:
    overall_start = time.time()

    print("=" * 65)
    print(f"S1 : {s1_path}" + (f"  [first {s1_limit:,} rows]" if s1_limit else "")
          + (f"  [id-filter: {len(s1_id_filter):,} IDs]" if s1_id_filter else ""))
    print(f"S2 : {s2_path}")
    print(f"S3 : {s3_path}")
    print(f"OUT: {out_path}")
    print(f"city_cap={CITY_CAP}  name_cap={NAME_SAFETY_CAP}  "
          f"nrev_garbage={len(NREV_GARBAGE_TOKENS)}tokens+cap{NREV_SAFETY_CAP}  "
          f"hnum_cap={HOUSENUM_CAP}  write_batch={WRITE_BATCH}")
    print("=" * 65, flush=True)

    # Step 1: freq stopwords
    print(f"\n[1/4] Building freq_stopwords from first {FREQ_SAMPLE_SIZE:,} S1 rows …",
          flush=True)
    t0 = time.time()
    sample_rows = load_sample(s1_path, n=FREQ_SAMPLE_SIZE)
    freq_stopwords = build_frequency_stopwords(sample_rows, threshold=50)
    print(f"      freq_stopwords ({len(freq_stopwords)}): {sorted(freq_stopwords)[:20]}  "
          f"[{time.time()-t0:.1f}s]", flush=True)

    # Step 2: S2 index
    print("\n[2/4] Building inverted index from S2 …", flush=True)
    t0 = time.time()
    index_s2 = _filter_index(_build_index_from_file(s2_path, freq_stopwords, "S2"), "S2")
    print(f"      S2 done [{time.time()-t0:.1f}s]", flush=True)

    # Step 3: S3 index
    print("\n[3/4] Building inverted index from S3 …", flush=True)
    t0 = time.time()
    index_s3 = _filter_index(_build_index_from_file(s3_path, freq_stopwords, "S3"), "S3")
    print(f"      S3 done [{time.time()-t0:.1f}s]", flush=True)

    # Merge
    print("\n      Merging filtered S2+S3 indexes …", end=" ", flush=True)
    t0 = time.time()
    combined: dict[str, list[str]] = collections.defaultdict(list)
    for k, v in index_s2.items():
        combined[k].extend(v)
    for k, v in index_s3.items():
        combined[k].extend(v)
    combined_index = dict(combined)
    del index_s2, index_s3, combined
    print(f"{len(combined_index):,} keys  [{time.time()-t0:.1f}s]", flush=True)

    # Step 4: stream S1 and write output
    print(f"\n[4/4] Streaming S1 …", flush=True)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)

    # Pre-encode the tab and newline bytes used in every output line
    TAB = b"\t"
    NL  = b"\n"

    total_rows    = 0
    total_cands   = 0
    error_rows    = 0
    max_cands     = 0
    t_stream      = time.time()
    last_milestone = 0

    # Open output in BINARY mode with a large OS buffer — avoids text-mode
    # encoding overhead and lets us build lines as byte strings directly.
    with open(s1_path, encoding="utf-8", errors="replace", newline="") as fh_in, \
         open(out_path, "wb", buffering=WRITE_BUF_BYTES) as fh_out:

        reader = csv.DictReader(fh_in, delimiter="\t")
        s1_chunk: list[dict] = []
        line_batch: list[bytes] = []

        def _flush_chunk(chunk: list[dict]) -> tuple[int, int, int, int]:
            """Process chunk; accumulate byte lines; return (rows, cands, errs, max)."""
            rows_out = cands_out = errs = max_c = 0
            for row in chunk:
                eid = (row.get("entity_id") or "").strip()
                if not eid:
                    errs += 1
                    continue
                if s1_id_filter is not None and eid not in s1_id_filter:
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
                    print(f"\n  ERROR on {eid}: {exc}", flush=True)
                    errs += 1
                    continue

                # Filter to S2-/S3- only — no sort (order irrelevant for blocking)
                filtered = [c for c in cands
                            if c.startswith("S2-") or c.startswith("S3-")]

                # Build output line as bytes directly — avoids str→bytes per write
                eid_b   = eid.encode()
                cands_b = b",".join(c.encode() for c in filtered)
                line_batch.append(eid_b + TAB + cands_b + NL)

                n = len(filtered)
                cands_out += n
                rows_out  += 1
                if n > max_c:
                    max_c = n

                # Flush write batch when it reaches WRITE_BATCH lines
                if len(line_batch) >= WRITE_BATCH:
                    fh_out.write(b"".join(line_batch))
                    line_batch.clear()

            return rows_out, cands_out, errs, max_c

        rows_seen = 0
        for row in reader:
            rows_seen += 1
            if s1_limit and rows_seen > s1_limit:
                break
            s1_chunk.append(row)

            if len(s1_chunk) >= CHUNK_SIZE:
                r, c, e, m = _flush_chunk(s1_chunk)
                total_rows  += r
                total_cands += c
                error_rows  += e
                max_cands    = max(max_cands, m)
                s1_chunk = []

                milestone = total_rows // PROGRESS_INTERVAL
                if milestone > last_milestone:
                    last_milestone = milestone
                    elapsed = time.time() - t_stream
                    rate = rows_seen / elapsed if elapsed > 0 else 0
                    avg  = total_cands / total_rows if total_rows else 0
                    print(
                        f"  … {total_rows:>9,} written ({rows_seen:,} seen) | "
                        f"avg cands={avg:.1f} | max={max_cands} | "
                        f"{rate:,.0f} rows/s | elapsed={elapsed:.0f}s",
                        flush=True,
                    )

        # Final partial chunk
        if s1_chunk:
            r, c, e, m = _flush_chunk(s1_chunk)
            total_rows  += r
            total_cands += c
            error_rows  += e
            max_cands    = max(max_cands, m)

        # Flush any remaining buffered lines
        if line_batch:
            fh_out.write(b"".join(line_batch))
            line_batch.clear()

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
    p = argparse.ArgumentParser()
    p.add_argument("--s1",       required=True)
    p.add_argument("--s2",       required=True)
    p.add_argument("--s3",       required=True)
    p.add_argument("--out",      required=True)
    p.add_argument("--s1-limit", type=int, default=0,
                   help="Process only first N S1 rows (0=all)")
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
