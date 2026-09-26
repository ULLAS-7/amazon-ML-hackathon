"""
build_fallback_submission.py — Generate a format-valid fallback submission.

Strategy
--------
* ALL entities in test_source1.tsv get an EMPTY match list and an EMPTY
  candidate list.  This is safe: it always scores 0 but is never rejected
  by the validator.

* Streams test_source1.tsv in a single pass — no full load into RAM.

Outputs (relative to repo root)
--------
  output/matching_results.tsv   — one row per test S1 entity, empty matches
  output/candidate_pairs.tsv    — one row per test S1 entity, empty candidates

Usage
-----
    python src/build_fallback_submission.py
"""

from __future__ import annotations

import os
import sys
import time

REPO_ROOT  = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TEST_S1    = os.path.join(REPO_ROOT, "dataset", "test", "test_source1.tsv")
OUTPUT_DIR = os.path.join(REPO_ROOT, "output")
MATCHING   = os.path.join(OUTPUT_DIR, "matching_results.tsv")
CANDIDATES = os.path.join(OUTPUT_DIR, "candidate_pairs.tsv")


def build_fallback() -> int:
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print(f"Streaming {TEST_S1} …", flush=True)
    t0 = time.time()

    n = 0
    with open(TEST_S1, encoding="utf-8", errors="replace") as src, \
         open(MATCHING,   "w", encoding="utf-8", newline="") as fmatch, \
         open(CANDIDATES, "w", encoding="utf-8", newline="") as fcand:

        # Write headers
        fmatch.write("source1_entity_id\tmatched_entity_ids\n")
        fcand.write("source1_entity_id\tcandidate_entity_ids\n")

        # Skip source header
        src.readline()

        for line in src:
            entity_id = line.split("\t", 1)[0].strip()
            if not entity_id:
                continue
            # Empty match list and empty candidate list (just the entity_id + tab + newline)
            fmatch.write(f"{entity_id}\t\n")
            fcand.write(f"{entity_id}\t\n")
            n += 1
            if n % 200_000 == 0:
                print(f"  … {n:,} rows written ({time.time()-t0:.1f}s)", flush=True)

    elapsed = time.time() - t0
    print(f"Done — {n:,} rows written in {elapsed:.1f}s", flush=True)
    print(f"  matching_results.tsv  → {MATCHING}")
    print(f"  candidate_pairs.tsv   → {CANDIDATES}")
    return n


if __name__ == "__main__":
    count = build_fallback()
    if count == 0:
        print("ERROR: No rows written — check test_source1.tsv path.", file=sys.stderr)
        sys.exit(1)
    print("\nFallback submission files ready.")
