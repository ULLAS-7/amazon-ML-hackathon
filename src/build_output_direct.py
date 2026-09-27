"""
build_output_direct.py — Streaming output builder (no-model mode).

Reads test_source1.tsv for the complete ordered S1 entity list, then
streams candidate_pairs.tsv to build both output files in a single pass.
No large in-memory structures — O(N) memory where N = number of S1 entities.

Usage:
    python src/build_output_direct.py \
        --candidates path/to/test_candidate_pairs_vectorized.tsv \
        --source1    path/to/test_source1.tsv \
        --output-dir output_v4
"""
import argparse, csv, os, sys
from pathlib import Path

BUF = 64 << 20  # 64 MB write buffer

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--candidates", required=True)
    p.add_argument("--source1",    required=True)
    p.add_argument("--output-dir", default="output_v4")
    args = p.parse_args()

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    match_out = os.path.join(args.output_dir, "matching_results.tsv")
    cand_out  = os.path.join(args.output_dir, "candidate_pairs.tsv")

    # Load candidate map from blocking output (streaming, store only IDs)
    print(f"Loading candidates from {args.candidates} …", flush=True)
    cand_map: dict[str, str] = {}
    with open(args.candidates, encoding="utf-8") as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) >= 1:
                cand_map[parts[0]] = parts[1] if len(parts) > 1 else ""
    print(f"  {len(cand_map):,} entries loaded", flush=True)

    # Stream source1 for ordered S1 IDs, write both output files
    print(f"Streaming source1 and writing outputs …", flush=True)
    total = non_empty = 0
    TAB, NL = b"\t", b"\n"

    with open(args.source1, encoding="utf-8", newline="") as s1f, \
         open(match_out, "wb", buffering=BUF) as mf, \
         open(cand_out,  "wb", buffering=BUF) as cf:

        # Write required headers
        mf.write(b"source1_entity_id\tmatched_entity_ids\n")
        cf.write(b"source1_entity_id\tcandidate_entity_ids\n")

        reader = csv.DictReader(s1f, delimiter="\t")
        mbatch: list[bytes] = []
        cbatch: list[bytes] = []

        for row in reader:
            eid = (row.get("entity_id") or "").strip()
            if not eid:
                continue
            cands = cand_map.get(eid, "")
            line = eid.encode() + TAB + cands.encode() + NL
            mbatch.append(line)
            cbatch.append(line)
            total += 1
            if cands:
                non_empty += 1
            if len(mbatch) >= 50_000:
                mf.write(b"".join(mbatch)); mbatch.clear()
                cf.write(b"".join(cbatch)); cbatch.clear()

        if mbatch:
            mf.write(b"".join(mbatch))
            cf.write(b"".join(cbatch))

    print(f"Done. {total:,} rows, {non_empty:,} non-empty.", flush=True)
    print(f"  {match_out}")
    print(f"  {cand_out}")

if __name__ == "__main__":
    main()
