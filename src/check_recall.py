"""
check_recall.py — Measure blocking recall against the first 2000 non-singleton
                  ground-truth entities using the full candidate_pairs.tsv.

Usage:
    python src/check_recall.py
"""

from __future__ import annotations
import csv
from pathlib import Path

GT_PATH        = "dataset/train/train_ground_truth.tsv"
CANDS_PATH     = "submissions/candidate_pairs.tsv"
S1_PATH        = "dataset/train/train_source1.tsv"
SAMPLE_SIZE    = 2000

# ---------------------------------------------------------------------------
# Step 1: collect first 2000 non-singleton GT entries
# ---------------------------------------------------------------------------
print(f"[1/4] Streaming {GT_PATH} for first {SAMPLE_SIZE:,} non-singleton entities …")
gt: dict[str, set[str]] = {}
with open(GT_PATH, encoding="utf-8", errors="replace") as fh:
    fh.readline()  # skip header
    for line in fh:
        if len(gt) >= SAMPLE_SIZE:
            break
        line = line.rstrip("\n")
        if not line:
            continue
        parts = line.split("\t")
        s1id = parts[0].strip()
        matched_str = parts[1].strip() if len(parts) > 1 else ""
        if not matched_str:
            continue  # skip singletons
        gt[s1id] = set(matched_str.split(","))

print(f"      Collected {len(gt):,} GT entities.")

# ---------------------------------------------------------------------------
# Step 2: stream candidate_pairs.tsv, keep only entries for GT keys
# ---------------------------------------------------------------------------
print(f"[2/4] Streaming {CANDS_PATH} (line-by-line, keeping only GT keys) …")
cands: dict[str, set[str]] = {}
gt_keys = set(gt.keys())
lines_read = 0
with open(CANDS_PATH, encoding="utf-8", errors="replace") as fh:
    for line in fh:
        lines_read += 1
        parts = line.rstrip("\n").split("\t")
        if len(parts) < 2:
            continue
        s1id = parts[0].strip()
        if s1id not in gt_keys:
            continue
        cand_str = parts[1].strip()
        cands[s1id] = set(cand_str.split(",")) if cand_str else set()

print(f"      Read {lines_read:,} candidate lines; matched {len(cands):,} GT keys.")

# ---------------------------------------------------------------------------
# Step 3: compute per-entity recall
# ---------------------------------------------------------------------------
print("[3/4] Computing recall …")
recalls: list[float] = []
zero_recall_ids: list[str] = []

for s1id, true_matches in gt.items():
    candidate_set = cands.get(s1id, set())
    hits = len(true_matches & candidate_set)
    r = hits / len(true_matches) if true_matches else 1.0
    recalls.append(r)
    if r == 0.0:
        zero_recall_ids.append(s1id)

avg_recall   = sum(recalls) / len(recalls) if recalls else 0.0
n_zero       = len(zero_recall_ids)

# ---------------------------------------------------------------------------
# Step 4: look up business_name for up to 3 zero-recall entities
# ---------------------------------------------------------------------------
print("[4/4] Looking up zero-recall entity names …")
zero_details: dict[str, str] = {}
if zero_recall_ids:
    target_ids = set(zero_recall_ids[:3])
    with open(S1_PATH, encoding="utf-8", errors="replace", newline="") as fh:
        reader = csv.DictReader(fh, delimiter="\t")
        for row in reader:
            eid = row.get("entity_id", "").strip()
            if eid in target_ids:
                zero_details[eid] = row.get("business_name", "").strip()
            if len(zero_details) == len(target_ids):
                break

# ---------------------------------------------------------------------------
# Results
# ---------------------------------------------------------------------------
print()
print("=" * 60)
print("BLOCKING RECALL CHECK  (first 2,000 non-singleton GT entities)")
print("=" * 60)
print(f"  GT entities evaluated : {len(recalls):,}")
print(f"  Avg recall            : {avg_recall*100:.2f}%")
print(f"  Entities w/ recall=1.0: {sum(1 for r in recalls if r==1.0):,}")
print(f"  Entities w/ recall=0.0: {n_zero:,}")
if n_zero > 0:
    print(f"\n  Up to 3 zero-recall entities:")
    for eid in zero_recall_ids[:3]:
        name = zero_details.get(eid, "<not found>")
        true_m = gt[eid]
        cand_m = cands.get(eid, set())
        print(f"    {eid}  name={name!r}")
        print(f"      true_matches={sorted(true_m)}")
        print(f"      candidates  ={sorted(cand_m)[:10]}{'…' if len(cand_m)>10 else ''}")
print("=" * 60)
