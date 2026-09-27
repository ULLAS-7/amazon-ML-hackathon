"""
quick_recall_check.py — Validate recall on the 2000 GT-aligned entities WITHOUT
regenerating the full candidate_pairs.tsv.

Runs blocking directly in-memory for those 2000 S1 entities against the full
S2/S3 indexes built with the current scale_blocking.py filter logic, then
measures recall against ground truth.  Takes ~10 min (index build) but avoids
writing a 2+ GB file.
"""

from __future__ import annotations
import collections, csv, sys, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from blocking import (
    build_frequency_stopwords, generate_blocking_keys,
    get_candidates, load_sample,
)
from scale_blocking import (
    _build_index_from_file, _filter_index,
    CITY_CAP, FREQ_SAMPLE_SIZE,
)

GT_PATH  = "dataset/train/train_ground_truth.tsv"
S1_PATH  = "dataset/train/train_source1.tsv"
S2_PATH  = "dataset/train/train_source2.tsv"
S3_PATH  = "dataset/train/train_source3.tsv"
N_GT     = 2000

# ── Step 1: collect 2000 GT entities ────────────────────────────────────────
print(f"[1/5] Collecting first {N_GT} non-singleton GT entities …")
gt: dict[str, set[str]] = {}
with open(GT_PATH, encoding="utf-8", errors="replace") as fh:
    fh.readline()
    for line in fh:
        if len(gt) >= N_GT:
            break
        line = line.rstrip("\n")
        if not line:
            continue
        parts = line.split("\t")
        s1id = parts[0].strip()
        matched = parts[1].strip() if len(parts) > 1 else ""
        if matched:
            gt[s1id] = set(matched.split(","))
print(f"      {len(gt):,} GT entities collected.")

# ── Step 2: fetch S1 rows for those 2000 entities ───────────────────────────
print(f"[2/5] Streaming S1 for {len(gt):,} target entities …")
s1_rows: dict[str, dict] = {}
with open(S1_PATH, encoding="utf-8", errors="replace", newline="") as fh:
    reader = csv.DictReader(fh, delimiter="\t")
    for row in reader:
        eid = row.get("entity_id", "").strip()
        if eid in gt:
            s1_rows[eid] = dict(row)
        if len(s1_rows) == len(gt):
            break
print(f"      {len(s1_rows):,} S1 rows fetched.")

# ── Step 3: freq stopwords ───────────────────────────────────────────────────
print(f"[3/5] Building freq_stopwords …")
t0 = time.time()
sample = load_sample(S1_PATH, n=FREQ_SAMPLE_SIZE)
freq_sw = build_frequency_stopwords(sample, threshold=50)
print(f"      freq_stopwords ({len(freq_sw)}): {sorted(freq_sw)}  [{time.time()-t0:.1f}s]")

# ── Step 4: build filtered S2+S3 index ──────────────────────────────────────
print("[4/5] Building filtered S2+S3 index …")
t0 = time.time()
raw_s2 = _build_index_from_file(S2_PATH, freq_sw, "S2")
idx_s2 = _filter_index(raw_s2, "S2"); del raw_s2
raw_s3 = _build_index_from_file(S3_PATH, freq_sw, "S3")
idx_s3 = _filter_index(raw_s3, "S3"); del raw_s3

combined: dict[str, list[str]] = collections.defaultdict(list)
for k, v in idx_s2.items(): combined[k].extend(v)
for k, v in idx_s3.items(): combined[k].extend(v)
combined_index = dict(combined); del idx_s2, idx_s3, combined
print(f"      Combined index: {len(combined_index):,} keys  [{time.time()-t0:.1f}s]")

# ── Step 5: compute recall ───────────────────────────────────────────────────
print("[5/5] Computing recall for 2000 GT entities …")
recalls: list[float] = []
zero_ids: list[str]  = []
total_cands = 0

for s1id, true_matches in gt.items():
    row = s1_rows.get(s1id)
    if row is None:
        recalls.append(0.0)
        zero_ids.append(s1id)
        continue
    cands = get_candidates(
        s1id,
        row.get("business_name", "") or "",
        row.get("business_address", "") or "",
        row.get("country", "") or "",
        combined_index,
        freq_sw,
        city_cap=CITY_CAP,
    )
    filtered = {c for c in cands if c.startswith("S2-") or c.startswith("S3-")}
    hits = len(true_matches & filtered)
    r = hits / len(true_matches) if true_matches else 1.0
    recalls.append(r)
    total_cands += len(filtered)
    if r == 0.0:
        zero_ids.append(s1id)

avg_recall = sum(recalls) / len(recalls) if recalls else 0.0
avg_cands  = total_cands / len(recalls) if recalls else 0.0

print()
print("=" * 60)
print("QUICK RECALL CHECK  (2,000 GT-aligned entities, full S2/S3 index)")
print("=" * 60)
print(f"  GT entities evaluated : {len(recalls):,}")
print(f"  Avg recall            : {avg_recall*100:.2f}%")
print(f"  Avg candidates/S1     : {avg_cands:.1f}")
print(f"  Entities recall=1.0   : {sum(1 for r in recalls if r==1.0):,}")
print(f"  Entities recall=0.0   : {len(zero_ids):,}")
if zero_ids:
    print(f"\n  Up to 5 zero-recall entities:")
    for eid in zero_ids[:5]:
        row = s1_rows.get(eid, {})
        name = row.get("business_name", "<not found>")
        true_m = gt[eid]
        cands2 = get_candidates(
            eid,
            row.get("business_name","") or "",
            row.get("business_address","") or "",
            row.get("country","") or "",
            combined_index, freq_sw, city_cap=CITY_CAP,
        )
        filtered2 = sorted(c for c in cands2 if c.startswith("S2-") or c.startswith("S3-"))
        print(f"    {eid}  {name!r}")
        print(f"      true={sorted(true_m)}")
        print(f"      cands={filtered2[:5]}{'…' if len(filtered2)>5 else ''}")
print("=" * 60)
