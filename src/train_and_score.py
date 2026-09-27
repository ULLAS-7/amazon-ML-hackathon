"""
train_and_score.py — End-to-end entity matching pipeline.

Steps:
  1. Build labeled pairs from partial train candidate_pairs.tsv + ground truth
     (sample: first 50k S1 entities, max 20 negatives per entity)
  2. Compute 10 pairwise features (reusing features.py)
  3. Train LightGBM, pick F_0.5-optimal threshold on validation split
  4. Score test_candidate_pairs_trimmed.tsv in 100k-entity chunks
  5. Write submissions/test_final_scored.tsv

Usage (run from repo-model root):
    python src/train_and_score.py
"""

from __future__ import annotations
import csv, json, os, sys, time, random
import numpy as np
from pathlib import Path

# ── path setup ────────────────────────────────────────────────────────────
REPO_MODEL   = Path(__file__).resolve().parent.parent
REPO_BLOCKING = REPO_MODEL.parent / "repo-blocking"
sys.path.insert(0, str(REPO_MODEL / "src"))

from features import compute_pair_features
import lightgbm as lgb
from sklearn.metrics import fbeta_score, precision_score, recall_score
from sklearn.model_selection import train_test_split

# ── config ────────────────────────────────────────────────────────────────
TRAIN_S1_PATH  = REPO_BLOCKING / "dataset/train/train_source1.tsv"
TRAIN_S2_PATH  = REPO_BLOCKING / "dataset/train/train_source2.tsv"
TRAIN_S3_PATH  = REPO_BLOCKING / "dataset/train/train_source3.tsv"
GT_PATH        = REPO_BLOCKING / "dataset/train/train_ground_truth.tsv"
TRAIN_CANDS    = REPO_BLOCKING / "submissions/candidate_pairs.tsv"
TEST_CANDS     = REPO_BLOCKING / "submissions/test_candidate_pairs_trimmed.tsv"
TEST_S1_PATH   = REPO_BLOCKING / "dataset/test/test_source1.tsv"
TEST_S2_PATH   = REPO_BLOCKING / "dataset/test/test_source2.tsv"
TEST_S3_PATH   = REPO_BLOCKING / "dataset/test/test_source3.tsv"
SCORED_OUT     = REPO_BLOCKING / "submissions/test_final_scored.tsv"
MODEL_PATH     = REPO_MODEL / "models/matcher.txt"
THRESH_PATH    = REPO_MODEL / "models/threshold.json"

MAX_S1_TRAIN   = 50_000   # first N S1 entities from train candidates
MAX_NEG_PER_S1 = 20       # cap negatives per entity (balance)
CHUNK_SIZE     = 100_000  # S1 entities per scoring chunk
BUF            = 64 << 20 # 64 MB write buffer
RANDOM_SEED    = 42
FEATURE_NAMES  = [
    "name_jaccard","name_lev_ratio","name_token_sort_ratio",
    "addr_jaccard","addr_lev_ratio","house_num_match",
    "city_region_match","country_match","name_len_diff","addr_len_diff",
]

random.seed(RANDOM_SEED)
t_global = time.time()

def elapsed() -> str:
    return f"{(time.time()-t_global)/60:.1f}min"

def ram_gb() -> float:
    try:
        import psutil
        return psutil.virtual_memory().used / 1e9
    except:
        return -1.0

# ─────────────────────────────────────────────────────────────────────────
# STEP 1 — Build labeled training pairs
# ─────────────────────────────────────────────────────────────────────────
print(f"\n{'='*60}")
print(f"STEP 1 — Build labeled pairs  [{elapsed()}]")
print(f"{'='*60}", flush=True)

t0 = time.time()

# 1a. Load ground truth into dict: s1_id → set of true match IDs
print("  Loading ground truth …", flush=True)
gt: dict[str, set[str]] = {}
with open(GT_PATH, encoding="utf-8") as f:
    f.readline()  # header
    for line in f:
        parts = line.rstrip("\n").split("\t")
        s1id = parts[0].strip()
        matched = parts[1].strip() if len(parts) > 1 else ""
        gt[s1id] = set(matched.split(",")) if matched else set()
print(f"  GT loaded: {len(gt):,} S1 entities  [{time.time()-t0:.1f}s]", flush=True)

# 1b. Load train S1/S2/S3 entity rows into lookup dicts
print("  Loading train source rows …", flush=True)
t1 = time.time()

def load_lookup(path: str) -> dict[str, dict]:
    out = {}
    with open(path, encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            eid = (row.get("entity_id") or "").strip()
            if eid:
                out[eid] = dict(row)
    return out

s1_lookup = load_lookup(str(TRAIN_S1_PATH))
s2_lookup = load_lookup(str(TRAIN_S2_PATH))
s3_lookup = load_lookup(str(TRAIN_S3_PATH))
cand_lookup = {**s2_lookup, **s3_lookup}
del s2_lookup, s3_lookup
print(f"  S1={len(s1_lookup):,}  cand={len(cand_lookup):,}  [{time.time()-t1:.1f}s]",
      flush=True)

# 1c. Stream train candidate_pairs.tsv, build labeled pairs (sample)
print(f"  Building labeled pairs (first {MAX_S1_TRAIN:,} S1 entities, "
      f"max {MAX_NEG_PER_S1} neg/entity) …", flush=True)
t1 = time.time()

X_rows: list[dict] = []
y_labels: list[int] = []
groups: list[str] = []
s1_seen = 0
pos_total = neg_total = 0

with open(TRAIN_CANDS, encoding="utf-8") as f:
    for line in f:
        parts = line.rstrip("\n").split("\t")
        s1id = parts[0].strip()
        cands_str = parts[1].strip() if len(parts) > 1 else ""
        if not cands_str:
            continue
        s1_seen += 1
        if s1_seen > MAX_S1_TRAIN:
            break

        cand_ids = cands_str.split(",")
        s1_row = s1_lookup.get(s1id)
        if s1_row is None:
            continue
        true_matches = gt.get(s1id, set())

        positives = [c for c in cand_ids if c in true_matches]
        negatives = [c for c in cand_ids if c not in true_matches]

        # subsample negatives
        if len(negatives) > MAX_NEG_PER_S1:
            negatives = random.sample(negatives, MAX_NEG_PER_S1)

        for cid in positives + negatives:
            cand_row = cand_lookup.get(cid)
            if cand_row is None:
                continue
            feats = compute_pair_features(s1_row, cand_row)
            X_rows.append(feats)
            label = 1 if cid in true_matches else 0
            y_labels.append(label)
            groups.append(s1id)
            if label == 1: pos_total += 1
            else:          neg_total += 1

print(f"  Pairs: {len(X_rows):,}  pos={pos_total:,}  neg={neg_total:,}  "
      f"[{time.time()-t1:.1f}s]", flush=True)

if pos_total == 0:
    print("ERROR: Zero positive labels — check GT alignment and candidate file.", flush=True)
    sys.exit(1)

# Free lookup dicts — no longer needed after pair building
del cand_lookup, s1_lookup

# ─────────────────────────────────────────────────────────────────────────
# STEP 2 — Build feature matrix
# ─────────────────────────────────────────────────────────────────────────
print(f"\n{'='*60}")
print(f"STEP 2 — Build feature matrix  [{elapsed()}]")
print(f"{'='*60}", flush=True)

X = np.array([[row[f] for f in FEATURE_NAMES] for row in X_rows], dtype=np.float32)
y = np.array(y_labels, dtype=np.int32)
del X_rows, y_labels
print(f"  X.shape={X.shape}  RAM≈{ram_gb():.1f}GB  [{elapsed()}]", flush=True)

# ─────────────────────────────────────────────────────────────────────────
# STEP 3 — Train LightGBM + F_0.5-optimal threshold
# ─────────────────────────────────────────────────────────────────────────
print(f"\n{'='*60}")
print(f"STEP 3 — Train LightGBM  [{elapsed()}]")
print(f"{'='*60}", flush=True)

# Stratified 80/20 split
X_train, X_val, y_train, y_val, g_train, g_val = train_test_split(
    X, y, groups,
    test_size=0.2, random_state=RANDOM_SEED, stratify=y
)
print(f"  Train: {len(y_train):,} (pos={y_train.sum():,})  "
      f"Val: {len(y_val):,} (pos={y_val.sum():,})", flush=True)

scale_pos = int((y_train == 0).sum()) / max(int((y_train == 1).sum()), 1)
print(f"  scale_pos_weight={scale_pos:.1f}", flush=True)

t1 = time.time()
model = lgb.LGBMClassifier(
    objective="binary",
    n_estimators=200,
    learning_rate=0.05,
    num_leaves=31,
    max_depth=5,
    min_child_samples=10,
    subsample=0.8,
    colsample_bytree=0.8,
    scale_pos_weight=scale_pos,
    random_state=RANDOM_SEED,
    verbosity=-1,
    n_jobs=-1,
)
model.fit(
    X_train, y_train,
    eval_set=[(X_val, y_val)],
    callbacks=[lgb.early_stopping(30, verbose=False), lgb.log_evaluation(-1)],
)
print(f"  Trained in {time.time()-t1:.1f}s  best_iter={model.best_iteration_}", flush=True)
del X_train, X  # free training data

# Threshold sweep on validation — compute ENTITY-LEVEL macro F_0.5
# (group by s1_id, count TP/FP/FN per entity, then average)
probs_val = model.predict_proba(X_val)[:, 1]

def macro_f05_entity(probs, y_true, groups_list, thresh):
    """Compute macro-averaged entity-level F_0.5 at a given threshold."""
    from collections import defaultdict
    # group by entity
    entity_pred: dict[str, list[int]] = defaultdict(list)
    entity_true: dict[str, list[int]] = defaultdict(list)
    for p, yt, g in zip(probs, y_true, groups_list):
        pred = 1 if p >= thresh else 0
        entity_pred[g].append(pred)
        entity_true[g].append(yt)

    f05_scores = []
    for eid in entity_pred:
        preds = np.array(entity_pred[eid])
        trues = np.array(entity_true[eid])
        f05 = fbeta_score(trues, preds, beta=0.5, zero_division=1.0
                          if trues.sum() == 0 else 0)
        f05_scores.append(f05)
    return float(np.mean(f05_scores)) if f05_scores else 0.0

print("  Sweeping thresholds for macro F_0.5 …", flush=True)
best_thresh, best_f05 = 0.5, -1.0
for thresh in np.arange(0.10, 0.96, 0.05):
    f05 = macro_f05_entity(probs_val, y_val, g_val, thresh)
    if f05 > best_f05:
        best_f05, best_thresh = f05, float(thresh)

# Fine-grained sweep around best
for thresh in np.arange(max(0.05, best_thresh - 0.05),
                         min(0.99, best_thresh + 0.06), 0.01):
    f05 = macro_f05_entity(probs_val, y_val, g_val, thresh)
    if f05 > best_f05:
        best_f05, best_thresh = f05, float(thresh)

final_preds = (probs_val >= best_thresh).astype(int)
prec = precision_score(y_val, final_preds, zero_division=0)
rec  = recall_score(y_val, final_preds, zero_division=0)
f05_pair = fbeta_score(y_val, final_preds, beta=0.5, zero_division=0)

print(f"\n  {'='*50}")
print(f"  MODEL RESULTS")
print(f"  {'='*50}")
print(f"  Threshold (max entity F_0.5): {best_thresh:.2f}")
print(f"  Entity-level macro F_0.5:     {best_f05:.4f}")
print(f"  Pair-level F_0.5:             {f05_pair:.4f}")
print(f"  Precision (pair-level):       {prec:.4f}")
print(f"  Recall (pair-level):          {rec:.4f}")
print(f"  {'='*50}", flush=True)

if best_f05 < 0.10:
    print(f"\nWARNING: Entity F_0.5={best_f05:.4f} is very low — "
          "labeling or feature issue likely.", flush=True)
    # Still save and proceed — let the caller decide

# Save
Path(MODEL_PATH).parent.mkdir(parents=True, exist_ok=True)
model.booster_.save_model(str(MODEL_PATH))
with open(THRESH_PATH, "w") as fh:
    json.dump({"threshold": best_thresh, "val_f05": best_f05,
               "val_precision": prec, "val_recall": rec}, fh, indent=2)
print(f"  Saved model → {MODEL_PATH}", flush=True)
print(f"  Saved threshold → {THRESH_PATH}", flush=True)
del X_val, y_val, probs_val

# ─────────────────────────────────────────────────────────────────────────
# STEP 4 — Load test source lookups + score test candidates in chunks
# ─────────────────────────────────────────────────────────────────────────
print(f"\n{'='*60}")
print(f"STEP 4 — Score test candidates in chunks  [{elapsed()}]")
print(f"{'='*60}", flush=True)

print("  Loading test source lookups …", flush=True)
t1 = time.time()
ts1 = load_lookup(str(TEST_S1_PATH))
ts2 = load_lookup(str(TEST_S2_PATH))
ts3 = load_lookup(str(TEST_S3_PATH))
test_cand_lookup = {**ts2, **ts3}
del ts2, ts3
print(f"  S1={len(ts1):,}  cand_lookup={len(test_cand_lookup):,}  "
      f"[{time.time()-t1:.1f}s]  RAM≈{ram_gb():.1f}GB", flush=True)

booster = lgb.Booster(model_file=str(MODEL_PATH))

# Stream TEST_CANDS in chunks of CHUNK_SIZE S1 entities
print(f"  Scoring {TEST_CANDS.name} in chunks of {CHUNK_SIZE:,} …", flush=True)
t1 = time.time()

scored_rows: list[tuple[str, str]] = []  # (s1id, comma-joined kept cands)
chunk_s1ids: list[str] = []
chunk_cands: dict[str, list[str]] = {}   # s1id → [cand_ids]

total_s1 = s1_written = s1_nonempty = pairs_scored = pairs_kept = 0

def _score_chunk(chunk_s1ids, chunk_cands, ts1, test_cand_lookup, booster, thresh):
    """Score one chunk; return list of (s1id, kept_cands_str)."""
    results = []
    # Build feature matrix for all pairs in chunk
    pair_s1ids = []
    pair_cids  = []
    feat_rows  = []
    for s1id in chunk_s1ids:
        s1_row = ts1.get(s1id)
        if s1_row is None:
            results.append((s1id, ""))
            continue
        cids = chunk_cands.get(s1id, [])
        if not cids:
            results.append((s1id, ""))
            continue
        for cid in cids:
            cand_row = test_cand_lookup.get(cid)
            if cand_row is None:
                continue
            feat_rows.append(compute_pair_features(s1_row, cand_row))
            pair_s1ids.append(s1id)
            pair_cids.append(cid)

    if not feat_rows:
        for s1id in chunk_s1ids:
            results.append((s1id, ""))
        return results, 0, 0

    X_chunk = np.array([[r[f] for f in FEATURE_NAMES] for r in feat_rows],
                        dtype=np.float32)
    probs = booster.predict(X_chunk)

    # Group kept candidates by s1id
    kept: dict[str, list[str]] = {}
    for s1id, cid, prob in zip(pair_s1ids, pair_cids, probs):
        if prob >= thresh:
            kept.setdefault(s1id, []).append(cid)

    for s1id in chunk_s1ids:
        cands_kept = kept.get(s1id, [])
        results.append((s1id, ",".join(cands_kept)))

    return results, len(feat_rows), int((probs >= thresh).sum())

TAB, NL = b"\t", b"\n"
with open(TEST_CANDS, encoding="utf-8") as fin, \
     open(SCORED_OUT, "wb", buffering=BUF) as fout:

    # Write header
    fout.write(b"source1_entity_id\tcandidate_entity_ids\n")

    for line in fin:
        parts = line.rstrip("\n").split("\t")
        s1id = parts[0].strip()
        if not s1id:
            continue
        cands_str = parts[1].strip() if len(parts) > 1 else ""
        cids = cands_str.split(",") if cands_str else []

        chunk_s1ids.append(s1id)
        chunk_cands[s1id] = cids
        total_s1 += 1

        if len(chunk_s1ids) >= CHUNK_SIZE:
            results, n_scored, n_kept = _score_chunk(
                chunk_s1ids, chunk_cands, ts1, test_cand_lookup,
                booster, best_thresh
            )
            pairs_scored += n_scored
            pairs_kept   += n_kept
            batch = []
            for s1id_r, cands_r in results:
                batch.append(s1id_r.encode() + TAB + cands_r.encode() + NL)
                s1_written += 1
                if cands_r: s1_nonempty += 1
            fout.write(b"".join(batch))
            elapsed_s = time.time() - t1
            rate = s1_written / elapsed_s if elapsed_s > 0 else 0
            print(f"  … {s1_written:>9,} written | nonempty={s1_nonempty:,} | "
                  f"{rate:,.0f} rows/s | {elapsed()}", flush=True)
            chunk_s1ids = []
            chunk_cands = {}

    # Final partial chunk
    if chunk_s1ids:
        results, n_scored, n_kept = _score_chunk(
            chunk_s1ids, chunk_cands, ts1, test_cand_lookup,
            booster, best_thresh
        )
        pairs_scored += n_scored
        pairs_kept   += n_kept
        batch = []
        for s1id_r, cands_r in results:
            batch.append(s1id_r.encode() + TAB + cands_r.encode() + NL)
            s1_written += 1
            if cands_r: s1_nonempty += 1
        fout.write(b"".join(batch))

out_mb = SCORED_OUT.stat().st_size / 1e6
print(f"\n  Scoring complete:", flush=True)
print(f"    S1 rows written : {s1_written:,}")
print(f"    Non-empty       : {s1_nonempty:,}")
print(f"    Pairs scored    : {pairs_scored:,}")
print(f"    Pairs kept      : {pairs_kept:,}")
print(f"    Output size     : {out_mb:.1f} MB")
print(f"    Elapsed         : {elapsed()}", flush=True)

print(f"\n{'='*60}")
print(f"ALL STEPS COMPLETE — {elapsed()}")
print(f"  Model:     {MODEL_PATH}")
print(f"  Threshold: {best_thresh:.2f}  (val entity F_0.5={best_f05:.4f})")
print(f"  Scored:    {SCORED_OUT}")
print(f"  Non-empty: {s1_nonempty:,} / {s1_written:,}")
print(f"{'='*60}", flush=True)
