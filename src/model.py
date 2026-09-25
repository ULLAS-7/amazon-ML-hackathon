"""
model.py — Baseline LightGBM entity-matching pipeline.

Pipeline
--------
1. Load an aligned N=2000 sample via blocking.get_aligned_sample.
2. Build a blocking index and generate candidate pairs via blocking.get_candidates.
3. Label each candidate pair: 1 if the candidate is a true match per GT, else 0.
4. Compute pairwise features via features.compute_pair_features.
5. Train a LightGBM binary classifier on an 80/20 train/val split.
6. Select the probability threshold that maximises F_0.5 on validation.
7. Report F_0.5, precision, recall, chosen threshold, and top-5 feature importances.

Usage
-----
    python src/model.py
"""

from __future__ import annotations

import os
import sys
import time

# Allow running from repo root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import lightgbm as lgb
import numpy as np
from sklearn.metrics import fbeta_score, precision_score, recall_score
from sklearn.model_selection import train_test_split

from src.blocking import (
    build_blocking_index,
    build_frequency_stopwords,
    get_aligned_sample,
    get_candidates,
)
from src.features import compute_pair_features

# ---------------------------------------------------------------------------
# Paths (relative to repo root)
# ---------------------------------------------------------------------------

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TRAIN_DIR = os.path.join(REPO_ROOT, "dataset", "train")

GT_PATH = os.path.join(TRAIN_DIR, "train_ground_truth.tsv")
S1_PATH = os.path.join(TRAIN_DIR, "train_source1.tsv")
S2_PATH = os.path.join(TRAIN_DIR, "train_source2.tsv")
S3_PATH = os.path.join(TRAIN_DIR, "train_source3.tsv")

SAMPLE_SIZE = 2000
RANDOM_STATE = 42


# ---------------------------------------------------------------------------
# Step 1 — Aligned sample
# ---------------------------------------------------------------------------

def load_aligned_sample(n: int = SAMPLE_SIZE):
    """Return (s1_rows, s2_rows, s3_rows, gt_rows) for n Source-1 entities."""
    print(f"[1/5] Loading aligned sample (n={n}) …", flush=True)
    t0 = time.time()
    s1_rows, s2_rows, s3_rows, gt_rows = get_aligned_sample(
        GT_PATH, S1_PATH, S2_PATH, S3_PATH, n=n
    )
    print(
        f"      s1={len(s1_rows):,}  s2={len(s2_rows):,}  "
        f"s3={len(s3_rows):,}  gt={len(gt_rows):,}  "
        f"({time.time()-t0:.1f}s)",
        flush=True,
    )
    return s1_rows, s2_rows, s3_rows, gt_rows


# ---------------------------------------------------------------------------
# Step 2 — Candidate pair generation
# ---------------------------------------------------------------------------

def build_candidate_pairs(
    s1_rows: list[dict],
    s2_rows: list[dict],
    s3_rows: list[dict],
    gt_rows: list[dict],
):
    """Return (pairs, labels) where pairs is a list of (s1_row, cand_row)."""
    print("[2/5] Building blocking index and generating candidates …", flush=True)
    t0 = time.time()

    # Build a single lookup dict: entity_id → row (for S2 + S3 candidates)
    cand_lookup: dict[str, dict] = {}
    for row in s2_rows:
        cand_lookup[row["entity_id"]] = row
    for row in s3_rows:
        cand_lookup[row["entity_id"]] = row

    # Build the index from the candidate pool (S2 + S3)
    all_candidates = s2_rows + s3_rows
    freq_sw = build_frequency_stopwords(all_candidates, threshold=50)
    index = build_blocking_index(all_candidates, freq_sw)

    # Build GT lookup: s1_id → set of matched candidate ids
    gt_lookup: dict[str, set[str]] = {}
    for gt in gt_rows:
        s1_id = gt["source1_entity_id"]
        matched_str = gt.get("matched_entity_ids", "")
        matched = set(matched_str.split(",")) if matched_str.strip() else set()
        gt_lookup[s1_id] = matched

    # Build S1 lookup
    s1_lookup: dict[str, dict] = {r["entity_id"]: r for r in s1_rows}

    pairs: list[tuple[dict, dict]] = []
    labels: list[int] = []
    total_pos = 0
    total_neg = 0

    for s1_row in s1_rows:
        s1_id = s1_row["entity_id"]
        true_matches = gt_lookup.get(s1_id, set())

        # Get candidate ids from blocking
        cand_ids = get_candidates(
            entity_id=s1_id,
            business_name=s1_row.get("business_name", ""),
            business_address=s1_row.get("business_address", ""),
            country=s1_row.get("country", ""),
            index=index,
            freq_stopwords=freq_sw,
        )

        # Ensure all true matches are included (guarantee recall for training)
        cand_ids = cand_ids | true_matches

        for cid in cand_ids:
            cand_row = cand_lookup.get(cid)
            if cand_row is None:
                continue  # not in our sample pool
            label = 1 if cid in true_matches else 0
            pairs.append((s1_row, cand_row))
            labels.append(label)
            if label == 1:
                total_pos += 1
            else:
                total_neg += 1

    print(
        f"      pairs={len(pairs):,}  pos={total_pos:,}  neg={total_neg:,}  "
        f"({time.time()-t0:.1f}s)",
        flush=True,
    )
    return pairs, labels


# ---------------------------------------------------------------------------
# Step 3 — Feature matrix
# ---------------------------------------------------------------------------

def build_feature_matrix(pairs: list[tuple[dict, dict]]) -> tuple[np.ndarray, list[str]]:
    """Compute features for every pair; return (X, feature_names)."""
    print(f"[3/5] Computing features for {len(pairs):,} pairs …", flush=True)
    t0 = time.time()

    rows = [compute_pair_features(s1, cand) for s1, cand in pairs]
    feature_names = list(rows[0].keys())
    X = np.array([[r[f] for f in feature_names] for r in rows], dtype=np.float32)

    print(f"      X.shape={X.shape}  ({time.time()-t0:.1f}s)", flush=True)
    return X, feature_names


# ---------------------------------------------------------------------------
# Step 4 — Train LightGBM
# ---------------------------------------------------------------------------

def train_model(
    X: np.ndarray,
    y: np.ndarray,
    feature_names: list[str],
):
    """Train an 80/20 split LightGBM classifier; return (model, X_val, y_val)."""
    print("[4/5] Training LightGBM (80/20 split) …", flush=True)
    t0 = time.time()

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )
    print(
        f"      train={len(y_train):,}  val={len(y_val):,}  "
        f"pos_train={y_train.sum():,}  pos_val={y_val.sum():,}",
        flush=True,
    )

    # Compute scale_pos_weight to handle class imbalance
    n_neg = int((y_train == 0).sum())
    n_pos = int((y_train == 1).sum())
    scale_pos_weight = n_neg / max(n_pos, 1)

    params = {
        "objective": "binary",
        "metric": "binary_logloss",
        "n_estimators": 300,
        "learning_rate": 0.05,
        "num_leaves": 31,
        "min_child_samples": 10,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "scale_pos_weight": scale_pos_weight,
        "random_state": RANDOM_STATE,
        "verbosity": -1,
        "n_jobs": -1,
    }

    model = lgb.LGBMClassifier(**params)
    model.fit(
        X_train,
        y_train,
        eval_X=X_val,
        eval_y=y_val,
        callbacks=[lgb.early_stopping(stopping_rounds=30, verbose=False),
                   lgb.log_evaluation(period=-1)],
        feature_name=feature_names,
    )

    print(f"      best_iteration={model.best_iteration_}  ({time.time()-t0:.1f}s)", flush=True)
    return model, X_val, y_val


# ---------------------------------------------------------------------------
# Step 5 — Threshold selection & evaluation
# ---------------------------------------------------------------------------

def evaluate(
    model: lgb.LGBMClassifier,
    X_val: np.ndarray,
    y_val: np.ndarray,
    feature_names: list[str],
):
    """Select threshold maximising F_0.5 on val; print final metrics."""
    print("[5/5] Threshold selection and evaluation …", flush=True)

    probs = model.predict_proba(X_val)[:, 1]

    # Sweep thresholds from 0.05 to 0.95
    best_thresh = 0.5
    best_f05 = -1.0
    thresholds = np.arange(0.05, 0.96, 0.01)

    for thresh in thresholds:
        preds = (probs >= thresh).astype(int)
        if preds.sum() == 0:
            continue
        f05 = fbeta_score(y_val, preds, beta=0.5, zero_division=0)
        if f05 > best_f05:
            best_f05 = f05
            best_thresh = thresh

    final_preds = (probs >= best_thresh).astype(int)
    final_f05 = fbeta_score(y_val, final_preds, beta=0.5, zero_division=0)
    final_prec = precision_score(y_val, final_preds, zero_division=0)
    final_rec = recall_score(y_val, final_preds, zero_division=0)

    # Feature importances (gain)
    importances = model.booster_.feature_importance(importance_type="gain")
    fi = sorted(zip(feature_names, importances), key=lambda x: x[1], reverse=True)

    print("\n" + "=" * 55)
    print("  BASELINE MODEL RESULTS")
    print("=" * 55)
    print(f"  Threshold (max F_0.5):  {best_thresh:.2f}")
    print(f"  F_0.5  (beta=0.5):      {final_f05:.4f}")
    print(f"  Precision:              {final_prec:.4f}")
    print(f"  Recall:                 {final_rec:.4f}")
    print()
    print("  Top-5 Feature Importances (gain):")
    for rank, (name, score) in enumerate(fi[:5], 1):
        print(f"    {rank}. {name:<30}  {score:.1f}")
    print("=" * 55)

    return {
        "threshold": float(best_thresh),
        "f05": final_f05,
        "precision": final_prec,
        "recall": final_rec,
        "feature_importances": fi,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("\n=== Baseline LightGBM Entity Matching Pipeline ===\n")

    # 1. Aligned sample
    s1_rows, s2_rows, s3_rows, gt_rows = load_aligned_sample(SAMPLE_SIZE)

    # 2. Candidate pairs + labels
    pairs, labels = build_candidate_pairs(s1_rows, s2_rows, s3_rows, gt_rows)

    if not pairs:
        print("ERROR: No candidate pairs generated. Check blocking index.", file=sys.stderr)
        sys.exit(1)

    # 3. Feature matrix
    X, feature_names = build_feature_matrix(pairs)
    y = np.array(labels, dtype=np.int32)

    if y.sum() == 0:
        print("ERROR: No positive labels found. Check GT alignment.", file=sys.stderr)
        sys.exit(1)

    # 4. Train
    model, X_val, y_val = train_model(X, y, feature_names)

    # 5. Evaluate
    results = evaluate(model, X_val, y_val, feature_names)

    return results


if __name__ == "__main__":
    main()
