"""
build_submission.py — End-to-end submission assembly script.

Takes a candidate_pairs.tsv (from the blocking stage) and a trained model,
runs the real predict_matches function from model.py, then writes both
output files required by the ML Challenge 2026 submission format:

    output/matching_results.tsv   — final matches (scored on leaderboard)
    output/candidate_pairs.tsv    — blocking candidates fed to the model

Usage
-----
    # With a real trained model (full integration):
    python src/build_submission.py \\
        --candidate-pairs-path path/to/candidate_pairs.tsv \\
        --model                models/matcher.txt \\
        --threshold-path       models/threshold.json \\
        --source1              dataset/test/test_source1.tsv \\
        --source2              dataset/test/test_source2.tsv \\
        --source3              dataset/test/test_source3.tsv \\
        --output-dir           output

    # Quick smoke-test / no model yet (threshold-only fallback):
    python src/build_submission.py \\
        --candidate-pairs-path submissions/candidate_pairs_dummy.tsv \\
        --source1 dataset/test/test_source1.tsv \\
        --source2 dataset/test/test_source2.tsv \\
        --source3 dataset/test/test_source3.tsv

Real final integration (once repo-blocking finishes full-scale blocking)
------------------------------------------------------------------------
    python src/build_submission.py \\
        --candidate-pairs-path ../repo-blocking/submissions/candidate_pairs.tsv \\
        --model                models/matcher.txt \\
        --threshold-path       models/threshold.json \\
        --source1              ../repo-blocking/dataset/test/test_source1.tsv \\
        --source2              ../repo-blocking/dataset/test/test_source2.tsv \\
        --source3              ../repo-blocking/dataset/test/test_source3.tsv \\
        --output-dir           output

Design notes
------------
predict_matches() (from model.py) works on *row dicts* — it needs the full
business_name / business_address / country fields to compute similarity
features.  This script therefore loads all three test source TSVs into an
in-memory lookup (entity_id → row), then assembles (s1_row, cand_row) tuples
for each candidate pair before calling predict_matches.

For the full-scale test set (~1.7 M records) the source TSVs each fit
comfortably in RAM (they are the lookup side, not the cartesian product).
The candidate_pairs.tsv is streamed line-by-line so the script never holds
the entire 16 GB file in memory at once.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from typing import Optional

# ---------------------------------------------------------------------------
# Ensure src/ is on sys.path when the script is run directly
# ---------------------------------------------------------------------------
_SRC_DIR = os.path.dirname(os.path.abspath(__file__))
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

from utils import write_matching_results, write_candidate_pairs

# Real predict_matches — imported from model.py (copied from model-baseline branch).
# This replaces the dummy stub that was here during scaffolding.
try:
    from model import predict_matches as _real_predict_matches
    _MODEL_AVAILABLE = True
except ImportError as _model_import_err:
    _real_predict_matches = None  # type: ignore[assignment]
    _MODEL_AVAILABLE = False
    print(
        f"WARNING: could not import predict_matches from model.py "
        f"({_model_import_err}). Will run in no-model mode (all candidates "
        f"accepted as matches). Install lightgbm + rapidfuzz to enable the "
        f"real model.",
        file=sys.stderr,
    )


# ---------------------------------------------------------------------------
# Source TSV loaders
# ---------------------------------------------------------------------------

def load_entity_lookup(path: str) -> dict[str, dict]:
    """Load a source TSV into {entity_id: row_dict}.

    Handles source1, source2, and source3 — all share the same four columns:
    entity_id, business_name, business_address, country.

    Parameters
    ----------
    path : str — path to a tab-separated source file with a header row.

    Returns
    -------
    dict[str, dict]
    """
    lookup: dict[str, dict] = {}
    with open(path, encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh, delimiter="\t")
        for row in reader:
            eid = row.get("entity_id", "").strip()
            if eid:
                lookup[eid] = dict(row)
    print(f"[load_entity_lookup] {len(lookup):,} entities loaded from {path}")
    return lookup


def load_source1_ids(path: str) -> list[str]:
    """Return entity_ids from test_source1.tsv in file order.

    Every S1 entity must appear in the final output, even singletons with no
    blocking candidates.

    Parameters
    ----------
    path : str — path to test_source1.tsv.

    Returns
    -------
    list[str] — ordered list of S1 entity IDs.
    """
    ids: list[str] = []
    with open(path, encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh, delimiter="\t")
        for row in reader:
            eid = row.get("entity_id", "").strip()
            if eid:
                ids.append(eid)
    return ids


# ---------------------------------------------------------------------------
# Candidate-pairs streaming loader
# ---------------------------------------------------------------------------

def load_candidate_pairs(path: str) -> dict[str, list[str]]:
    """Read a candidate_pairs.tsv into {source1_entity_id: [candidate_ids]}.

    Handles both with-header and without-header variants.  The file may be
    very large (16 GB for the full test set) — reads fully into a dict of
    ID lists, which is much smaller than the raw file because we only keep
    the ID strings.

    Parameters
    ----------
    path : str — path to a tab-separated candidate-pairs file.

    Returns
    -------
    dict[str, list[str]]
    """
    result: dict[str, list[str]] = {}
    with open(path, encoding="utf-8", newline="") as fh:
        reader = csv.reader(fh, delimiter="\t")
        for i, row in enumerate(reader):
            if not row:
                continue
            s1_id = row[0].strip()
            if i == 0 and s1_id.lower() == "source1_entity_id":
                continue  # skip genuine header
            raw_ids = row[1].strip() if len(row) > 1 else ""
            candidate_ids = (
                [cid.strip() for cid in raw_ids.split(",") if cid.strip()]
                if raw_ids else []
            )
            result[s1_id] = candidate_ids
    return result


# ---------------------------------------------------------------------------
# Prediction bridge
# ---------------------------------------------------------------------------

def run_predict_matches(
    s1_ids: list[str],
    candidate_map: dict[str, list[str]],
    s1_lookup: dict[str, dict],
    cand_lookup: dict[str, dict],
    model_path: Optional[str],
    threshold: Optional[float],
    threshold_path: Optional[str],
) -> dict[str, list[str]]:
    """Build (s1_row, cand_row) pairs and call the real predict_matches.

    This bridges the ID-centric candidate map from the TSV file to the
    row-dict interface that predict_matches() requires for feature computation.

    When model_path is absent (or model.py failed to import), falls back to
    accepting ALL candidates as matches — so the pipeline still produces
    valid output files for testing.

    Parameters
    ----------
    s1_ids        : ordered list of all S1 entity IDs.
    candidate_map : {s1_id: [candidate_entity_ids]} from load_candidate_pairs.
    s1_lookup     : {entity_id: row_dict} for source 1.
    cand_lookup   : {entity_id: row_dict} for sources 2 + 3 combined.
    model_path    : path to models/matcher.txt, or None for no-model mode.
    threshold     : explicit threshold float, or None to load from threshold_path.
    threshold_path: path to models/threshold.json (used when threshold is None).

    Returns
    -------
    dict[str, list[str]] — {s1_id: [matched_ids]} for ALL s1_ids
    (singletons get an empty list).
    """
    use_real_model = (
        _MODEL_AVAILABLE
        and model_path is not None
        and os.path.isfile(model_path)
    )

    if not use_real_model:
        reason = (
            "model.py import failed"
            if not _MODEL_AVAILABLE
            else f"model file not found: {model_path}"
            if model_path is not None
            else "--model not provided"
        )
        print(
            f"[run_predict_matches] NO-MODEL MODE ({reason}) - "
            f"all blocking candidates accepted as matches.",
            file=sys.stderr,
        )

    # Assemble all candidate pairs as (s1_row, cand_row) tuples
    # keeping track of position so we can map results back to s1_ids.
    all_pairs: list[tuple[dict, dict]] = []
    pair_index: list[tuple[str, str]] = []   # (s1_id, cand_id) in same order

    missing_s1   = 0
    missing_cand = 0

    for s1_id in s1_ids:
        cands = candidate_map.get(s1_id, [])
        s1_row = s1_lookup.get(s1_id)
        if s1_row is None:
            missing_s1 += 1
            continue
        for cid in cands:
            cand_row = cand_lookup.get(cid)
            if cand_row is None:
                missing_cand += 1
                continue
            all_pairs.append((s1_row, cand_row))
            pair_index.append((s1_id, cid))

    if missing_s1:
        print(f"[run_predict_matches] WARNING: {missing_s1} S1 IDs not found in source1 lookup")
    if missing_cand:
        print(f"[run_predict_matches] WARNING: {missing_cand} candidate IDs not found in source2/3 lookup")

    print(f"[run_predict_matches] {len(all_pairs):,} candidate pairs assembled for scoring")

    # ------------------------------------------------------------------ #
    # REAL predict_matches path                                           #
    # ------------------------------------------------------------------ #
    if use_real_model:
        print(f"[run_predict_matches] running real predict_matches (model: {model_path})")

        # Resolve threshold kwargs — only pass what's actually set
        kwargs: dict = {"model_path": model_path}
        if threshold is not None:
            kwargs["threshold"] = threshold
        if threshold_path is not None and os.path.isfile(threshold_path):
            kwargs["threshold_path"] = threshold_path

        predicted: dict[str, list[str]] = _real_predict_matches(all_pairs, **kwargs)

        # Build the full result dict: every S1 id must appear, singletons → []
        results: dict[str, list[str]] = {s1_id: [] for s1_id in s1_ids}
        for s1_id, matched_ids in predicted.items():
            if s1_id in results:
                results[s1_id] = matched_ids

        n_matched = sum(1 for v in results.values() if v)
        print(
            f"[run_predict_matches] real model: "
            f"{n_matched}/{len(s1_ids)} S1 entities have ≥1 match"
        )
        return results

    # ------------------------------------------------------------------ #
    # NO-MODEL fallback: accept every candidate as a match               #
    # ------------------------------------------------------------------ #
    results = {}
    for s1_id in s1_ids:
        cands = candidate_map.get(s1_id, [])
        # Only include candidates that exist in the lookup (same guard as above)
        results[s1_id] = [c for c in cands if c in cand_lookup]

    n_matched = sum(1 for v in results.values() if v)
    print(
        f"[run_predict_matches] no-model fallback: "
        f"{n_matched}/{len(s1_ids)} S1 entities have ≥1 candidate"
    )
    return results


# ---------------------------------------------------------------------------
# Core assembly
# ---------------------------------------------------------------------------

def build_submission(
    candidates_path: str,
    model_path: Optional[str],
    threshold: Optional[float],
    threshold_path: Optional[str],
    source1_path: str,
    source2_path: str,
    source3_path: str,
    output_dir: str,
) -> None:
    """Run the full pipeline and write both output files.

    Steps
    -----
    1. Load source1/2/3 TSVs into entity-id → row-dict lookups.
    2. Load candidate pairs from the blocking TSV.
    3. Call run_predict_matches (real model or no-model fallback).
    4. Write output/matching_results.tsv via write_matching_results().
    5. Write output/candidate_pairs.tsv  via write_candidate_pairs().

    Parameters
    ----------
    candidates_path : path to candidate_pairs.tsv from blocking.
    model_path      : path to models/matcher.txt, or None.
    threshold       : explicit classification threshold, or None.
    threshold_path  : path to models/threshold.json, or None.
    source1_path    : path to test_source1.tsv.
    source2_path    : path to test_source2.tsv.
    source3_path    : path to test_source3.tsv.
    output_dir      : directory to write output files into.
    """
    os.makedirs(output_dir, exist_ok=True)

    # Step 1 — load entity lookups
    print(f"\n[build_submission] loading entity lookups …")
    s1_lookup   = load_entity_lookup(source1_path)
    s2_lookup   = load_entity_lookup(source2_path)
    s3_lookup   = load_entity_lookup(source3_path)
    cand_lookup = {**s2_lookup, **s3_lookup}   # merged S2 + S3 for candidate lookup

    # Ordered list of S1 IDs (every one must appear in output)
    s1_ids = list(s1_lookup.keys())
    print(f"[build_submission] {len(s1_ids)} S1 entities to process")

    # Step 2 — load blocking candidates
    print(f"[build_submission] loading candidate pairs from: {candidates_path}")
    candidate_map = load_candidate_pairs(candidates_path)
    print(f"[build_submission] {len(candidate_map):,} S1 entities in candidate map")

    # Step 3 — predict matches
    matching_results = run_predict_matches(
        s1_ids        = s1_ids,
        candidate_map = candidate_map,
        s1_lookup     = s1_lookup,
        cand_lookup   = cand_lookup,
        model_path    = model_path,
        threshold     = threshold,
        threshold_path= threshold_path,
    )

    # Rebuild full_candidates from the map (pass-through all blocking candidates)
    full_candidates: dict[str, list[str]] = {
        s1_id: candidate_map.get(s1_id, []) for s1_id in s1_ids
    }

    # Step 4 — write output/matching_results.tsv
    matching_out  = os.path.join(output_dir, "matching_results.tsv")
    write_matching_results(matching_results, matching_out)

    # Step 5 — write output/candidate_pairs.tsv
    candidate_out = os.path.join(output_dir, "candidate_pairs.tsv")
    write_candidate_pairs(full_candidates, candidate_out)

    print(f"\n[build_submission] done — outputs written to: {output_dir}")
    print(f"  matching_results.tsv : {matching_out}")
    print(f"  candidate_pairs.tsv  : {candidate_out}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Assemble the final submission files from blocking candidates "
            "and the trained matching model."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples
--------
  # Smoke-test on dummy data (no model needed):
  python src/build_submission.py \\
      --candidate-pairs-path submissions/candidate_pairs_dummy.tsv \\
      --source1 dataset/test/test_source1.tsv \\
      --source2 dataset/test/test_source2.tsv \\
      --source3 dataset/test/test_source3.tsv

  # Full real integration (run once blocking + training complete):
  python src/build_submission.py \\
      --candidate-pairs-path ../repo-blocking/submissions/candidate_pairs.tsv \\
      --model                models/matcher.txt \\
      --threshold-path       models/threshold.json \\
      --source1 ../repo-blocking/dataset/test/test_source1.tsv \\
      --source2 ../repo-blocking/dataset/test/test_source2.tsv \\
      --source3 ../repo-blocking/dataset/test/test_source3.tsv \\
      --output-dir output
""",
    )
    parser.add_argument(
        "--candidate-pairs-path",
        "--candidates",           # legacy alias kept for backward compat
        dest="candidate_pairs_path",
        default="submissions/candidate_pairs.tsv",
        help=(
            "Path to candidate_pairs.tsv from the blocking stage. "
            "Can be the small dummy file or the full-scale one from "
            "repo-blocking - no code changes needed. "
            "(default: %(default)s)"
        ),
    )
    parser.add_argument(
        "--model",
        default=None,
        help=(
            "Path to trained LightGBM model file (models/matcher.txt). "
            "Omit to run in no-model mode (all candidates accepted as matches)."
        ),
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="Explicit classification threshold (overrides threshold.json).",
    )
    parser.add_argument(
        "--threshold-path",
        default=None,
        help=(
            "Path to models/threshold.json. "
            "Used only when --model is set and --threshold is not. "
            "(default: models/threshold.json alongside --model)"
        ),
    )
    parser.add_argument(
        "--source1",
        default="dataset/test/test_source1.tsv",
        help="Path to test_source1.tsv (default: %(default)s).",
    )
    parser.add_argument(
        "--source2",
        default="dataset/test/test_source2.tsv",
        help="Path to test_source2.tsv (default: %(default)s).",
    )
    parser.add_argument(
        "--source3",
        default="dataset/test/test_source3.tsv",
        help="Path to test_source3.tsv (default: %(default)s).",
    )
    parser.add_argument(
        "--output-dir",
        default="output",
        help="Directory to write matching_results.tsv and candidate_pairs.tsv "
             "(default: %(default)s).",
    )
    args = parser.parse_args()

    # Resolve threshold_path: default to same dir as model when not given
    threshold_path = args.threshold_path
    if threshold_path is None and args.model:
        threshold_path = os.path.join(os.path.dirname(args.model), "threshold.json")

    # Pre-flight existence checks
    missing = []
    for label, path in [
        ("--candidate-pairs-path", args.candidate_pairs_path),
        ("--source1",              args.source1),
        ("--source2",              args.source2),
        ("--source3",              args.source3),
    ]:
        if not os.path.isfile(path):
            missing.append(f"  {label}: {path}")
    if missing:
        print("ERROR: required file(s) not found:", file=sys.stderr)
        for m in missing:
            print(m, file=sys.stderr)
        return 1

    if args.model and not os.path.isfile(args.model):
        print(
            f"WARNING: --model not found ({args.model}) - running in "
            f"no-model mode (all candidates treated as matches).",
            file=sys.stderr,
        )

    build_submission(
        candidates_path = args.candidate_pairs_path,
        model_path      = args.model,
        threshold       = args.threshold,
        threshold_path  = threshold_path,
        source1_path    = args.source1,
        source2_path    = args.source2,
        source3_path    = args.source3,
        output_dir      = args.output_dir,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
