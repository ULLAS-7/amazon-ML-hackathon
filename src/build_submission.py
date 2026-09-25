"""
build_submission.py — End-to-end submission assembly script.

Takes a candidate_pairs.tsv (from the blocking stage) and a trained model
path, runs match prediction, then writes both output files required by the
ML Challenge 2026 submission format:

    output/matching_results.tsv   — final matches (scored on leaderboard)
    output/candidate_pairs.tsv    — blocking candidates fed to the model

Usage
-----
    python src/build_submission.py \\
        --candidates submissions/candidate_pairs.tsv \\
        --model      models/lgbm_model.pkl \\
        --source1    dataset/test/test_source1.tsv \\
        --output-dir output

The script is deliberately self-contained: it only imports from the stdlib
and from src/utils.py, so it works before any other src/ module is finished.

Integration note
----------------
The dummy_predict_matches() function below is a STUB that must be replaced
once Agent 2's model-baseline branch is merged.  See the TODO block inside
the function for the exact replacement contract.
"""

from __future__ import annotations

import argparse
import csv
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


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def load_candidate_pairs(path: str) -> dict[str, list[str]]:
    """Read a candidate_pairs.tsv into {source1_entity_id: [candidate_ids]}.

    Handles both the with-header and without-header variants.  If the first
    line looks like a real header (starts with 'source1_entity_id') it is
    skipped; otherwise it is treated as data.

    Parameters
    ----------
    path : str — path to a tab-separated candidate-pairs file.

    Returns
    -------
    dict[str, list[str]] — one entry per S1 entity.  Empty candidate list
    when the second column is absent or blank.
    """
    result: dict[str, list[str]] = {}
    with open(path, encoding="utf-8", newline="") as fh:
        reader = csv.reader(fh, delimiter="\t")
        for i, row in enumerate(reader):
            if not row:
                continue
            s1_id = row[0].strip()
            # Skip a genuine header row
            if i == 0 and s1_id.lower() == "source1_entity_id":
                continue
            raw_ids = row[1].strip() if len(row) > 1 else ""
            candidate_ids = [cid.strip() for cid in raw_ids.split(",") if cid.strip()] if raw_ids else []
            result[s1_id] = candidate_ids
    return result


def load_source1_ids(path: str) -> list[str]:
    """Return all entity_ids from a test_source1.tsv in file order.

    Every S1 entity must appear in the final output, even if the blocking
    stage produced no candidates for it (singleton → empty match list).

    Parameters
    ----------
    path : str — path to a tab-separated source-1 file with a header row.

    Returns
    -------
    list[str]
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
# Dummy / stub prediction
# ---------------------------------------------------------------------------

def dummy_predict_matches(
    s1_id: str,
    candidate_ids: list[str],
    model_path: Optional[str],
) -> list[str]:
    """Stub match predictor — returns a deterministic dummy subset of candidates.

    !! TODO: replace this entire function with the real predict_matches() from
    !!       the model-baseline branch once Agent 2's work is merged.
    !!
    !! Contract that the real function must satisfy:
    !!   - Signature: predict_matches(s1_id, candidate_ids, model_path) -> list[str]
    !!   - Load the trained model from `model_path` (LightGBM .pkl or similar).
    !!   - For each candidate_id in candidate_ids, compute the feature vector
    !!     (name + address similarity features from src/features.py).
    !!   - Run model.predict_proba() and apply the tuned classification threshold.
    !!   - Return only the candidate IDs whose predicted probability >= threshold.
    !!   - Return an empty list when candidate_ids is empty (singleton entity).
    !!
    !! The stub below mimics the same input/output shape so the rest of this
    !! script can be tested end-to-end today.

    Dummy logic
    -----------
    - Returns every *other* candidate starting from index 0 (simulates a
      model that accepts ~50 % of candidates).
    - When fewer than 2 candidates exist, returns all of them.
    - model_path is accepted but intentionally unused.
    """
    if not candidate_ids:
        return []
    if len(candidate_ids) < 2:
        return list(candidate_ids)
    # Accept every other candidate: indices 0, 2, 4, ...
    return candidate_ids[::2]


# ---------------------------------------------------------------------------
# Core assembly logic
# ---------------------------------------------------------------------------

def build_submission(
    candidates_path: str,
    model_path: Optional[str],
    source1_path: str,
    output_dir: str,
) -> None:
    """Run the full assembly pipeline and write both output files.

    Steps
    -----
    1. Load all Source 1 entity IDs (ensures singletons are included).
    2. Load candidate pairs from the blocking stage.
    3. For each S1 entity, run predict_matches (currently the dummy stub).
    4. Write output/matching_results.tsv via write_matching_results().
    5. Write output/candidate_pairs.tsv  via write_candidate_pairs().

    Parameters
    ----------
    candidates_path : str  — path to candidate_pairs.tsv from blocking.
    model_path      : str | None — path to a trained model file.
                      Pass None when running in stub mode.
    source1_path    : str  — path to test_source1.tsv (all S1 IDs).
    output_dir      : str  — directory to write output files into.
    """
    os.makedirs(output_dir, exist_ok=True)

    # Step 1 — collect every S1 entity that must appear in the output
    print(f"[build_submission] loading S1 entities from: {source1_path}")
    s1_ids = load_source1_ids(source1_path)
    print(f"[build_submission] {len(s1_ids)} S1 entities found")

    # Step 2 — load blocking candidates
    print(f"[build_submission] loading candidate pairs from: {candidates_path}")
    candidate_map = load_candidate_pairs(candidates_path)
    print(f"[build_submission] {len(candidate_map)} S1 entities in candidate map")

    # Step 3 — predict matches for each S1 entity
    matching_results: dict[str, list[str]] = {}
    full_candidates:  dict[str, list[str]] = {}

    for s1_id in s1_ids:
        # Entities not in the blocking output are singletons (no candidates)
        cands = candidate_map.get(s1_id, [])

        # TODO: swap dummy_predict_matches for the real predict_matches once
        #       the model-baseline branch is merged.
        matches = dummy_predict_matches(s1_id, cands, model_path)

        matching_results[s1_id] = matches
        full_candidates[s1_id]  = cands  # pass through all candidates

    # Step 4 — write output/matching_results.tsv
    matching_out = os.path.join(output_dir, "matching_results.tsv")
    write_matching_results(matching_results, matching_out)

    # Step 5 — write output/candidate_pairs.tsv
    candidate_out = os.path.join(output_dir, "candidate_pairs.tsv")
    write_candidate_pairs(full_candidates, candidate_out)

    print(f"[build_submission] done — outputs in: {output_dir}")
    print(f"  matching_results.tsv : {matching_out}")
    print(f"  candidate_pairs.tsv  : {candidate_out}")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Assemble the final submission files from blocking candidates "
            "and a trained matching model."
        )
    )
    parser.add_argument(
        "--candidates",
        default="submissions/candidate_pairs.tsv",
        help="Path to candidate_pairs.tsv produced by the blocking stage "
             "(default: %(default)s).",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Path to trained model file (.pkl or similar). "
             "Optional — omit to run in stub/dummy mode.",
    )
    parser.add_argument(
        "--source1",
        default="dataset/test/test_source1.tsv",
        help="Path to test_source1.tsv so every S1 entity appears in output "
             "(default: %(default)s).",
    )
    parser.add_argument(
        "--output-dir",
        default="output",
        help="Directory to write matching_results.tsv and candidate_pairs.tsv "
             "(default: %(default)s).",
    )
    args = parser.parse_args()

    # Basic existence checks before kicking off the pipeline
    for label, path in [("--candidates", args.candidates), ("--source1", args.source1)]:
        if not os.path.isfile(path):
            print(f"ERROR: {label} file not found: {path}", file=sys.stderr)
            return 1

    if args.model and not os.path.isfile(args.model):
        print(f"WARNING: --model file not found: {args.model} — running in stub mode.",
              file=sys.stderr)

    build_submission(
        candidates_path=args.candidates,
        model_path=args.model,
        source1_path=args.source1,
        output_dir=args.output_dir,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
