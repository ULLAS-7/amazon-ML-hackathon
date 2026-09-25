import csv
import os
import pandas as pd

# NOTE: 'entity_id' and 'matches' are placeholder column names.
# Adjust these once the actual dataset columns are visible.
ENTITY_ID_COL = 'entity_id'
MATCHES_COL = 'matches'

# ---------------------------------------------------------------------------
# Output writers
# ---------------------------------------------------------------------------

def write_matching_results(results: dict, output_path: str) -> None:
    """Write entity-matching results to a tab-separated TSV file.

    Parameters
    ----------
    results     : dict mapping source1_entity_id (str) to a list/set of
                  matched entity IDs from Source 2 / Source 3 (str).
                  An empty list/set means the S1 entity is a singleton.
    output_path : str — destination file path (created if absent; parent
                  directory must already exist).

    Output format
    -------------
    Two tab-separated columns::

        source1_entity_id\\tmatched_entity_ids
        S1-00001\\tS2-00047,S2-00193,S3-00812
        S1-00002\\tS3-00004
        S1-00003\\t

    - One row per Source 1 entity (singletons get an empty second column).
    - IDs within ``matched_entity_ids`` are comma-separated, no spaces.
    - File is UTF-8 with Unix line endings.
    """
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)
    with open(output_path, "w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh, delimiter="\t", lineterminator="\n")
        writer.writerow(["source1_entity_id", "matched_entity_ids"])
        for s1_id, matched_ids in sorted(results.items()):
            # Deduplicate while preserving a deterministic order
            seen: dict = {}
            for mid in matched_ids:
                seen[mid] = None
            writer.writerow([s1_id, ",".join(seen.keys())])
    print(f"[write_matching_results] wrote {len(results)} rows → {output_path}")


def write_candidate_pairs(candidates: dict, output_path: str) -> None:
    """Write blocking candidate pairs to a tab-separated TSV file.

    Parameters
    ----------
    candidates  : dict mapping source1_entity_id (str) to a list/set of
                  candidate entity IDs from Source 2 / Source 3 (str).
                  An empty list/set means no blocking candidates were found.
    output_path : str — destination file path (created if absent; parent
                  directory must already exist).

    Output format
    -------------
    Two tab-separated columns::

        source1_entity_id\\tcandidate_entity_ids
        S1-00001\\tS2-00047,S2-00193,S3-00812,S3-00999
        S1-00002\\tS3-00004
        S1-00003\\t

    - One row per Source 1 entity (no candidates → empty second column).
    - IDs within ``candidate_entity_ids`` are comma-separated, no spaces.
    - Every ID in ``matching_results.tsv`` should appear in the corresponding
      candidate list (the validator warns when this is violated).
    - File is UTF-8 with Unix line endings.
    """
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)
    with open(output_path, "w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh, delimiter="\t", lineterminator="\n")
        writer.writerow(["source1_entity_id", "candidate_entity_ids"])
        for s1_id, candidate_ids in sorted(candidates.items()):
            seen: dict = {}
            for cid in candidate_ids:
                seen[cid] = None
            writer.writerow([s1_id, ",".join(seen.keys())])
    print(f"[write_candidate_pairs] wrote {len(candidates)} rows → {output_path}")


def load_tsv(filepath):
    """Read a tab-separated file into a pandas DataFrame."""
    df = pd.read_csv(filepath, sep='\t')
    print(f"Loaded '{filepath}': shape={df.shape}, columns={list(df.columns)}")
    return df


def load_all_sources(data_dir):
    """
    Load source1.tsv, source2.tsv, source3.tsv from data_dir.
    Returns a dict {'source1': df1, 'source2': df2, 'source3': df3}.
    Prints a clear error for any missing file instead of crashing.
    """
    sources = {}
    for name in ['source1', 'source2', 'source3']:
        filepath = os.path.join(data_dir, f'{name}.tsv')
        if not os.path.exists(filepath):
            print(f"ERROR: Missing file '{filepath}'")
            continue
        sources[name] = load_tsv(filepath)
    return sources


def load_ground_truth(filepath):
    """
    Load the labels file and print a summary of matched vs singleton entities.
    Assumes the matches column contains comma-separated IDs;
    empty string or NaN means singleton.
    """
    df = load_tsv(filepath)

    def is_singleton(val):
        if pd.isna(val):
            return True
        return str(val).strip() == ''

    singletons = df[MATCHES_COL].apply(is_singleton).sum()
    matched = len(df) - singletons

    print(f"Ground truth summary — total: {len(df)}, "
          f"with at least one match: {matched}, "
          f"singletons: {singletons}")
    return df


def validate_submission(filepath):
    """
    Validate a submission file. Checks:
    - File is tab-separated
    - Has exactly the expected columns (ENTITY_ID_COL and MATCHES_COL)
    - No duplicate entity IDs
    Prints a summary: total rows, empty match lists, rows with at least one match.
    """
    df = pd.read_csv(filepath, sep='\t')

    expected_cols = {ENTITY_ID_COL, MATCHES_COL}
    actual_cols = set(df.columns)
    if actual_cols != expected_cols:
        print(f"ERROR: Expected columns {expected_cols}, got {actual_cols}")
        return

    duplicates = df[ENTITY_ID_COL].duplicated().sum()
    if duplicates > 0:
        print(f"ERROR: Found {duplicates} duplicate entity ID(s) in '{ENTITY_ID_COL}'")
        return

    def is_empty(val):
        if pd.isna(val):
            return True
        return str(val).strip() == ''

    empty_matches = df[MATCHES_COL].apply(is_empty).sum()
    has_matches = len(df) - empty_matches

    print(f"Submission summary — total rows: {len(df)}, "
          f"empty match lists: {empty_matches}, "
          f"with at least one match: {has_matches}")
