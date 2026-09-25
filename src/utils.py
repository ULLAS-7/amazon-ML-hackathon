import os
import pandas as pd

# NOTE: 'entity_id' and 'matches' are placeholder column names.
# Adjust these once the actual dataset columns are visible.
ENTITY_ID_COL = 'entity_id'
MATCHES_COL = 'matches'


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
