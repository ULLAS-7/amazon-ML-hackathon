"""
find_garbage_keys.py — List the top oversized name/nrev/housenum buckets in the
full S2 index so we can identify all garbage tokens to explicitly drop.
"""
import csv, collections, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from blocking import generate_blocking_keys

freq_sw = frozenset(['pediatric'])

name_counts   = collections.Counter()
nrev_counts   = collections.Counter()
hnum_counts   = collections.Counter()

with open('dataset/train/train_source2.tsv', encoding='utf-8', errors='replace', newline='') as fh:
    reader = csv.DictReader(fh, delimiter='\t')
    for row in reader:
        keys = generate_blocking_keys(
            row.get('business_name','') or '',
            row.get('business_address','') or '',
            row.get('country','') or '',
            freq_sw,
        )
        for k in keys:
            if '::name:' in k:
                tok = k.split('::name:')[1]
                name_counts[tok] += 1
            elif '::nrev:' in k:
                tok = k.split('::nrev:')[1]
                nrev_counts[tok] += 1
            elif '::housenum:' in k:
                tok = k.split('::housenum:')[1]
                hnum_counts[tok] += 1

print("=== TOP 30 NAME tokens (bucket size) ===")
for tok, cnt in name_counts.most_common(30):
    print(f"  {cnt:>8,}  {tok!r}")

print("\n=== TOP 30 NREV tokens (bucket size) ===")
for tok, cnt in nrev_counts.most_common(30):
    print(f"  {cnt:>8,}  {tok!r}")

print("\n=== TOP 30 HOUSENUM tokens (bucket size) ===")
for tok, cnt in hnum_counts.most_common(30):
    print(f"  {cnt:>8,}  {tok!r}")
