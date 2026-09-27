"""Top name tokens only — quick scan."""
import csv, collections, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from blocking import generate_blocking_keys

freq_sw = frozenset(['pediatric'])
name_counts = collections.Counter()

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
                name_counts[k.split('::name:')[1]] += 1

print("=== TOP 40 NAME tokens ===")
for tok, cnt in name_counts.most_common(40):
    print(f"  {cnt:>8,}  {tok!r}")
