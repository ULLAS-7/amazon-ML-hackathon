"""Fast row count + head/tail check for a large TSV."""
import sys
path = sys.argv[1]
total = 0
first3 = []
last3 = collections.deque(maxlen=3)
import collections
with open(path, encoding="utf-8") as f:
    for line in f:
        total += 1
        parts = line.rstrip("\n").split("\t")
        n = len(parts[1].split(",")) if len(parts) > 1 and parts[1] else 0
        rec = f"{parts[0]}  ncands={n}"
        if total <= 3:
            first3.append(rec)
        last3.append(rec)
print(f"Total rows: {total:,}")
print("First 3:", first3)
print("Last  3:", list(last3))
