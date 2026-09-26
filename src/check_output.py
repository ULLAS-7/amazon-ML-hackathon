"""Quick format check on candidate_pairs output files."""
import sys

files = [
    ("TRAIN", "submissions/candidate_pairs.tsv"),
    ("TEST",  "submissions/test_candidate_pairs.tsv"),
]

for label, path in files:
    total = s1ok = empty = 0
    s23ok = True
    bad_prefix = []
    bad_col = 0
    with open(path, encoding="utf-8") as f:
        for line in f:
            total += 1
            p = line.rstrip("\n").split("\t")
            if len(p) != 2:
                bad_col += 1
                continue
            s1id, cs = p
            if s1id.startswith("S1-"):
                s1ok += 1
            else:
                bad_prefix.append(s1id[:20])
            if not cs:
                empty += 1
            else:
                for c in cs.split(","):
                    if not (c.startswith("S2-") or c.startswith("S3-")):
                        s23ok = False
                        print(f"  {label} BAD CAND: {c!r} on row {total}")
                        break
    print(
        f"{label}: {total:,} rows | S1-prefix OK={s1ok:,} | "
        f"empty-cand={empty:,} | S2/S3-only={s23ok} | "
        f"bad-col={bad_col} | bad-s1-prefix={len(bad_prefix)}"
    )
    if bad_prefix:
        print(f"  First bad prefix examples: {bad_prefix[:5]}")
