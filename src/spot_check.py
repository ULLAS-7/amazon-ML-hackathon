target = {'S1-965667', 'S1-55344266', 'S1-343815751'}
found = {}
with open('submissions/candidate_pairs.tsv', encoding='utf-8') as f:
    for line in f:
        s1id = line.split('\t')[0]
        if s1id in target:
            parts = line.rstrip('\n').split('\t')
            cands = parts[1] if len(parts) > 1 else ''
            cand_list = cands.split(',') if cands else []
            found[s1id] = cand_list
        if len(found) == len(target):
            break

for eid in target:
    cl = found.get(eid)
    if cl is None:
        print(f"{eid}: NOT IN FILE")
    else:
        print(f"{eid}: {len(cl)} candidates, first 3: {cl[:3]}")
