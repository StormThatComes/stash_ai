"""Resolve the middle of a two-person crossing against fixed faces on both sides."""
from itertools import permutations


def crossings(by_cue, roots, decisions, transition, margin):
    claims = {}
    for cue, middle in by_cue.items():
        before, after = by_cue.get(cue-1, ()), by_cue.get(cue+1, ())
        if any(len(group)!=2 for group in (before,middle,after)):
            continue
        owners = [roots.get(r['id']) for r in before]
        if None in owners or len(set(owners))!=2 or any(r.get('face_person') for r in middle):
            continue
        ends = {roots.get(r['id']):r for r in after}
        if set(ends)!=set(owners):
            continue
        candidates = []
        for assignment in permutations(middle):
            paths = []
            for owner, first, row in zip(owners,before,assignment):
                last = ends[owner]
                left, right = transition(first,row), transition(row,last)
                if left is None or right is None:
                    break
                paths.append((row,owner,first,last,(left+right)/2))
            if len(paths)==2:
                candidates.append((sum(item[-1] for item in paths)/2,paths))
        candidates.sort(key=lambda item:-item[0])
        if not candidates or len(candidates)>1 and candidates[0][0]-candidates[1][0]<margin:
            continue
        for row, owner, first, last, score in candidates[0][1]:
            if decisions[row['id']].get('track_blocker',{}).get('reason')=='different_face_root':
                continue
            claims[row['id']] = (owner,first,last,score)
    return claims
