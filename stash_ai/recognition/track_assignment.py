"""One-to-one continuations for physically independent components of a cue pair."""
from collections import defaultdict

from .crowd_context import pair
from .joint_claims import solve


def components(edges):
    neighbours = defaultdict(set)
    for left, right in edges:
        neighbours[left].add(right)
        neighbours[right].add(left)
    while neighbours:
        pending = [next(iter(neighbours))]
        members = set()
        while pending:
            member = pending.pop()
            if member not in members:
                members.add(member)
                pending.extend(neighbours.pop(member))
        yield members


def separated(ids, geometry):
    from itertools import combinations
    for a, b in combinations(ids, 2):
        first, second = geometry[a], geometry[b]
        x,y,r,t = first['bounds']
        u,v,s,w = second['bounds']
        if min(r,s)>max(x,u) and min(t,w)>max(y,v) and pair([first,second]) is None:
            return False
    return True


def joint_links(edges, physical, threshold, margin):
    by_left = defaultdict(dict)
    for (left, right), score in edges.items():
        by_left[left][right] = score
    for members in components(edges):
        left = sorted(members & by_left.keys())
        right = members.difference(left)
        if max(len(left), len(right)) > 4:
            continue
        if not separated(left, physical) or not separated(right, physical):
            continue
        domains = [[(None, threshold-margin), *sorted(by_left[a].items())] for a in left]
        best = solve(domains, lambda i,a,j,b: a[0] is None or a[0]!=b[0], margin)
        if best is not None and all(b is None or score>=threshold for b,score in best):
            for a,(b,score) in zip(left,best):
                if b is not None and score >= threshold:
                    yield a,b,score
