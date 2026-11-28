"""One-to-one continuations for physically independent components of a cue pair."""
from collections import defaultdict

from .crowd_context import pair



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
        if not separated(left, physical) or not separated(right, physical):
            continue
        import numpy as np
        from scipy.optimize import linear_sum_assignment
        targets = sorted(right)
        columns = {target:i for i,target in enumerate(targets)}
        weights = np.full((len(left), len(targets)+len(left)), -np.inf)
        weights[:,len(targets):] = threshold-margin
        for i,source in enumerate(left):
            for target,score in by_left[source].items():
                weights[i,columns[target]] = score
        rows, assigned = linear_sum_assignment(weights, maximize=True)
        best = weights[rows,assigned].sum()
        for i,j in zip(rows,assigned):
            if j>=len(targets) or weights[i,j]<threshold:
                continue
            score = weights[i,j]
            weights[i,j] = -np.inf
            a,b = linear_sum_assignment(weights, maximize=True)
            alternative = weights[a,b].sum()
            weights[i,j] = score
            if best-alternative+1e-12>=margin and best>alternative:
                yield left[i],targets[j],float(score)
