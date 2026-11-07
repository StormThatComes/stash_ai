"""Bounded complete subgroups proposed from one immutable weighted graph.

Only score prefixes with an outsider margin are candidates. No output becomes
an input, no maximal-clique search, and no tie-breaking creates ownership.
"""


class Edges:
    def __init__(self, budget):
        self.limit = max(0, budget // 256)
        self.scores = {}
        self.available = True

    def add(self, left, right, score):
        if not self.available:
            return
        if len(self.scores) >= self.limit:
            self.scores.clear()
            self.available = False
            return
        self.scores[tuple(sorted((left, right)))] = score

    def score(self, left, right):
        return self.scores.get(tuple(sorted((left, right))), float('-inf'))

    def proposals(self, people, threshold, margin):
        if not self.available or len(people) > 128:
            return ()  # Keep the existing conservative component result.
        found = set()
        for person in sorted(people):
            neighbours = sorted((p for p in people if p != person and self.score(p, person) >= threshold),
                                key=lambda p: (-self.score(p, person), p))
            for size in range(1, len(neighbours) + 1):
                if size < len(neighbours) and self.score(person, neighbours[size-1]) - self.score(person, neighbours[size]) < margin:
                    continue
                members = tuple(sorted([person] + neighbours[:size]))
                if len(members) == len(people) or members in found:
                    continue
                if all(self.score(a, b) >= threshold for i, a in enumerate(members) for b in members[i+1:]):
                    found.add(members)
        return tuple(sorted(found, key=lambda p: (-len(p), p)))

    def floors(self, people):
        return {p: min(self.score(p, q) for q in people if p != q) for p in people}


def nonoverlapping(proposals):
    """Incompatible frozen proposals veto each other instead of greedy wins."""
    maximal = [p for p in proposals if not any(set(p) < set(q) for q in proposals)]
    return [p for p in maximal if not any(set(p) & set(q) and p != q for q in maximal)]


def large_subgroups(people, edges, pairs, threshold, margin, budget, maximum=8):
    """Search short score prefixes on the existing sparse graph. All adjacent
    outsiders still compete, including neighbours beyond the nomination limit.
    """
    from heapq import nsmallest
    def score(a,b):
        return pairs[tuple(sorted((a,b)))]['score']

    proposals = set()
    used = 0
    limited = False
    for owner in people:
        best = nsmallest(maximum, edges[owner], key=lambda p:(-score(owner,p),p))
        for size in range(1,min(len(best),maximum-1)+1):
            if size<len(best) and score(owner,best[size-1])-score(owner,best[size]) < margin:
                continue
            group = tuple(sorted([owner,*best[:size]]))
            if group in proposals:
                continue
            members = set(group)
            for p in group:
                internal = members-{p}
                if not internal <= edges[p]:
                    break
                weakest = min(score(p,q) for q in internal)
                if weakest<threshold or any(weakest-score(p,q)<margin for q in edges[p] if q not in members):
                    break
            else:
                cost = 512+len(group)*192
                if used+cost > budget:
                    # An undiscovered conflicting proposal must not lose by iteration order.
                    return (), 'subgroup_budget'
                proposals.add(group)
                used += cost
        if len(edges[owner])>=maximum:
            limited = True
    # Exact separated sets can be nested. Incomparable overlapping sets veto
    # each other. Index memberships to avoid a library-wide Cartesian product.
    by_owner = {}
    maximal = []
    for group in sorted(proposals,key=lambda g:(-len(g),g)):
        overlap = set().union(*(by_owner.get(p,()) for p in group))
        if any(set(group)<=set(maximal[i]) for i in overlap):
            continue
        index = len(maximal)
        maximal.append(group)
        for p in group:
            by_owner.setdefault(p,set()).add(index)
    return [g for g in maximal if all(len(by_owner[p])==1 for p in g)], (
        'subgroup_size_limit' if limited else 'incomplete_body_component')
