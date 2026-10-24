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
