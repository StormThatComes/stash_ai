"""Exact partition metrics for a fixed baseline and sparse assignment changes."""
from collections import Counter
from collections.abc import Mapping


class Predictions(Mapping):
    def __init__(self, baseline, changes):
        self.baseline, self.changes = baseline, changes
        self.changed_owners = None
        self.outsider_delta = None
        if any(key not in baseline for key in changes):
            raise ValueError('Invalid baseline prediction delta.')

    def __getitem__(self, key):
        return self.changes[key] if key in self.changes else self.baseline[key]

    def __len__(self):
        return len(self.baseline)

    def __iter__(self):
        return iter(self.baseline)

    def outsider_counts(self, owners):
        if self.outsider_delta is None:
            self.outsider_delta = self.baseline.outsider_changes(self.changes)
        counts = self.baseline.outsider_counts(owners)
        for owner in owners:
            counts.update(self.outsider_delta.get(owner, {}))
        return counts

    def for_owners(self, owners):
        if hasattr(self.baseline, 'for_owners'):
            rows = self.baseline.for_owners(owners)
        else:
            rows = ((key,person) for key,person in self.baseline.items() if person in owners)
        yield from ((key,person) for key,person in rows if key not in self.changes)
        if self.changed_owners is None:
            from collections import defaultdict
            self.changed_owners = defaultdict(list)
            for key,person in self.changes.items():
                self.changed_owners[person].append(key)
        for person in owners:
            yield from ((key,person) for key in self.changed_owners.get(person,()))

    def items(self):
        return ((key, self.changes.get(key, value)) for key, value in self.baseline.items())


def pairs(n):
    return n * (n - 1) // 2


def pair_delta(counts, changes):
    return sum(pairs(counts[key] + n) - pairs(counts[key]) for key, n in changes.items())


class PartitionImpact:
    def __init__(self, before, baseline, excluded=frozenset()):
        self.before, self.baseline, self.excluded = before, baseline, excluded
        self.cells, self.left, self.right, self.population = (Counter() for _ in range(4))
        self.common = self.gained = self.lost = self.assigned = 0
        from .solver_slot_index import Slots
        indexed = isinstance(before, Slots) and isinstance(baseline, Slots) and before.db is baseline.db
        rows = baseline.db.execute(f'SELECT id,{before.column},{baseline.column} FROM slots') if indexed else (
            (key,before.get(key),new) for key,new in baseline.items())
        for key, old, new in rows:
            if key in excluded:
                continue
            if new is not None:
                self.assigned += 1
                self.population[new] += 1
            if not indexed and key not in before:
                continue
            self.common += 1
            self.gained += old is None and new is not None
            self.lost += old is not None and new is None
            if old is not None and new is not None:
                self.left[old] += 1
                self.right[new] += 1
                self.cells[old, new] += 1
        shared = sum(map(pairs, self.cells.values()))
        self.joined = sum(map(pairs, self.right.values())) - shared
        self.split = sum(map(pairs, self.left.values())) - shared

    def measure(self, delta):
        cells, left, right, population = (Counter() for _ in range(4))
        assigned, gained, lost = self.assigned, self.gained, self.lost
        for key, new in delta.items():
            if key in self.excluded:
                continue
            previous = self.baseline[key]
            if new == previous:
                continue
            for person, sign in ((previous, -1), (new, 1)):
                if person is not None:
                    population[person] += sign
                    assigned += sign
                if key in self.before:
                    old = self.before[key]
                    gained += sign * (old is None and person is not None)
                    lost += sign * (old is not None and person is None)
                    if old is not None and person is not None:
                        left[old] += sign
                        right[person] += sign
                        cells[old, person] += sign
        identities = len(self.population) + sum(
            int(self.population[k] + change > 0) - int(self.population[k] > 0)
            for k, change in population.items())
        shared = pair_delta(self.cells, cells)
        return dict(common_observations=self.common,
                    newly_joined_pairs=self.joined + pair_delta(self.right, right) - shared,
                    split_pairs=self.split + pair_delta(self.left, left) - shared,
                    coverage_gained=gained, coverage_lost=lost, assigned=assigned, identities=identities)
