"""Independent query votes across frozen views of one identity.

Existing per-prototype support stays valid. Additional support may cross prototype
boundaries only inside a tight, frozen query cohort. Its immutable first witness
defines half the angular cohesion radius, preventing single-link chaining. Raw
reference/training cues must never be passed as query votes.
"""
from dataclasses import dataclass, field
import math

from .clustering import vector


@dataclass(frozen=True)
class QuerySupport:
    family: object
    observations: tuple
    witnesses: tuple
    prototypes: tuple
    independent_support: int


@dataclass
class _Votes:
    ids: list = field(default_factory=list)
    times: set = field(default_factory=set)
    prototypes: set = field(default_factory=set)
    pivot: object = None

    def add(self, row, prototype):
        self.ids.append(row['id'])
        self.times.add((row['scan'], row['seconds']))
        self.prototypes.add(prototype)


def cohorts(rows, config, *, key, prototype, minimum=None):
    """Yield disjoint accepted IDs, with all original query voters in proof.

    Callers stream in canonical embedding order and perform their own per-query
    identity, ambiguity and co-presence checks before entering this function.
    Only one immutable embedding per discovered cohort is retained.
    """
    import numpy as np
    grouping = config.data['clustering']
    required = minimum or (lambda family: grouping['view_attach_min_support'])
    boundary = math.sqrt((1 + grouping['view_cohesion_threshold']) / 2)
    families, used = {}, 0
    for row in rows:
        family, source = key(row), prototype(row)
        exact, covered = families.setdefault(family, ({}, []))
        exact.setdefault(source, _Votes()).add(row, source)
        query = vector(row)
        slot = next((v for v in covered if float(np.dot(v.pivot, query)) >= boundary), None)
        if slot is None:
            used += query.nbytes
            if used > grouping['max_reference_bytes']:
                raise ValueError('Frozen query cohorts exceed clustering.max_reference_bytes.')
            slot = _Votes(pivot=query); covered.append(slot)
        slot.add(row, source)
    for family, (exact, covered) in families.items():
        accepted = set()
        # A supported tight cohort can pool different frozen views. Remaining
        # queries retain their original per-prototype support semantics.
        for votes in covered + list(exact.values()):
            if len(votes.times) < required(family):
                continue
            fresh = tuple(i for i in votes.ids if i not in accepted)
            if fresh:
                accepted.update(fresh)
                yield QuerySupport(family, fresh, tuple(votes.ids),
                                   tuple(sorted(votes.prototypes)), len(votes.times))


def eligible(rows, config, *, key, prototype, minimum=None):
    return {i for cohort in cohorts(rows, config, key=key, prototype=prototype, minimum=minimum)
            for i in cohort.observations}
