"""Independent query votes across frozen views of one identity.

Existing per-prototype support stays valid. Additional support may cross prototype
boundaries inside overlapping hypotheses with immutable observed centres. Each
hypothesis uses half the angular cohesion radius; neither its accepted output nor
another hypothesis can extend that radius. Raw training cues are never votes.
"""
from dataclasses import dataclass, field
import hashlib
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
    seen: set = field(default_factory=set)

    def add(self, row, prototype):
        if row['id'] in self.seen:
            return
        self.seen.add(row['id'])
        self.ids.append(row['id'])
        self.times.add((row['scan'], row['seconds']))
        self.prototypes.add(prototype)


def cohorts(rows, config, *, key, prototype, minimum=None):
    """Yield disjoint accepted IDs, with all original query voters in proof.

    Rows must be replayable, or a factory returning a fresh canonical stream.
    Callers perform per-query identity, ambiguity and co-presence checks first.
    Exact supported prototypes take one pass without retaining their vectors.
    Only unsupported query vectors are kept; other witnesses are streamed when
    needed. Freeze every result before yielding, since callers may then mutate
    the candidate rows used by the factory.
    """
    import numpy as np
    grouping = config.data['clustering']
    required = minimum or (lambda family: grouping['view_attach_min_support'])
    boundary = math.sqrt((1 + grouping['view_cohesion_threshold']) / 2)
    if callable(rows):
        stream = rows
    elif iter(rows) is not rows:
        stream = lambda: iter(rows)
    else:
        raise ValueError('Frozen query support requires replayable rows or a row factory.')
    families, used = {}, 0
    for row in stream():
        family, origin = key(row), prototype(row)
        exact, unfinished = families.setdefault(family, ({}, {}))
        votes = exact.setdefault(origin, _Votes())
        votes.add(row, origin)
        query = vector(row)
        if len(votes.times) < required(family):
            metadata = {k: row[k] for k in ('id', 'scan', 'seconds')}
            unfinished.setdefault(origin, []).append((metadata, origin, query))
            used += query.nbytes
            if used > grouping['max_reference_bytes']:
                raise ValueError('Frozen query cohorts exceed clustering.max_reference_bytes.')
        else:
            used -= sum(q.nbytes for _, _, q in unfinished.pop(origin, ()))
    output, pending = [], {}
    for family, (exact, unfinished) in families.items():
        for votes in exact.values():
            if len(votes.times) >= required(family):
                output.append(QuerySupport(family, tuple(votes.ids), tuple(votes.ids),
                                           tuple(sorted(votes.prototypes)), len(votes.times)))
        if len(set().union(*(v.times for v in exact.values()))) >= required(family):
            missing = {r['id']: (r, p, q) for values in unfinished.values() for r, p, q in values}
            if missing:
                pending[family] = missing
    families.clear()
    seen = set()
    if pending:
        for row in stream():
            family = key(row)
            missing = pending.get(family)
            if not missing:
                continue
            signature = family, hashlib.sha256(row['embedding']).digest()
            if signature in seen:
                continue
            seen.add(signature)
            centre = vector(row)
            targets = [(r, p) for r, p, q in missing.values() if float(np.dot(centre, q)) >= boundary]
            if not targets:
                continue
            votes = _Votes()
            votes.add(row, prototype(row))
            for record, p in targets:
                votes.add(record, p)
            if len(votes.times) < required(family):
                for witness in stream():
                    if key(witness) != family or witness['id'] in votes.seen:
                        continue
                    if float(np.dot(centre, vector(witness))) < boundary:
                        continue
                    votes.add(witness, prototype(witness))
                    if len(votes.times) >= required(family):
                        break
            if len(votes.times) >= required(family):
                identifiers = tuple(r['id'] for r, _ in targets)
                output.append(QuerySupport(family, identifiers, tuple(votes.ids),
                                           tuple(sorted(votes.prototypes)), len(votes.times)))
                for identifier in identifiers:
                    missing.pop(identifier)
                if not missing:
                    pending.pop(family)
                if not pending:
                    break
    yield from output


def eligible(rows, config, *, key, prototype, minimum=None):
    return {i for cohort in cohorts(rows, config, key=key, prototype=prototype, minimum=minimum)
            for i in cohort.observations}
