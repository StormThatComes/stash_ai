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
from .independent_evidence import Votes


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
    times: Votes = field(default_factory=Votes)
    prototypes: set = field(default_factory=set)
    seen: set = field(default_factory=set)

    def add(self, row, prototype):
        if row['id'] in self.seen:
            return
        self.seen.add(row['id'])
        self.ids.append(row['id'])
        self.times.add(row)
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
    from .memory_size import size, Reservation
    from .query_matrix import QueryMatrix
    families, used, metadata_used = {}, 0, 0
    reservation=Reservation(config.data['execution']['rebuild_memory_bytes']//4)
    for row in stream():
        family, origin = key(row), prototype(row)
        metadata_used += size((row['id'],row['scan'],row['seconds'],family,origin)) + 512
        reservation.set(metadata_used+used)
        exact, unfinished = families.setdefault(family, ({}, {}))
        votes = exact.setdefault(origin, _Votes())
        votes.add(row, origin)
        query = vector(row)
        if len(votes.times) < required(family):
            metadata = {k: row[k] for k in ('id', 'scan', 'seconds')}
            if 'quality' in row.keys():
                metadata['quality'] = row['quality']
            unfinished.setdefault(origin, []).append((metadata, origin, query))
            used += query.nbytes
            if used > grouping['max_reference_bytes']:
                raise ValueError('Frozen query cohorts exceed clustering.max_reference_bytes.')
        else:
            used -= sum(q.nbytes for _, _, q in unfinished.pop(origin, ()))
    reservation.set(metadata_used+used)
    output, pending = [], {}
    for family, (exact, unfinished) in families.items():
        for votes in exact.values():
            if len(votes.times) >= required(family):
                output.append(QuerySupport(family, tuple(votes.ids), tuple(votes.ids),
                                           tuple(sorted(votes.prototypes)), len(votes.times)))
        combined = Votes()
        for votes in exact.values():
            combined.update(votes.times)
        if len(combined) >= required(family):
            missing = {r['id']: (r, p, q) for values in unfinished.values() for r, p, q in values}
            if missing:
                pending[family] = missing
    families.clear()
    seen = set()
    # Index only unresolved families. Fall back to the original streaming path
    # when the bounded index would exceed the reference budget.
    indexed, index_bytes = {}, used
    if pending:
        for row in stream():
            family = key(row)
            if family not in pending:
                continue
            index_bytes += size(dict(row)) + 3*len(row['embedding']) + 1024
            if index_bytes > grouping['max_reference_bytes']:
                indexed = None
                break
            indexed.setdefault(family, []).append(dict(row))
    reservation.set(metadata_used+(index_bytes if indexed is not None else used))
    matrices = ({family: QueryMatrix([(r,prototype(r),vector(r)) for r in records])
                 for family,records in indexed.items()} if indexed is not None else {})
    def witnesses(family, centre):
        if family in matrices:
            return (value[0] for value in matrices[family].matches(centre,boundary))
        return stream()
    # Targets use the same immutable index; removed IDs are filtered below.
    targets_index = ({family: QueryMatrix(list(records.values())) for family,records in pending.items()}
                     if index_bytes + 3*used <= grouping['max_reference_bytes'] else {})
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
            values = (targets_index[family].matches(centre,boundary) if family in targets_index else
                      (v for v in missing.values() if float(np.dot(centre,v[-1])) >= boundary))
            targets = [(r,p) for r,p,_ in values if r['id'] in missing]
            if not targets:
                continue
            votes = _Votes()
            votes.add(row, prototype(row))
            for record, p in targets:
                votes.add(record, p)
            if len(votes.times) < required(family):
                for witness in witnesses(family, centre):
                    if key(witness) != family or witness['id'] in votes.seen:
                        continue
                    if family not in matrices and float(np.dot(centre, vector(witness))) < boundary:
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
