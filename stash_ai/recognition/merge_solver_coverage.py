"""Assignment coverage on the fixed replay population, including non-goal rows."""
from collections import Counter


def measure(baseline, predicted, target_ids, outside, groups):
    """Reuse outside counts and sparse deltas; never materialize the full corpus.

    Goal addresses may overlap or be absent from the selected scans. Count each
    present address once, with the baseline population as the fixed denominator.
    """
    assigned = outside['assigned']
    gained, lost = outside['coverage_gained'], outside['coverage_lost']
    for oid in target_ids:
        if oid not in baseline:
            continue
        old, new = baseline[oid], predicted.get(oid)
        assigned += new is not None
        gained += old is None and new is not None
        lost += old is not None and new is None
    destinations = Counter(d for group in groups for d in group['resulting_identities'])
    from .partition_delta import Predictions
    from .solver_slot_index import Slots
    rows = (predicted.changes.items() if isinstance(predicted, Predictions) else
            () if isinstance(predicted, Slots) else predicted.items())
    # Previously unassigned rows are eligible recovery, not established outsiders.
    # Explicit new-outsider limits still count these attachments without exception.
    recovered_target_attachments = sum(destinations.get(person, 0) for oid, person in rows
        if person is not None and oid not in target_ids and baseline.get(oid) is None)
    return dict(total=len(baseline), assigned=assigned, unassigned=len(baseline)-assigned,
                gained=gained, lost=lost, recovered_target_attachments=recovered_target_attachments)


def summary(coverage):
    return (f"Replay assignments: assigned={coverage['assigned']}/{coverage['total']} "
            f"unassigned={coverage['unassigned']} gained={coverage['gained']} lost={coverage['lost']}")
