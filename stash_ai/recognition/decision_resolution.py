"""Resolve simultaneous identity claims before independent support is counted."""
from itertools import groupby

from .clustering import vector
from .database import encoded
from .replay_addresses import address


def conflicting_claims(rows, threshold):
    """Stream claims sorted by (person, scan, cue); exact duplicates share a vote.

    All pairs must satisfy the stronger reflection threshold. Different views
    of one identity cannot hide each other from the per-frame conflict check.
    Scratch storage is bounded by detections in one cue.
    """
    import numpy as np
    for _, claims in groupby(rows, lambda r: (r['person'], r['scan'], r['cue'])):
        claims = list(claims)
        samples = []
        signatures = set()
        conflict = False
        for row in claims:
            signature = (row['geometry'], row['embedding'])
            if signature in signatures:
                continue
            signatures.add(signature)
            sample = vector(row)
            if any(float(np.dot(sample, other)) < threshold for other in samples):
                conflict = True
            samples.append(sample)
        if conflict:
            yield from claims


def resolve_pending(db, policy, config, *, scans=None):
    """Accepted claims participate, but only this stage's pending claims change."""
    condition = '' if scans is None else 'AND c.scan IN (SELECT value FROM json_each(?)) '
    parameters = (policy,) if scans is None else (policy, encoded(list(scans)))
    rows = db.execute('SELECT o.id,o.scan,o.cue,o.geometry,o.dimensions,o.embedding,'
        'c.id AS candidate,c.person,c.status FROM candidates c JOIN observations o '
        f'ON o.id={address(db,"c")} WHERE c.policy=? AND c.person IS NOT NULL '
        "AND c.status IN ('accepted','pending') AND o.rejected IS NULL " + condition +
        'ORDER BY c.person,o.scan,o.cue,o.id', parameters)
    threshold = max(config.data['recognition']['match_threshold'],
                    config.data['recognition']['cooccurrence_merge_threshold'])
    for row in conflicting_claims(rows, threshold):
        if row['status'] != 'pending':
            continue
        proof = dict(identity=row['person'], scan=row['scan'], cue=row['cue'],
                     threshold=threshold, simultaneous=True)
        db.execute("UPDATE candidates SET person=NULL,status='withheld',payload=json_set(payload,"
            "'$.prior_reason',json_extract(payload,'$.reason'),'$.reason','co_present_identity_conflict',"
            "'$.decision_conflict',json(?)) WHERE id=? AND status='pending'",
            (encoded(proof), row['candidate']))
