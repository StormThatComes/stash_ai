"""Count immutable retained body assignments once, then apply trial-local changes."""
from collections import Counter
from .database import encoded


def groups(db, scan_ids, contract, *, source='main', changed=False):
    # Body observations are shared through a TEMP view over retained evidence.
    # Only the mutable membership table changes between these two projections.
    prefix = 'trial_changed d JOIN ' if changed else ''
    join = ' ON m.observation=d.id' if changed else ''
    sql = (f'SELECT m.person,m.status,m.role,CASE WHEN json_type(m.proof,\'$.reason\') IS NULL '
           f"THEN 'unknown' ELSE json_extract(m.proof,'$.reason') END,COUNT(*) FROM {prefix}"
           f'{source}.body_members m{join} JOIN body_observations b ON b.id=m.observation '
           'WHERE b.scan IN (SELECT value FROM json_each(?)) AND b.contract=? GROUP BY 1,2,3,4')
    return db.execute(sql, (encoded(scan_ids),contract))


def report(rows, coverage):
    counts, reasons, identities = Counter(), Counter(), {}
    for (owner,status,role,reason), n in rows.items():
        if not n:
            continue
        if n < 0:
            raise ValueError('Inconsistent solver body assignment count.')
        counts[status] += n
        counts[role] += n
        reasons[reason] += n
        person = identities.setdefault((owner,status),dict(person=owner,status=status,observations=0,anchors=0,auxiliary=0))
        person['observations'] += n
        person['anchors'] += n*(role=='anchor')
        person['auxiliary'] += n*(role=='auxiliary')
    ordered = sorted(identities,key=lambda pair:(pair[0] is not None,pair[0] or '',pair[1]))
    return dict(coverage,assignments=dict(counts),reasons=dict(reasons),identities=[identities[k] for k in ordered])


def incremental(index, target, coverage, scan_ids, total):
    # The dense path also supports disposable pre-index formats.
    if (coverage is None or not index.execute("SELECT 1 FROM sqlite_master WHERE name='retained_body_groups'").fetchone()
            or not any(row[1]=='retained' for row in target.execute('PRAGMA database_list'))
            or target.execute('SELECT COUNT(*) FROM trial_changed').fetchone()[0] > total//2):
        return None
    counts = Counter({tuple(row[:4]):row[4] for row in index.execute('SELECT * FROM retained_body_groups')})
    for source, sign in [('retained',-1),('main',1)]:
        for *key,n in groups(target,scan_ids,coverage['contract'],source=source,changed=True):
            counts[tuple(key)] += sign*n
    return report(counts,coverage)
