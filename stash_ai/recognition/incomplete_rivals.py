"""A failed replacement scan cannot turn a historical rival into no rival."""
from .scan_heads import predicate as current_scan


def scenes(db,scope,namespace):
    body=db.execute("SELECT 1 FROM sqlite_master WHERE name='body_members' AND type='table'").fetchone()
    evidence="EXISTS (SELECT 1 FROM observations o JOIN members m ON m.observation=o.id WHERE o.scan=old.id AND o.rejected IS NULL)"
    if body:
        evidence+=" OR EXISTS (SELECT 1 FROM body_observations b JOIN body_members m ON m.observation=b.id WHERE b.scan=old.id AND b.rejected IS NULL AND m.person IS NOT NULL)"
    return [r[0] for r in db.execute(
        "SELECT s.scene FROM scans s WHERE s.scope=? AND s.namespace=? AND s.status<>'complete' AND "+current_scan()+
        ' AND EXISTS (SELECT 1 FROM scans old WHERE old.scope=s.scope AND old.namespace=s.namespace '
        "AND old.scene=s.scene AND old.id<>s.id AND ("+evidence+')) ORDER BY CAST(s.scene AS INTEGER),s.scene',
        (scope,namespace))]


def require_complete(db,scope,namespace,selected=()):
    pending=scenes(db,scope,namespace)
    if pending and selected:
        from .database import encoded
        from .reference_gap_scope import MissingReferences
        scans = [item['scan'] for item in selected]
        # The preflight can narrow the affected set only when every usable
        # selected address already has a measured owner. Unknown inputs defer.
        owners = set()
        for table,members,gate in (('observations','members',''),('body_observations','body_members'," AND m.status='accepted'")):
            if not db.execute("SELECT 1 FROM sqlite_master WHERE name=?", (members,)).fetchone():
                continue
            owners.update(row[0] for row in db.execute(f'SELECT DISTINCT m.person FROM {table} o '
                f'LEFT JOIN {members} m ON m.observation=o.id'+gate+
                ' WHERE o.rejected IS NULL AND o.embedding IS NOT NULL AND o.scan IN (SELECT value FROM json_each(?))',
                (encoded(scans),)))
        if owners and None not in owners:
            pending = MissingReferences(db,scope,namespace,pending,exclude_scans=scans).affected(owners)
    if pending:
        raise ValueError('Historical recognition rivals have incomplete replacement scans; finish recognition for scenes: '+', '.join(pending)+'.')


def protect(db,scope,namespace,result):
    pending=scenes(db,scope,namespace)
    if pending:
        result['incomplete_reference_scenes']=pending
        from .reference_gap_scope import MissingReferences
        MissingReferences(db,scope,namespace,pending).protect(result,'incomplete_reference_scans')
    return result
