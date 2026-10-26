"""Current verified scans, independent of immutable scan creation order."""
from .database import encoded

PREFIX = 'recognition-scan-head:'


def key(scope, namespace, scene):
    return PREFIX + ':'.join(value.encode('utf-8').hex().upper() for value in (scope, namespace, scene))


def predicate(alias='s'):
    """A later insertion supersedes a remembered reuse, including partial scans.

    The creation marker is an ID, not a rowid: private replay databases may
    renumber rows while preserving their original order. Never choose an older
    compatible scan merely because the current one is incomplete/incompatible.
    """
    newest = ('(SELECT head_new.id FROM scans head_new '
              f'WHERE head_new.scope={alias}.scope AND head_new.namespace={alias}.namespace '
              f'AND head_new.scene={alias}.scene ORDER BY head_new.rowid DESC LIMIT 1)')
    address = f"'{PREFIX}'||hex({alias}.scope)||':'||hex({alias}.namespace)||':'||hex({alias}.scene)"
    remembered = ("(SELECT json_extract(head_meta.value,'$.scan') FROM metadata head_meta "
                  f"WHERE head_meta.key={address} AND json_extract(head_meta.value,'$.latest_created_scan')={newest})")
    return f'{alias}.id=COALESCE({remembered},{newest})'


def activate(db, scan):
    """Called only by a run that has checked the current media and scan policy."""
    address = key(scan['scope'], scan['namespace'], scan['scene'])
    newest = db.execute('SELECT id FROM scans WHERE scope=? AND namespace=? AND scene=? '
                        'ORDER BY rowid DESC LIMIT 1',
                        (scan['scope'], scan['namespace'], scan['scene'])).fetchone()[0]
    if newest == scan['id']:
        changed = db.execute('DELETE FROM metadata WHERE key=?', (address,)).rowcount
    else:
        value = encoded(dict(scan=scan['id'], latest_created_scan=newest))
        changed = db.execute('INSERT INTO metadata(key,value) VALUES (?,?) '
            'ON CONFLICT(key) DO UPDATE SET value=excluded.value WHERE value<>excluded.value',
            (address, value)).rowcount
    if changed:
        # Reuse changes reference admission even when embedding bytes do not.
        # Existing audited triggers invalidate scan/global dependent caches.
        db.execute('UPDATE scans SET status=status WHERE id=?', (scan['id'],))


def copy_heads(source, target):
    """Retain current selections only for complete scene histories copied to RAM."""
    for row in source.execute('SELECT key,value FROM metadata WHERE key LIKE ?', (PREFIX+'%',)):
        target.execute('INSERT INTO metadata(key,value) SELECT ?,? '
            "WHERE EXISTS (SELECT 1 FROM scans WHERE id=json_extract(?,'$.scan')) "
            "AND EXISTS (SELECT 1 FROM scans WHERE id=json_extract(?,'$.latest_created_scan'))",
            (row['key'], row['value'], row['value'], row['value']))
