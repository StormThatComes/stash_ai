"""Reversible selected-scene associations; profiles and manual metadata survive."""
import json

from .publication import PendingRecognition, operation_id, perform
from .stash import checked_id, owns
from .database import encoded
from .progress import tracking


def _inactive(db, scope, scene, person):
    identifier = operation_id(scope, scene, person, 'link')
    row = db.execute('SELECT payload FROM publications WHERE id=?', (identifier,)).fetchone()
    if row:
        payload = json.loads(row['payload'])
        payload['active'] = False
        db.execute('UPDATE publications SET payload=? WHERE id=?', (encoded(payload), identifier))


def reconcile_unlinks(database, client, scope, scene):
    """Do not add links over an unresolved earlier removal, even after retuning."""
    rows = database.connection.execute("SELECT * FROM publications WHERE scope=? AND scene=? AND kind='unlink' "
                                        "AND status IN ('pending','unknown')", (scope, scene)).fetchall()
    if not rows:
        return
    live = {checked_id(p)['id'] for p in client.scene(scene)['performers']}
    with tracking(
        rows,
        'Reconcile pending unlinks',
        unit='operation',
    ) as tracked:
        for row in tracked:
            if row['target'] in live:
                raise PendingRecognition('Earlier scene unlink is unresolved; no blind retry or replacement link.')
            with database.connection:
                database.connection.execute("UPDATE publications SET status='observed',error=NULL WHERE id=?", (row['id'],))
                _inactive(database.connection, scope, scene, row['person'])


def remove_stale_links(database, client, report, scope, scene, wanted, namespace):
    """REMOVE only journaled application links to freshly verified owned profiles."""
    db, removed = database.connection, 0
    rows = db.execute("SELECT w.*,p.token FROM publications w JOIN people p ON p.id=w.person "
                      "WHERE w.scope=? AND w.scene=? AND w.kind='link' AND p.namespace=? "
                      "AND w.status IN ('confirmed','observed','pending','unknown')", (scope, scene, namespace)).fetchall()
    with tracking(
        rows,
        'Remove stale scene links',
        unit='link',
    ) as tracked:
        for row in tracked:
            payload = json.loads(row['payload'])
            if row['target'] in wanted or not payload.get('managed', True):
                continue
            if not payload.get('active', True):
                removal = db.execute('SELECT status FROM publications WHERE id=?',
                    (operation_id(scope, scene, row['person'], 'unlink'),)).fetchone()
                if row['status'] not in ('pending', 'unknown') or removal and removal['status'] in ('confirmed', 'observed'):
                    continue
                # Older writers retired absent uncertain ADDs without an unlink.
                # Recover that unresolved ownership, preserving acknowledged removals.
                payload['active'] = True
                with db:
                    db.execute('UPDATE publications SET payload=? WHERE id=?', (encoded(payload), row['id']))
            live = {checked_id(p)['id'] for p in client.scene(scene)['performers']}
            if row['target'] not in live:
                if row['status'] in ('pending', 'unknown'):
                    raise PendingRecognition('Earlier scene ADD is unresolved; retain ownership until its outcome is observed.')
                with db:
                    _inactive(db, scope, scene, row['person'])
                continue
            if row['status'] in ('pending', 'unknown'):
                with db:
                    db.execute("UPDATE publications SET status='observed',error=NULL WHERE id=?", (row['id'],))
            if not owns(client.performer(row['target']), row['token']):
                raise PendingRecognition('Performer ownership changed before stale scene link removal.')
            identifier = operation_id(scope, scene, row['person'], 'unlink')
            previous = db.execute('SELECT status FROM publications WHERE id=?', (identifier,)).fetchone()
            if previous and previous['status'] in ('pending', 'unknown'):
                raise PendingRecognition('Earlier scene unlink is unresolved; no blind retry.')
            def remove():
                # Refresh both ownership and association immediately before dispatch.
                if not owns(client.performer(row['target']), row['token']):
                    raise PendingRecognition('Concurrent performer ownership edit preserved.')
                acknowledgement = client.remove_scene_person(scene, row['target'])
                checked_id(acknowledgement, scene)
                if row['target'] in {checked_id(p)['id'] for p in acknowledgement['performers']}:
                    raise PendingRecognition('Scene unlink acknowledgement still contains performer.')
                if row['target'] in {checked_id(p)['id'] for p in client.scene(scene)['performers']}:
                    raise PendingRecognition('Scene unlink was not confirmed by a fresh read.')
            perform(database, client, report, scope=scope, scene=scene, person=row['person'], kind='unlink', target=row['target'],
                    payload={'performer': row['target'], 'input_contract': 'bulk-scene-remove-ids-v1'},
                    precondition={'token': row['token'], 'performers': sorted(live)}, callback=remove)
            with db:
                _inactive(db, scope, scene, row['person'])
            removed += 1
    return removed


def reconcile_legacy_merges(database, client, scope, scenes):
    """Observe uncertain old destructive merges before assigning their profiles."""
    from .merge_publication import _reconcile
    from .identity_merges import canonical
    selected = set(scenes)
    with tracking(
        database.connection.execute("SELECT * FROM publications WHERE scope=? AND kind='merge' "
                                            "AND status IN ('pending','unknown')", (scope,)).fetchall(),
        'Reconcile legacy merges',
        unit='operation',
        total=lambda: database.connection.execute(
            "SELECT COUNT(*) FROM publications WHERE scope=? AND kind='merge' AND status IN "
            "('pending','unknown')",
            (scope,),
        ).fetchone()[0],
    ) as tracked:
        for row in tracked:
            root = canonical(database.connection, row['person'])
            involved = database.connection.execute('SELECT DISTINCT s.scene FROM scans s JOIN observations o ON o.scan=s.id '
                'JOIN members m ON m.observation=o.id WHERE m.person=? AND s.scope=?',
                (root, scope)).fetchall()
            if not any(r[0] in selected for r in involved):
                continue
            _reconcile(database, client, row)
            with database.connection:
                database.connection.execute("UPDATE identity_merges SET status='observed' WHERE source=?", (row['person'],))
