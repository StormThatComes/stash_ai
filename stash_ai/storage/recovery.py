"""Abandon unfinished plans while retaining their evidence and ownership.

No Stash writes happen during abandonment. Interrupted publications become
recovery evidence; interrupted cleanup masks become conditional on observed
outputs. Publication recovery observes a scene lazily before applying new cleanup policies.
"""

from collections import Counter
import json
import uuid


def initialize(store):
    db = store.connection
    db.executescript('''
        CREATE TABLE IF NOT EXISTS recovery_publications (
            scope TEXT NOT NULL, scene TEXT NOT NULL, kind TEXT NOT NULL,
            operation TEXT NOT NULL, data BLOB NOT NULL,
            PRIMARY KEY(scope,scene,kind,operation));
        CREATE TABLE IF NOT EXISTS recovery_cleanups (
            scope TEXT NOT NULL, generation INTEGER NOT NULL, kinds TEXT NOT NULL,
            PRIMARY KEY(scope,generation));
    ''')
    if store.preview_source:
        db.execute('INSERT INTO recovery_cleanups SELECT * FROM cleanup_source.recovery_cleanups WHERE scope=?',
                   (store.scope,))
        db.commit()
    store.recovery_available = bool(db.execute('SELECT 1 FROM recovery_publications WHERE scope=? LIMIT 1',
                                              (store.scope,)).fetchone())
    if store.preview_source:
        store.recovery_available |= bool(db.execute(
            'SELECT 1 FROM cleanup_source.recovery_publications WHERE scope=? LIMIT 1', (store.scope,)).fetchone())
    load_cleanups(store)


def load_cleanups(store):
    store.abandoned_cleanups = {generation: set(json.loads(kinds)) for generation, kinds in
        store.connection.execute('SELECT generation,kinds FROM recovery_cleanups WHERE scope=?', (store.scope,))}


def abandon(store, report):
    """One local transaction; the ordinary state DB writer lock stays held."""
    from stash_ai.inference.results import now
    from stash_ai.storage.preview import check_source
    check_source(store)
    db = store.connection
    pending = store.table('snapshots')
    count = db.execute('SELECT count(*) FROM ' + pending + ' WHERE scope=? AND pending IS NOT NULL',
                       (store.scope,)).fetchone()[0]
    stats = Counter(publications_abandoned=count, cleanups_abandoned=int(bool(store.cleanup_pending)))
    if not any(stats.values()):
        report('No pending merge or cleanup plan to abandon.')
        return stats
    operation = now() + ':' + uuid.uuid4().hex
    with db:
        if not db.in_transaction:
            db.execute('BEGIN')
        if count:
            db.execute('INSERT INTO recovery_publications SELECT scope,scene,kind,?,pending FROM ' + pending +
                       ' WHERE scope=? AND pending IS NOT NULL', (operation, store.scope))
            if store.preview_source:
                db.execute('INSERT OR IGNORE INTO snapshots SELECT * FROM cleanup_source.snapshots '
                           'WHERE scope=? AND pending IS NOT NULL', (store.scope,))
            db.execute('UPDATE snapshots SET pending=NULL WHERE scope=? AND pending IS NOT NULL', (store.scope,))
        if store.cleanup_pending:
            # Keep the masks and generation, but no longer treat this intent as
            # proof that every planned deletion succeeded. Applies to v5/v6 too.
            kinds = set()
            for name, revisions in store.cleanup_masks.items():
                if store.cleanup_generation not in revisions.values():
                    continue
                kinds.update(['markers'] if name == 'core' or name.startswith('marker_') else
                             ['scene'] if name.startswith('scene_') else ['scene', 'markers'])
            db.execute('INSERT OR IGNORE INTO recovery_cleanups VALUES (?,?,?)',
                       (store.scope, store.cleanup_generation, json.dumps(sorted(kinds or {'scene', 'markers'}))))
            db.execute('UPDATE cleanup_state SET pending=NULL WHERE scope=?', (store.scope,))
        db.execute('DELETE FROM merge_checks WHERE scope=?', (store.scope,))
    store.cleanup_pending = None
    store.recovery_available |= bool(count)
    load_cleanups(store)
    store.cache.clear()
    report(f"Abandoned plans: cleanup={stats['cleanups_abandoned']}, publications={count}. "
           'Applied Stash changes and raw inference are retained; new commands use current YAML policies.')
    return stats


def publications(store, scene, kind):
    if not store.recovery_available:
        return []
    key = ('recovery', str(scene), kind)
    cached = store.cached(key)
    if cached is not None:
        return cached
    from stash_ai.cleanup.state import mask_snapshot
    rows = store.read_rows('SELECT data FROM ' + store.table('recovery_publications') +
        ' WHERE scope=? AND scene=? AND kind=? ORDER BY operation DESC', (store.scope, str(scene), kind))
    result = [mask_snapshot(store, store.unpack(row[0]), kind) for row in rows]
    store.retain(key, result)
    return result


def clear_publications(store, scene, kind):
    """Called only inside a successful publication's transaction."""
    if not store.recovery_available:
        return
    from stash_ai.storage.preview import delete
    delete(store, 'recovery_publications', scene, kind=kind)
    store.connection.execute('DELETE FROM recovery_publications WHERE scope=? AND scene=? AND kind=?',
                             (store.scope, str(scene), kind))
    store.cache.discard(('recovery', str(scene), kind))
