"""Abandon unfinished plans while retaining their evidence and ownership.

No Stash writes happen during abandonment. Interrupted publications become
recovery evidence; interrupted cleanup masks become conditional on observed
outputs. Cleanup reconciles a scene lazily before applying its new policies.
"""
from collections import Counter
from copy import deepcopy
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
        available = {row[0] for row in db.execute("SELECT name FROM cleanup_source.sqlite_master WHERE type='table'")}
        if 'recovery_cleanups' in available:
            db.execute('INSERT INTO recovery_cleanups SELECT * FROM cleanup_source.recovery_cleanups WHERE scope=?',
                       (store.scope,))
            db.commit()
    store.recovery_available = bool(db.execute('SELECT 1 FROM recovery_publications WHERE scope=? LIMIT 1',
                                              (store.scope,)).fetchone())
    if store.preview_source and 'recovery_publications' in available:
        store.recovery_available |= bool(db.execute(
            'SELECT 1 FROM cleanup_source.recovery_publications WHERE scope=? LIMIT 1', (store.scope,)).fetchone())
    load_cleanups(store)


def load_cleanups(store):
    store.abandoned_cleanups = {generation: set(json.loads(kinds)) for generation, kinds in
        store.connection.execute('SELECT generation,kinds FROM recovery_cleanups WHERE scope=?', (store.scope,))}


def abandon(store, report):
    """One local transaction; the ordinary state DB writer lock stays held."""
    from stash_merge import now
    from stash_preview import check_source
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
        # Version 7 prevents older executables from ignoring recovery evidence.
        db.execute('PRAGMA user_version=7')
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
    from stash_cleanup import mask_snapshot
    rows = store.read_rows('SELECT data FROM ' + store.table('recovery_publications') +
        ' WHERE scope=? AND scene=? AND kind=? ORDER BY operation DESC', (store.scope, str(scene), kind))
    result = [mask_snapshot(store, store.unpack(row[0]), kind) for row in rows]
    store.retain(key, result)
    return result


def clear_publications(store, scene, kind):
    """Called only inside a successful publication's transaction."""
    if not store.recovery_available:
        return
    from stash_preview import delete
    delete(store, 'recovery_publications', scene, kind=kind)
    store.connection.execute('DELETE FROM recovery_publications WHERE scope=? AND scene=? AND kind=?',
                             (store.scope, str(scene), kind))
    store.cache.discard(('recovery', str(scene), kind))


def reconcile(app, stash, store, scene, kind, published):
    """Observe surviving outputs; never replay an abandoned plan or infer votes."""
    from stash_merge import AI, union_intervals, source_codes
    from stash_occurrences import OCCURRENCE_TAG, is_occurrence
    from stash_fusion import prune_blocked_extras
    from stash_performance import freeze
    archived = publications(store, scene['id'], kind)
    if not archived and not (published or {}).get('recovery_required'):
        return published, False
    candidates = [*([published] if published is not None else []), *archived]
    result, contributions, seen = {}, [], set()

    def append(item):
        identity = freeze(item)
        if identity not in seen:
            seen.add(identity)
            contributions.append(item)

    if kind == 'scene':
        current_ids = {str(t['id']) for t in scene.get('tags', [])}
        current_names = {t['name'] for t in scene.get('tags', [])}
        result['owned_ids'] = sorted(current_ids & {str(i) for s in candidates for i in s.get('owned_ids', [])})
        # Existing labels retain their last confirmed proofs. Interrupted plans
        # supply newly attached labels; never combine different inference runs.
        labels = set()
        for snapshot in candidates:
            selected = [i for i in snapshot.get('contributions', [])
                        if AI.prefix + i['tag'] in current_names and i['tag'] not in labels]
            for item in selected:
                append(item)
            labels.update(i['tag'] for i in selected)
    else:
        versions = [union_intervals(s.get('contributions', []), borrow=True) for s in candidates]
        for marker in scene.get('scene_markers', []):
            if not app.owned_marker(marker, AI):
                continue
            occurrence = any(t['name'] == OCCURRENCE_TAG for t in marker.get('tags', []))
            present = {t['name'][len(AI.prefix):] for t in marker.get('tags', [])
                       if t['name'].startswith(AI.prefix) and t['name'] not in (AI.owner_tag, OCCURRENCE_TAG)}
            matches = [component for version in versions for component in version
                if marker['primary_tag']['name'] == AI.prefix + component['tag']
                and abs(round(component['start'], 3) - marker['seconds']) <= .0005
                and (marker.get('end_seconds') is None or (component['end'] is not None
                     and abs(round(component['end'], 3) - marker['end_seconds']) <= .0005))
                and is_occurrence(component) == occurrence]
            if not matches:
                raise RuntimeError(f"Cannot identify surviving marker {marker['id']} from saved publications. "
                                   'Run --merge-only with the current YAML policies to reconcile this scene.')
            # Same time/core can occur in both plans with different extras.
            # Prefer the version matching what Stash actually received.
            def rank(component):
                title = AI.title_prefix + component['tag'] + ' [' + '+'.join(source_codes(component['sources'])) + ']'
                if is_occurrence(component):
                    title += ' · wykrycie'
                return (title != marker['title'], len(present - set(component['extra_tags'])),
                        len(set(component['extra_tags']) - present))
            match = min(matches, key=rank)
            present_ids = {str(t['id']) for t in marker.get('tags', [])}
            for item in match['contributors']:
                extras = set(item.get('extra_tags', [])) | set(item.get('extra_support', {}))
                observed = prune_blocked_extras(deepcopy(item), extras - present)
                if 'extra_ids' in observed:
                    observed['extra_ids'] = [i for i in observed['extra_ids'] if str(i) in present_ids]
                append(observed)
    result['contributions'] = contributions
    # Save this observed baseline even if the new cleanup has nothing to remove.
    # The dry-run StateStore writes only its temporary overlay.
    store.publish(str(scene['id']), {kind: result})
    return result, True


def run_only(app, args):
    from stash_state import StateStore, scope_for
    with StateStore(args.state_db, scope_for(app.STASH_URL), app.DRY_RUN, allow_cleanup=True) as store:
        stats = abandon(store, args.report.write)
    args.report.finish('Abandon pending' + (' (planned)' if app.DRY_RUN else ''), stats)
    return 0
