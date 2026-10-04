"""Completion indexes, output reuse and refresh requests."""
from __future__ import annotations

from stash_ai.inference import checkpoints
from stash_ai.storage.codec import decode


def initialize_progress(store, previous_version):
    db = store.connection
    db.executescript("""
        CREATE TABLE IF NOT EXISTS result_index (
            scope TEXT NOT NULL, scene TEXT NOT NULL, model TEXT NOT NULL, kind TEXT NOT NULL,
            inference INTEGER NOT NULL, populated INTEGER NOT NULL, signature TEXT,
            PRIMARY KEY(scope,scene,model,kind));
        CREATE TABLE IF NOT EXISTS refresh_requests (
            scope TEXT NOT NULL, scene TEXT NOT NULL, model TEXT NOT NULL, kind TEXT NOT NULL,
            PRIMARY KEY(scope,scene,model,kind));
        CREATE TABLE IF NOT EXISTS checkpoints (
            scope TEXT NOT NULL, scene TEXT NOT NULL, model TEXT NOT NULL,
            signature TEXT NOT NULL, processed INTEGER NOT NULL, total INTEGER NOT NULL,
            top_rows INTEGER NOT NULL, top_scores BLOB NOT NULL,
            PRIMARY KEY(scope,scene,model));
        CREATE TABLE IF NOT EXISTS checkpoint_frames (
            scope TEXT NOT NULL, scene TEXT NOT NULL, model TEXT NOT NULL,
            processed INTEGER NOT NULL, data BLOB NOT NULL,
            PRIMARY KEY(scope,scene,model,processed));
        CREATE TABLE IF NOT EXISTS checkpoint_analysis (
            scope TEXT NOT NULL, scene TEXT NOT NULL, model TEXT NOT NULL, data BLOB NOT NULL,
            PRIMARY KEY(scope,scene,model));
    """)
    if previous_version < 2:
        # One-time migration; future status checks never decompress prediction blobs.
        with db:
            for scope, scene, model, kind, data in db.execute("SELECT scope,scene,model,kind,data FROM results"):
                index_result(store, scene, model, kind, decode(data), scope=scope)
    if previous_version < 4:
        with db:
            # Video sampling is retired; base VTT results and checkpoints stay valid.
            db.execute("DROP TABLE IF EXISTS refinement_progress")
            db.execute("PRAGMA user_version=4")


def index_result(store, scene, model, kind, segments, scope=None, *, metadata_only=False):
    """Index a validated effective view, including a frame-free threshold plan."""
    from stash_ai.evidence.thresholds import indexed_signature
    inference = next((segment for segment in reversed(segments)
                      if segment["meta"].get("origin") == "inference"), None)
    store.connection.execute("INSERT INTO result_index VALUES (?,?,?,?,?,?,?) "
        "ON CONFLICT(scope,scene,model,kind) DO UPDATE SET inference=excluded.inference,"
        "populated=excluded.populated,signature=excluded.signature",
        (scope or store.scope, str(scene), model, kind,
         int(inference is not None),
         int(any(s["populated"] if metadata_only else bool(s["items"]) for s in segments)),
         indexed_signature(inference) if inference is not None else None))


def output_kinds(app):
    return [kind for kind, enabled in (("scene", app.APPLY_SCENE_TAGS), ("markers", app.CREATE_MARKERS)) if enabled]


def completed(store, scene, model, kinds, signature=None):
    cache_key = ('status', str(scene), model)
    rows = store.cached(cache_key)
    if rows is None:
        rows = {kind: (inference, populated, saved) for kind, inference, populated, saved in store.connection.execute(
        "SELECT kind,inference,populated,signature FROM " + store.table('result_index') + " AS i WHERE scope=? AND scene=? AND model=? "
        "AND NOT EXISTS (SELECT 1 FROM " + store.table('refresh_requests') + " AS r WHERE "
        "r.scope=i.scope AND r.scene=i.scene AND r.model=i.model AND r.kind=i.kind)",
        (store.scope, str(scene), model))}
        store.retain(cache_key, rows)
    return all(kind in rows and
               ((rows[kind][0] and rows[kind][2] == signature) if signature is not None
                else (rows[kind][0] or rows[kind][1])) for kind in kinds)


def reuse_replacement_policy(store, scene, model, kinds, signature, replace):
    """Reuse current inference while applying replacement semantics to retained history."""
    from stash_ai.evidence.thresholds import indexed_signature
    matches = {}
    records = {}
    for kind in kinds:
        if store.connection.execute(
                "SELECT 1 FROM " + store.table('refresh_requests') +
                " WHERE scope=? AND scene=? AND model=? AND kind=?",
                (store.scope, str(scene), model, kind)).fetchone():
            return None
        segments = store.records(scene, kind, model, borrow=True).get(model, [])
        # Match the effective policy as completed() does. A raw inference
        # signature can match restored thresholds while overrides still retain
        # the raised values; let threshold reuse reconcile those first.
        current = next((segment for segment in reversed(segments)
                        if segment['meta'].get('origin') == 'inference'
                        and indexed_signature(segment) == signature), None)
        if current is None:
            return None
        records[kind] = segments
        matches[kind] = current

    targets = [(kind, matches[kind]) for kind in kinds
               if replace.get(kind, True)
               and (len(records[kind]) != 1 or records[kind][0] != matches[kind])]
    if not targets:
        return 0
    # Never rewrite source evidence underneath an interrupted publication plan.
    if any(store.snapshot(scene, kind, borrow=True)[1] is not None for kind in ('scene', 'markers')):
        raise RuntimeError(f'{scene}: finish the pending merge or use --abandon-pending before changing replacement policy.')
    with store.connection:
        for kind, segment in targets:
            store.replace_records(scene, model, kind, [segment])
    return len(targets)

def collapse_replacement_history(store, scene, model, kinds, replace):
    """Apply replacement after another cache path has already proven the latest result."""
    targets = []
    for kind in kinds:
        if not replace.get(kind, True):
            continue
        segments = store.records(scene, kind, model, borrow=True).get(model, [])
        if len(segments) <= 1:
            continue
        segment = segments[-1]
        if segment['meta'].get('origin') != 'inference':
            return None
        targets.append((kind, segment))
    if not targets:
        return 0
    if any(store.snapshot(scene, kind, borrow=True)[1] is not None for kind in ('scene', 'markers')):
        raise RuntimeError(f'{scene}: finish the pending merge or use --abandon-pending before changing replacement policy.')
    with store.connection:
        for kind, segment in targets:
            store.replace_records(scene, model, kind, [segment])
    return len(targets)


def request_refresh(store, scenes, models, kinds):
    with store.connection:
        for scene in scenes:
            for model in models:
                store.cache.discard(('status', str(scene), model))
                checkpoints.clear_checkpoint(store, scene, model)
                for kind in kinds:
                    store.connection.execute("INSERT OR IGNORE INTO refresh_requests VALUES (?,?,?,?)",
                                             (store.scope, str(scene), model, kind))


def request_stale_refresh(store, scene, model, kinds, signature):
    """Mark mismatched completed inference stale without destroying resumable work."""
    kinds = tuple(dict.fromkeys(kinds))
    if not kinds:
        return []
    placeholders = ",".join("?" for _ in kinds)
    rows = store.connection.execute(
        "SELECT kind FROM " + store.table('result_index') +
        f" WHERE scope=? AND scene=? AND model=? AND kind IN ({placeholders}) "
        "AND inference=1 AND (signature IS NULL OR signature<>?) ORDER BY kind",
        (store.scope, str(scene), model, *kinds, signature)).fetchall()
    stale = [kind for kind, in rows]
    if not stale:
        return []
    with store.connection:
        store.cache.discard(('status', str(scene), model))
        for kind in stale:
            store.connection.execute(
                "INSERT OR IGNORE INTO refresh_requests VALUES (?,?,?,?)",
                (store.scope, str(scene), model, kind))
    return stale


def pending_refreshes(store, kinds):
    """Group outstanding refresh requests that can affect the selected outputs."""
    kinds = tuple(dict.fromkeys(kinds))
    if not kinds:
        return {}
    placeholders = ",".join("?" for _ in kinds)
    rows = store.connection.execute(
        "SELECT scene,model,kind FROM " + store.table('refresh_requests') +
        f" WHERE scope=? AND kind IN ({placeholders}) ORDER BY scene,model,kind",
        (store.scope, *kinds))
    result = {}
    for scene, model, kind in rows:
        result.setdefault(str(scene), []).append((model, kind))
    return result


def preload_status(store):
    """One small ordered query instead of per-pair status reads across all passes."""
    if not store.cache.limit:
        return
    last, values = None, {}
    rows = store.connection.execute(
        "SELECT i.scene,i.model,i.kind,i.inference,i.populated,i.signature FROM " + store.table('result_index') + " AS i "
        "LEFT JOIN " + store.table('refresh_requests') + " AS r ON r.scope=i.scope AND r.scene=i.scene AND r.model=i.model AND r.kind=i.kind "
        "WHERE i.scope=? AND r.kind IS NULL ORDER BY i.scene,i.model", (store.scope,))
    for scene, model, kind, inference, populated, signature in rows:
        key = ('status', scene, model)
        if key != last:
            if last is not None:
                store.cache.put(last, values)
            last, values = key, {}
        values[kind] = (inference, populated, signature)
    if last is not None:
        store.cache.put(last, values)
