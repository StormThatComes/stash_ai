"""Completion indexes, output reuse and refresh requests."""
from __future__ import annotations

from stash_ai.inference import checkpoints


def initialize_progress(store):
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
def index_result(store, scene, model, kind, record, scope=None, *, metadata_only=False):
    from stash_ai.evidence.thresholds import indexed_signature
    store.connection.execute("INSERT INTO result_index VALUES (?,?,?,?,?,?,?) "
        "ON CONFLICT(scope,scene,model,kind) DO UPDATE SET inference=excluded.inference,"
        "populated=excluded.populated,signature=excluded.signature",
        (scope or store.scope, str(scene), model, kind, 1,
         int(record['populated'] if metadata_only else bool(record['items'])), indexed_signature(record)))


def output_kinds(app):
    return [kind for kind, enabled in (("scene", app.APPLY_SCENE_TAGS), ("markers", app.CREATE_MARKERS)) if enabled]


def completed(store, scene, model, kinds, signature):
    cache_key = ('status', str(scene), model)
    rows = store.cached(cache_key)
    if rows is None:
        rows = {kind: (inference, populated, saved) for kind, inference, populated, saved in store.connection.execute(
        "SELECT kind,inference,populated,signature FROM " + store.table('result_index') + " AS i WHERE scope=? AND scene=? AND model=? "
        "AND NOT EXISTS (SELECT 1 FROM " + store.table('refresh_requests') + " AS r WHERE "
        "r.scope=i.scope AND r.scene=i.scene AND r.model=i.model AND r.kind=i.kind)",
        (store.scope, str(scene), model))}
        store.retain(cache_key, rows)
    return signature is not None and all(kind in rows and rows[kind][0]
                                        and rows[kind][2] == signature for kind in kinds)


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

