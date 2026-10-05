"""Move historical inference into a publication-pinned archive during --db-update."""
from types import SimpleNamespace

from stash_ai.storage.codec import encode
from stash_ai.storage.archive import prune
from .result_history import active_segments, retain
from .result_history import overlay, separate
from .result_history import FORMAT, summarize, encode_metadata
from .result_history import index_result
from stash_ai.storage.snapshots import decode as decode_snapshot


def migrate(db):
    # Fetch keys, not every large evidence BLOB at once.
    from .result_history import history, threshold_column
    column = threshold_column(db)
    keys = db.execute('SELECT scope,scene,model,kind FROM results').fetchall()
    store = SimpleNamespace(connection=db, table=lambda name: name,
                            unpack_snapshot=decode_snapshot,
                            pack=encode, preview_source=False)
    for scope, scene, model, kind in keys:
        store.scope = scope
        blob, thresholds = db.execute(
            f'SELECT r.data,t.{column} FROM results AS r LEFT JOIN result_thresholds AS t '
            'USING(scope,scene,model,kind) WHERE r.scope=? AND r.scene=? AND r.model=? AND r.kind=?',
            (scope, scene, model, kind)).fetchone()
        previous = overlay(history(blob), kind, thresholds)
        active = active_segments(previous)
        if len(active) == len(previous):
            continue
        retain(store, scene, model, kind, previous, active)
        raw, effective = separate(active, kind)
        db.execute('UPDATE results SET data=? WHERE scope=? AND scene=? AND model=? AND kind=?',
                   (encode(raw), scope, scene, model, kind))
        db.execute('DELETE FROM result_thresholds WHERE scope=? AND scene=? AND model=? AND kind=?',
                   (scope, scene, model, kind))
        if effective is not None:
            db.execute('INSERT INTO result_thresholds VALUES (?,?,?,?,?)', (scope, scene, model, kind, effective))
        db.execute('INSERT OR REPLACE INTO result_metadata VALUES (?,?,?,?,?,?)',
                   (scope, scene, model, kind, FORMAT, encode_metadata(summarize(raw))))
        index_result(store, scene, model, kind, active, scope=scope)

    for scope, scene in db.execute('SELECT DISTINCT scope,scene FROM inference_archive').fetchall():
        store.scope = scope
        prune(store, scene)
