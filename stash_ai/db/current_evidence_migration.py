"""Retire obsolete v18 records while retaining exact publication pins."""
from types import SimpleNamespace

from stash_ai.storage.archive import pinned
from stash_ai.storage.codec import decode, encode
from stash_ai.storage.snapshots import decode as decode_snapshot
from stash_ai.evidence.formats import current, validate_archived
from stash_ai.evidence.identity import marker_segment_signature


def retain(store, scene, model, kind, record):
    signature = marker_segment_signature(record)
    if (model, signature) in pinned(store, scene):
        store.connection.execute('INSERT OR IGNORE INTO inference_archive VALUES (?,?,?,?,?,?)',
                                 (store.scope, str(scene), model, kind, signature, encode(record)))


def migrate(db):
    store = SimpleNamespace(connection=db, table=lambda name: name,
                            unpack_snapshot=decode_snapshot)
    # Fetch keys only; one scene's evidence may contain many thousands of frames.
    for scope, scene, model, kind in db.execute('SELECT scope,scene,model,kind FROM results').fetchall():
        params = scope, scene, model, kind
        blob = db.execute('SELECT data FROM results WHERE scope=? AND scene=? AND model=? AND kind=?',
                          params).fetchone()[0]
        record = decode(blob)
        if not isinstance(record, dict) or not isinstance(record.get('meta'), dict):
            raise ValueError('Schema v18 requires a single inference record; inspect the state DB.')
        if record['meta'].get('origin') != 'inference':
            raise ValueError('Schema v18 requires inference evidence; inspect the state DB.')
        if current(record['meta']):
            continue
        validate_archived(record)
        store.scope = scope
        retain(store, scene, model, kind, record)
        db.execute('DELETE FROM results WHERE scope=? AND scene=? AND model=? AND kind=?', params)
        db.execute('DELETE FROM result_index WHERE scope=? AND scene=? AND model=? AND kind=?', params)
        db.execute('INSERT OR IGNORE INTO refresh_requests VALUES (?,?,?,?)', params)
        for table in ('checkpoints', 'checkpoint_frames', 'checkpoint_analysis'):
            db.execute(f'DELETE FROM {table} WHERE scope=? AND scene=? AND model=?', params[:3])
