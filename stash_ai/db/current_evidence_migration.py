"""Retire obsolete active inference without losing pinned publication proofs."""
from types import SimpleNamespace

from stash_ai.storage.codec import decode, encode
from stash_ai.storage.snapshots import decode as decode_snapshot
from stash_ai.storage.archive import retain
from stash_ai.storage.thresholds import overlay, separate
from stash_ai.storage.metadata import FORMAT, summarize, encode as encode_metadata
from stash_ai.storage.progress import index_result
from stash_ai.evidence.formats import current, validate_archived


def migrate(db):
    retire(db, lambda value: value['meta'].get('origin') == 'inference' and not current(value['meta']))


def retire(db, obsolete):
    store = SimpleNamespace(connection=db, table=lambda name: name,
                            unpack_snapshot=decode_snapshot, pack=encode)
    keys = db.execute('SELECT scope,scene,model,kind FROM results').fetchall()
    for scope, scene, model, kind in keys:
        store.scope = scope
        blob, thresholds = db.execute(
            'SELECT r.data,t.thresholds FROM results AS r LEFT JOIN result_thresholds AS t '
            'USING(scope,scene,model,kind) WHERE r.scope=? AND r.scene=? AND r.model=? AND r.kind=?',
            (scope, scene, model, kind)).fetchone()
        previous = overlay(decode(blob), kind, thresholds)
        active = [segment for segment in previous
                  if not obsolete(segment)]
        if len(active) == len(previous):
            continue
        for segment in previous:
            if segment['meta'].get('origin') == 'inference' and not current(segment['meta']):
                validate_archived(segment)
        retain(store, scene, model, kind, previous, active)
        params = scope, scene, model, kind
        db.execute('DELETE FROM result_thresholds WHERE scope=? AND scene=? AND model=? AND kind=?', params)
        db.execute('DELETE FROM result_metadata WHERE scope=? AND scene=? AND model=? AND kind=?', params)
        if active:
            raw, effective = separate(active, kind)
            db.execute('UPDATE results SET data=? WHERE scope=? AND scene=? AND model=? AND kind=?', (encode(raw), *params))
            if effective is not None:
                db.execute('INSERT INTO result_thresholds VALUES (?,?,?,?,?)', (*params, effective))
            db.execute('INSERT INTO result_metadata VALUES (?,?,?,?,?,?)', (*params, FORMAT, encode_metadata(summarize(raw))))
            index_result(store, scene, model, kind, active, scope=scope)
        else:
            db.execute('DELETE FROM results WHERE scope=? AND scene=? AND model=? AND kind=?', params)
            db.execute('DELETE FROM result_index WHERE scope=? AND scene=? AND model=? AND kind=?', params)
        db.execute('INSERT OR IGNORE INTO refresh_requests VALUES (?,?,?,?)', params)
        for table in ('checkpoints', 'checkpoint_frames', 'checkpoint_analysis'):
            db.execute(f'DELETE FROM {table} WHERE scope=? AND scene=? AND model=?', params[:3])
