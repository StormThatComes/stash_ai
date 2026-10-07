"""Verified inference identities in state DB; diagnostic history is never evidence."""
from pathlib import Path

from stash_ai.signatures import digest
from stash_ai.storage.codec import encode, decode
from .input_identity import inference_signature


def initialize(store):
    store.connection.execute('''CREATE TABLE IF NOT EXISTS inference_identities (
        scope TEXT NOT NULL, scene TEXT NOT NULL, model TEXT NOT NULL,
        signature TEXT NOT NULL, data BLOB NOT NULL,
        PRIMARY KEY(scope,scene,model,signature))''')


def predictive(identity, *, checkpoint=False):
    """Batch and asset location describe execution, never new detections.

    Retention/threshold changes require a completed-evidence coverage check;
    partial checkpoints retain their original capture policy.
    """
    excluded = {'batch_size'}
    if not checkpoint:
        excluded.update(('retention', 'scene_threshold', 'marker_threshold'))
    result = {key: value for key, value in identity.items() if key not in excluded}
    result['files'] = [[Path(path).name, checksum] for path, checksum in identity['files']]
    return result


def register(store, scene, model, signature, identity):
    """Caller has verified these inputs or is about to create their checkpoint."""
    key = (store.scope, str(scene), model, signature)
    data = encode(identity)
    saved = store.connection.execute('SELECT data FROM ' + store.table('inference_identities') +
            ' WHERE scope=? AND scene=? AND model=? AND signature=?', key).fetchone()
    if saved and saved[0] == data:
        return
    with store.connection:
        store.connection.execute('INSERT INTO inference_identities VALUES (?,?,?,?,?) '
            'ON CONFLICT(scope,scene,model,signature) DO UPDATE SET data=excluded.data', (*key, data))


def prune(store, scene, model, *, keep=(), checkpoint=None, kinds=(), analysis_format=None):
    """Keep only current raw outputs and the selected in-progress checkpoint."""
    if store.preview_source:
        return
    signatures = set(keep)
    for kind in ('scene', 'markers'):
        record = store.result_metadata(scene, kind, model)
        if record is not None:
            signatures.add(record['meta']['inference_signature'])
    rows = store.connection.execute('SELECT signature FROM inference_identities '
        'WHERE scope=? AND scene=? AND model=?', (store.scope, str(scene), model)).fetchall()
    obsolete = [signature for signature, in rows if signature not in signatures and
        (checkpoint is None or digest({'input': signature, 'kinds': kinds, 'scene_frames': 1,
                                     'scene_analysis': analysis_format}) != checkpoint)]
    if obsolete:
        with store.connection:
            store.connection.executemany('DELETE FROM inference_identities '
                'WHERE scope=? AND scene=? AND model=? AND signature=?',
                [(store.scope, str(scene), model, value) for value in obsolete])


def saved_identity(store, scene, model, signature, identity, manifest, hashes, meta):
    row = store.connection.execute('SELECT data FROM ' + store.table('inference_identities') +
        ' WHERE scope=? AND scene=? AND model=? AND signature=?', (store.scope, str(scene), model, signature)).fetchone()
    if row:
        try:
            candidate = decode(row[0])
            if inference_signature(candidate, manifest, hashes) == signature:
                return candidate
        except (ValueError, TypeError, KeyError):
            return None
        return None
    # Old raw records remain unchanged. Register only an exactly proven identity.
    candidate = {**identity, 'scene_threshold': meta['scene_threshold'], 'marker_threshold': meta['marker_threshold']}
    if inference_signature(candidate, manifest, hashes) == signature:
        register(store, scene, model, signature, candidate)
        return candidate
    return None


def resume_signature(store, scene, model, identity, manifest, hashes, kinds, analysis_format):
    def checkpoint_signature(value):
        return digest({'input': value, 'kinds': kinds, 'scene_frames': 1, 'scene_analysis': analysis_format})
    target = checkpoint_signature(inference_signature(identity, manifest, hashes))
    row = store.connection.execute('SELECT signature FROM ' + store.table('checkpoints') +
        ' WHERE scope=? AND scene=? AND model=?', (store.scope, str(scene), model)).fetchone()
    if not row or row[0] == target:
        return target
    for signature, data in store.connection.execute('SELECT signature,data FROM ' + store.table('inference_identities') +
            ' WHERE scope=? AND scene=? AND model=?', (store.scope, str(scene), model)):
        try:
            candidate = decode(data)
            if (predictive(candidate, checkpoint=True) == predictive(identity, checkpoint=True)
                    and inference_signature(candidate, manifest, hashes) == signature
                    and checkpoint_signature(signature) == row[0]):
                return row[0]
        except (ValueError, TypeError, KeyError):
            continue
    return target
