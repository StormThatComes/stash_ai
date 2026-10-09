"""Verified inference identities in state DB; diagnostic history is never evidence."""
from pathlib import Path

from stash_ai.signatures import digest
from stash_ai.storage.codec import encode, decode
from .input_identity import inference_signature

_STATE_FIELDS = frozenset(('_input_options', '_signature_proof'))


def decode_identity(data, signature):
    """Read old plain identities and optional state-only consistency metadata.

    These fields never enter a model, input or checkpoint signature. The checksum
    catches damaged registry associations without fetching otherwise ready inputs.
    """
    value = dict(decode(data))
    required = {'format', 'source', 'repo', 'revision', 'files', 'vocabulary',
                'batch_size', 'scene_threshold', 'marker_threshold', 'aggregation', 'top_k', 'retention'}
    if not isinstance(value, dict) or not required <= value.keys() or not isinstance(value['retention'], dict):
        raise ValueError('Incomplete inference identity.')
    proof = value.pop('_signature_proof', None)
    if proof is not None and proof != digest({'signature': signature, 'identity': value}):
        raise ValueError('Inference identity consistency check failed.')
    options = value.pop('_input_options', None)
    if options is not None and (not isinstance(options, dict) or set(options) != {'paths', 'sample_limit'}
            or not isinstance(options['paths'], str) or type(options['sample_limit']) is not int
            or options['sample_limit'] < 0):
        raise ValueError('Invalid cached input options.')
    return value, options


def registered_identity(store, scene, model, signature):
    """Read authoritative state provenance; run history never authorizes reuse."""
    row = store.connection.execute('SELECT data FROM ' + store.table('inference_identities') +
        ' WHERE scope=? AND scene=? AND model=? AND signature=?',
        (store.scope, str(scene), model, signature)).fetchone()
    return decode_identity(row[0], signature) if row else (None, None)


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
    from .identity_types import ProvenanceIdentity
    return ProvenanceIdentity.from_legacy(identity).predictive(checkpoint=checkpoint).legacy()


def register(store, scene, model, signature, identity, *, input_options=None):
    """Caller has verified these inputs or is about to create their checkpoint."""
    key = (store.scope, str(scene), model, signature)
    value = dict(identity)
    if input_options is not None:
        value['_input_options'] = input_options
    value['_signature_proof'] = digest({'signature': signature, 'identity': value})
    data = encode(value)
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
            candidate, _ = decode_identity(row[0], signature)
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
            candidate, _ = decode_identity(data, signature)
            if (predictive(candidate, checkpoint=True) == predictive(identity, checkpoint=True)
                    and inference_signature(candidate, manifest, hashes) == signature
                    and checkpoint_signature(signature) == row[0]):
                return row[0]
        except (ValueError, TypeError, KeyError):
            continue
    return target
