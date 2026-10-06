"""Exact inference proofs pinned by durable publication state, outside active results."""
from stash_ai.evidence.identity import marker_segment_signature


def initialize(store):
    store.connection.execute('''CREATE TABLE IF NOT EXISTS inference_archive (
        scope TEXT NOT NULL, scene TEXT NOT NULL, model TEXT NOT NULL,
        kind TEXT NOT NULL, signature TEXT NOT NULL, data BLOB NOT NULL,
        PRIMARY KEY(scope,scene,model,kind,signature))''')


def references(snapshot):
    """Collect exact source signatures, including nested scene/occurrence proofs."""
    required, pending, seen = set(), [snapshot], set()
    while pending:
        value = pending.pop()
        if isinstance(value, (list, dict)):
            if id(value) in seen:
                continue
            seen.add(id(value))
        if isinstance(value, list):
            pending.extend(value)
        elif isinstance(value, dict):
            if value.get('requires_rebase'):
                continue
            source, signature = value.get('source'), value.get('evidence_signature')
            if isinstance(source, str) and isinstance(signature, str):
                required.add((source, signature))
            for field in ('frame_context', 'scene_context', 'support_context'):
                required.update(value.get(field, {}).items())
            pending.extend(value.values())
    return required


def pinned(store, scene):
    """Use unmasked persisted state: cleanup/recovery may still need its proofs."""
    params = (store.scope, str(scene))
    required = set()
    for published, pending in store.connection.execute(
            'SELECT published,pending FROM ' + store.table('snapshots') + ' WHERE scope=? AND scene=?', params):
        for blob in (published, pending):
            if blob is not None:
                required.update(references(store.unpack_snapshot(blob)))
    for blob, in store.connection.execute(
            'SELECT data FROM ' + store.table('recovery_publications') + ' WHERE scope=? AND scene=?', params):
        required.update(references(store.unpack_snapshot(blob)))
    return required


def retain(store, scene, model, kind, previous, active):
    if previous is None:
        return
    signature = marker_segment_signature(previous)
    if signature == marker_segment_signature(active) or (model, signature) not in pinned(store, scene):
        return
    raw = {key: value for key, value in previous.items()
           if key != 'publication_threshold'}
    store.connection.execute('INSERT OR IGNORE INTO inference_archive VALUES (?,?,?,?,?,?)',
                             (store.scope, str(scene), model, kind, signature, store.pack(raw)))

def prune(store, scene):
    """Publication promotion releases proofs no longer referenced by any output."""
    rows = store.connection.execute(
        'SELECT model,kind,signature FROM ' + store.table('inference_archive') +
        ' WHERE scope=? AND scene=?', (store.scope, str(scene))).fetchall()
    if not rows:
        return
    required = pinned(store, scene)
    # A sparse preview shadows the source rows before keeping only live pins.
    if store.preview_source:
        store.connection.execute('INSERT OR IGNORE INTO inference_archive SELECT * FROM '
                                 'cleanup_source.inference_archive WHERE scope=? AND scene=?',
                                 (store.scope, str(scene)))
        from stash_ai.storage.preview import delete
        delete(store, 'inference_archive', scene)
    for model, kind, signature in rows:
        if (model, signature) not in required:
            store.connection.execute('DELETE FROM inference_archive WHERE scope=? AND scene=? '
                                     'AND model=? AND kind=? AND signature=?',
                                     (store.scope, str(scene), model, kind, signature))


def records(store, scene, kind):
    """Read only for frozen replay; active fusion and reports never consult this table."""
    from stash_ai.storage.segments import StoredSegment
    from stash_ai.evidence.formats import validate_current
    from stash_ai.db.payloads import valid_result
    result = {}
    for model, blob in store.read_rows('SELECT model,data FROM ' + store.table('inference_archive') +
                                      ' WHERE scope=? AND scene=? AND kind=?',
                                      (store.scope, str(scene), kind)):
        record = store.unpack(blob)
        validate_current(record)
        if not valid_result(record, kind):
            raise ValueError('Invalid archived inference proof.')
        result.setdefault(model, []).append(StoredSegment(record))
    return result
