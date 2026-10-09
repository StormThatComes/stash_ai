"""One bounded read of authoritative provenance for cached-plan selection."""
from contextlib import contextmanager
from contextvars import ContextVar
from copy import deepcopy

_current = ContextVar('provenance_batch', default=None)


@contextmanager
def identity_batch(store, pairs):
    pairs = tuple(dict.fromkeys((str(scene), model) for scene, model in pairs))
    if len(pairs) > 128:
        raise ValueError('Identity batches are limited to 128 scene/model pairs.')
    blobs = {}
    if pairs:
        wanted = ','.join('(?,?)' for _ in pairs)
        rows = store.read_rows('WITH wanted(scene,model) AS (VALUES ' + wanted + ') '
            'SELECT i.scene,i.model,i.signature,i.data FROM wanted AS w CROSS JOIN ' +
            store.table('inference_identities') + ' AS i '
            'WHERE i.scope=? AND i.scene=w.scene AND i.model=w.model',
            (*(value for pair in pairs for value in pair), store.scope))
        blobs = {(scene, model, signature): data for scene, model, signature, data in rows}
    token = _current.set((store, frozenset(pairs), blobs, {}))
    try:
        yield
    finally:
        _current.reset(token)


def lookup(store, scene, model, signature, decode):
    batch = _current.get()
    if batch is None or batch[0] is not store or (str(scene), model) not in batch[1]:
        return None
    key = str(scene), model, signature
    if key not in batch[3]:
        blob = batch[2].get(key)
        batch[3][key] = decode(blob, signature) if blob is not None else (None, None)
    return (deepcopy(batch[3][key]),)
