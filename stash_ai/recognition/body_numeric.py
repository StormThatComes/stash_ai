"""Exact ordered body/face reductions, reusable across independent trials."""
import numpy as np
from .run_cache import current, MISSING
from .solver_shared import current as shared_current, key as shared_key


def vector_digest(value):
    from hashlib import sha256
    cache = current()
    owner=value
    while isinstance(owner.base,np.ndarray):owner=owner.base
    if cache is None or value.flags.writeable or not isinstance(owner.base, bytes):
        return sha256(value.tobytes()).digest()
    # Retain the exact immutable view, including its offset/strides. Slices of
    # one byte owner can have equal shapes and different contents.
    key = ('vector-sha', id(value), value.dtype.str, value.shape, value.strides)
    result = cache.get(key)
    if result is MISSING:
        result = sha256(value.tobytes()).digest()
        cache.put(key, (value,result), len(owner.base)+256)
        return result
    return result[1]


def domain(row):
    from .run_cache import metadata
    if 'embedding_foreground' in row.keys():return row['embedding_foreground']
    return metadata(row['quality']).get('quality',{}).get('embedding_foreground') if 'quality' in row.keys() else None


def body_similarity(scores,left,right):
    value=scores.face_similarity(left['_vector'],right['_vector'])
    penalty=getattr(scores,'mask_penalty',0.)
    if penalty:
        a,b=domain(left),domain(right)
        if a is not None and b is not None and a!=b:value-=penalty
    return value


def lookup(left, right, kind):
    cache, shared = current(), shared_current()
    arrays = tuple(left) + tuple(right)
    extra = (kind, len(left), len(right))
    signature = None
    key = ('body-exact', extra, tuple((v.dtype.str, v.shape, vector_digest(v)) for v in arrays)) if cache else None
    found = cache.get(key) if cache else MISSING
    # Files/ledger/checksums cost more than a local float32 dot. Reserve the
    # shared cache for complete reductions with meaningful recomputation work.
    work = len(left)*len(right)
    profitable = work >= 256 and work*(len(arrays[0]) if arrays else 0) >= 1024**2
    from .derived_cache import current as derived
    from .config import digest
    store=derived() if profitable else None
    durable=None
    if found is MISSING and store:
        durable=digest([extra,[(v.dtype.str,v.shape,vector_digest(v).hex()) for v in arrays]])
        result=store.get('body-witness',durable)
        if result is not None:found=tuple(result)
    if found is MISSING and shared and profitable:
        signature = shared_key('ordered-witness-digests-f32-v2',
                               [np.frombuffer(vector_digest(v),dtype=np.uint8) for v in arrays],
                               (extra,[(v.dtype.str,v.shape) for v in arrays]))
        value = shared.get(signature)
        if value is not None:
            found = tuple(float(v) for v in value)
            if cache and key is not None:
                cache.put(key,found,sum(len(v[2])+256 for v in key[2])+256)
    return (key, signature, durable), None if found is MISSING else found


def publish(identity, result):
    key, signature, durable = identity
    shared, cache = shared_current(), current()
    if shared and signature is not None:
        result = tuple(float(v) for v in shared.put(signature, np.asarray(result, dtype='<f8')))
    if cache and key is not None:
        cache.put(key, result, sum(len(v[2])+256 for v in key[2])+256)
    if durable is not None:
        from .derived_cache import current as derived
        derived().put('body-witness',durable,result)
    return result


def dot(first, second, charge, kind='body-dot'):
    # A scalar pair is below the shared/durable cache admission threshold.
    # Scene workers already retain numerical results in their local Scores.
    charge()
    return float(np.dot(first, second))
