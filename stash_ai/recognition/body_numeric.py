"""Exact ordered body/face reductions, reusable across independent trials."""
import numpy as np
from .run_cache import current, MISSING
from .solver_shared import current as shared_current, key as shared_key


def vector_digest(value):
    from hashlib import sha256
    cache = current()
    if cache is None or value.flags.writeable or not isinstance(value.base, bytes):
        return sha256(value.tobytes()).digest()
    key = ('vector-sha', value.dtype.str, value.shape, value.base)
    result = cache.get(key)
    if result is MISSING:
        result = sha256(value.tobytes()).digest()
        cache.put(key, result, len(value.base)+256)
    return result


def lookup(left, right, kind):
    cache, shared = current(), shared_current()
    arrays = tuple(left) + tuple(right)
    extra = (kind, len(left), len(right))
    signature = None
    key = ('body-exact', extra, tuple((v.dtype.str, v.shape, v.tobytes()) for v in arrays)) if cache else None
    found = cache.get(key) if cache else MISSING
    # Files/ledger/checksums cost more than a local float32 dot. Reserve the
    # shared cache for complete reductions with meaningful recomputation work.
    work = len(left)*len(right)
    profitable = work >= 256 and work*(len(arrays[0]) if arrays else 0) >= 1024**2
    if found is MISSING and shared and profitable:
        signature = shared_key('ordered-witness-f32-v1', arrays, extra)
        value = shared.get(signature)
        if value is not None:
            found = tuple(float(v) for v in value)
            if cache and key is not None:
                cache.put(key,found,sum(len(v[2])+256 for v in key[2])+256)
    return (key, signature), None if found is MISSING else found


def publish(identity, result):
    key, signature = identity
    shared, cache = shared_current(), current()
    if shared and signature is not None:
        result = tuple(float(v) for v in shared.put(signature, np.asarray(result, dtype='<f8')))
    if cache and key is not None:
        cache.put(key, result, sum(len(v[2])+256 for v in key[2])+256)
    return result


def dot(first, second, charge, kind='body-dot'):
    identity, found = lookup((first,), (second,), kind)
    if found is None:
        charge()
        found = publish(identity, (float(np.dot(first, second)),))
    return found[0]
