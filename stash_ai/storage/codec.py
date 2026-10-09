"""Stable compressed JSON encoding for persisted application state."""
import json
import zlib
from contextlib import contextmanager
from contextvars import ContextVar

from stash_ai.runtime.timing import stage

_decoded = ContextVar('decoded_state_check_cache', default=None)


@contextmanager
def decoded_cache(limit=32 * 2**20):
    """Bounded validated-read sharing, scoped only to a deep consistency check.

    Checkers borrow decoded JSON and must copy any mutable top-level overlay.
    Normal runtime reads retain their existing detached ownership contract.
    """
    from stash_ai.runtime.cache import RamCache
    cache = RamCache(limit)
    token = _decoded.set(cache)
    try:
        yield cache
    finally:
        _decoded.reset(token)
        cache.clear()


def validate_cached(value, contract, validate):
    """Reuse successful validation only while this check owns the read payload."""
    cache = _decoded.get()
    key = ('validated', contract, id(value))
    if cache is not None and cache.get(key) is value:
        return
    validate(value)
    if cache is not None:
        cache.put(key, value)  # Retaining the object also prevents id reuse.


def compress(raw, report=None, *, stage_name='state_compress'):
    """Keep the existing zlib format while prioritizing inference throughput."""
    with stage(report, stage_name):
        return zlib.compress(raw, level=zlib.Z_BEST_SPEED)


def encode(value, report=None, *, stage_prefix='state'):
    with stage(report, f'{stage_prefix}_json_encode'):
        raw = json.dumps(value, ensure_ascii=False, allow_nan=False,
                         sort_keys=True, separators=(",", ":")).encode("utf-8")
    result = compress(raw, report, stage_name=f'{stage_prefix}_compress')
    if report is not None:
        report.record_counts(**{f'{stage_prefix}_json_bytes_written': len(raw),
                                f'{stage_prefix}_blob_bytes_written': len(result)})
    return result


def decode(value, report=None):
    cache = _decoded.get()
    if cache is not None:
        saved = cache.get(value)
        if saved is not None:
            return saved
    with stage(report, 'state_decompress'):
        raw = zlib.decompress(value)
    with stage(report, 'state_json_decode'):
        result = json.loads(raw)
    if report is not None:
        report.record_counts(state_json_bytes_read=len(raw), state_blob_bytes_read=len(value))
    if cache is not None:
        cache.put(value, result)
    return result
