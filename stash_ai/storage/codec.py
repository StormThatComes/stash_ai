"""Stable compressed JSON encoding for persisted application state."""
import json
import zlib

from stash_ai.runtime.timing import stage


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
    with stage(report, 'state_decompress'):
        raw = zlib.decompress(value)
    with stage(report, 'state_json_decode'):
        result = json.loads(raw)
    if report is not None:
        report.record_counts(state_json_bytes_read=len(raw), state_blob_bytes_read=len(value))
    return result
