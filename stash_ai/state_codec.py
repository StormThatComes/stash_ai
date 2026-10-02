"""Stable compressed JSON encoding for persisted application state."""
import json
import zlib

from stash_ai.runtime import stage
from stash_ai.state_snapshot import FORMAT_KEY, expand


def encode(value, report=None):
    with stage(report, 'state_json_encode'):
        raw = json.dumps(value, ensure_ascii=False, allow_nan=False,
                         sort_keys=True, separators=(",", ":")).encode("utf-8")
    with stage(report, 'state_compress'):
        result = zlib.compress(raw)
    if report is not None:
        report.record_counts(state_json_bytes_written=len(raw), state_blob_bytes_written=len(result))
    return result


def decode(value, report=None):
    with stage(report, 'state_decompress'):
        raw = zlib.decompress(value)
    with stage(report, 'state_json_decode'):
        result = json.loads(raw)
    if isinstance(result, dict) and FORMAT_KEY in result:
        with stage(report, 'state_snapshot_expand'):
            result = expand(result)
    if report is not None:
        report.record_counts(state_json_bytes_read=len(raw), state_blob_bytes_read=len(value))
    return result
