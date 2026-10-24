"""One timestamp or repeated retained crop is one witness within a scan.

Only an explicit pixel fingerprint is deduplicated. Equal embeddings alone do
not imply equal pixels. Legacy evidence retains its timestamp-based semantics.
"""
from .run_cache import metadata


def unit(row):
    if 'quality' in row.keys():
        try:
            value = metadata(row['quality']).get('evidence_crop_sha256')
            if isinstance(value, str) and len(value) == 64 and all(c in '0123456789abcdef' for c in value):
                return row['scan'], 'pixels', value
        except (ValueError, TypeError, AttributeError):
            pass
    return row['scan'], 'time', row['seconds']
