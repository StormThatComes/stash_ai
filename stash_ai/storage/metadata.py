"""Small derived validation records; frame evidence remains in results.data."""
from contextlib import contextmanager
import json

FORMAT = 2
META_KEYS = frozenset((
    'origin', 'inference_signature', 'scene_threshold', 'marker_threshold',
    'scene_frames_format', 'scene_score_floor', 'scene_support_floor',
    'marker_score_floor', 'marker_tag_blocklist',
))


def summarize(segments):
    """Keep every historical segment, including missing validation fields."""
    return [dict(mode=segment['mode'],
                 meta={key: value for key, value in segment['meta'].items() if key in META_KEYS},
                 has_frames='frames' in segment, populated=bool(segment['items']))
            for segment in segments]


def encode(segments):
    return json.dumps(segments, ensure_ascii=False, sort_keys=True,
                      separators=(',', ':'), allow_nan=False)


def decode(data):
    segments = json.loads(data)
    if not isinstance(segments, list):
        raise ValueError('Invalid result metadata.')
    required = {'mode', 'meta', 'has_frames', 'populated'}
    for segment in segments:
        if (not isinstance(segment, dict) or not required <= segment.keys()
                or segment.keys() - required
                or not isinstance(segment['mode'], str)
                or not isinstance(segment['meta'], dict)
                or segment['meta'].keys() - META_KEYS
                or type(segment['has_frames']) is not bool
                or type(segment['populated']) is not bool):
            raise ValueError('Invalid result metadata.')
    return segments


@contextmanager
def backfill_batch(store):
    """One transaction for legacy metadata, preserving any caller transaction."""
    if store.connection.in_transaction:
        yield
    else:
        with store.connection:
            store.connection.execute('BEGIN')
            yield
