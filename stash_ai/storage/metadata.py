"""Small validation metadata for the single active inference record."""
from contextlib import contextmanager
import json

FORMAT = 4
META_KEYS = frozenset((
    'origin', 'inference_signature', 'frame_input_signature', 'analysis_format', 'scene_threshold', 'marker_threshold',
    'scene_frames_format', 'scene_score_floor', 'scene_support_floor', 'marker_score_floor', 'marker_tag_blocklist',
))


def summarize(record):
    return dict(mode=record['mode'], meta={key: value for key, value in record['meta'].items() if key in META_KEYS},
                has_frames='frames' in record, populated=bool(record['items']))


def encode(record):
    return json.dumps(record, ensure_ascii=False, sort_keys=True, separators=(',', ':'), allow_nan=False)


def decode(data):
    record = json.loads(data)
    required = {'mode', 'meta', 'has_frames', 'populated'}
    if (not isinstance(record, dict) or set(record) != required
            or record['mode'] not in ('scene', 'frames') or not isinstance(record['meta'], dict)
            or record['meta'].keys() - META_KEYS or type(record['has_frames']) is not bool
            or type(record['populated']) is not bool):
        raise ValueError('Invalid result metadata.')
    return record


@contextmanager
def backfill_batch(store):
    if store.connection.in_transaction:
        yield
    else:
        with store.connection:
            store.connection.execute('BEGIN')
            yield
