"""List-shaped results and JSON threshold arrays, used only by offline upgrades."""
import json
from stash_ai.storage.archive import pinned
from stash_ai.evidence.identity import marker_segment_signature

THRESHOLD_KEYS = {
    'scene': ('scene_threshold_override', 'scene_threshold'),
    'markers': ('marker_threshold_override', 'marker_threshold'),
}


def decode_thresholds(data):
    from stash_ai.evidence.thresholds import _score
    if type(data) in (int, float):
        if not _score(data):
            raise ValueError('Invalid result thresholds; run --db-check.')
        return [data]
    values = json.loads(data)
    if not isinstance(values, list) or any(value is not None and not _score(value) for value in values):
        raise ValueError('Invalid result thresholds; run --db-check.')
    return values


def encode_thresholds(values):
    data = json.dumps(values, allow_nan=False, separators=(',', ':'))
    decode_thresholds(data)
    return data


def overlay(segments, kind, thresholds):
    """Expose transient overrides to domain code, never persist them in evidence."""
    if thresholds is None:
        return segments
    values = decode_thresholds(thresholds)
    if len(values) != len(segments):
        raise ValueError('Result thresholds do not match segment history; run --db-check.')
    key, meta_key = THRESHOLD_KEYS[kind]
    for segment, threshold in zip(segments, values):
        segment.pop(key, None)
        if threshold is not None and threshold != segment['meta'].get(meta_key):
            segment[key] = threshold
    return segments


def separate(segments, kind):
    """Return immutable evidence and its per-segment threshold policy."""
    key, _ = THRESHOLD_KEYS[kind]
    other = THRESHOLD_KEYS['markers' if kind == 'scene' else 'scene'][0]
    if any(other in segment for segment in segments):
        raise ValueError('Threshold override does not match result kind.')
    values = [segment.get(key) for segment in segments]
    raw = [{field: value for field, value in segment.items() if field != key} for segment in segments]
    return raw, encode_thresholds(values) if any(value is not None for value in values) else None

FORMAT = 3
META_KEYS = frozenset((
    'origin', 'inference_signature', 'frame_input_signature', 'analysis_format', 'scene_threshold', 'marker_threshold',
    'scene_frames_format', 'scene_score_floor', 'scene_support_floor',
    'marker_score_floor', 'marker_tag_blocklist',
))


def summarize(segments):
    """Keep every historical segment, including missing validation fields."""
    return [dict(mode=segment['mode'],
                 meta={key: value for key, value in segment['meta'].items() if key in META_KEYS},
                 has_frames='frames' in segment, populated=bool(segment['items']))
            for segment in segments]


def encode_metadata(segments):
    return json.dumps(segments, ensure_ascii=False, sort_keys=True,
                      separators=(',', ':'), allow_nan=False)


def decode_metadata(data):
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



def index_result(store, scene, model, kind, segments, scope=None, *, metadata_only=False):
    """Index a validated effective view, including a frame-free threshold plan."""
    from stash_ai.evidence.thresholds import indexed_signature
    inference = next((segment for segment in reversed(segments)
                      if segment["meta"].get("origin") == "inference"), None)
    store.connection.execute("INSERT INTO result_index VALUES (?,?,?,?,?,?,?) "
        "ON CONFLICT(scope,scene,model,kind) DO UPDATE SET inference=excluded.inference,"
        "populated=excluded.populated,signature=excluded.signature",
        (scope or store.scope, str(scene), model, kind,
         int(inference is not None),
         int(any(s["populated"] if metadata_only else bool(s["items"]) for s in segments)),
         indexed_signature(inference) if inference is not None else None))


def retain(store, scene, model, kind, previous, active):
    """Called in the result writer's transaction before replacing active evidence."""
    active_signatures = {marker_segment_signature(segment) for segment in active}
    retired = [segment for segment in previous if segment['meta'].get('origin') == 'inference'
               and marker_segment_signature(segment) not in active_signatures]
    if not retired:
        return
    required = pinned(store, scene)
    for segment in retired:
        signature = marker_segment_signature(segment)
        if (model, signature) in required:
            # Threshold views never enter proof storage or proof identity.
            raw = {key: value for key, value in segment.items()
                   if key not in ('scene_threshold_override', 'marker_threshold_override')}
            store.connection.execute('INSERT OR IGNORE INTO inference_archive VALUES (?,?,?,?,?,?)',
                                     (store.scope, str(scene), model, kind, signature, store.pack(raw)))


def active_segments(segments):
    """Reduce old history before obsolete non-inference records are retired."""
    latest = next((index for index in range(len(segments) - 1, -1, -1)
                   if segments[index]['meta'].get('origin') == 'inference'), None)
    return [segment for index, segment in enumerate(segments)
            if segment['meta'].get('origin') != 'inference' or index == latest]



def history(blob):
    from stash_ai.storage.codec import decode
    value = decode(blob)
    return [value] if isinstance(value, dict) else value


def threshold_column(db):
    return 'threshold' if 'threshold' in {row[1] for row in db.execute('PRAGMA table_info(result_thresholds)')} else 'thresholds'
