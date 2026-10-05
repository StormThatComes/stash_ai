"""Historical payload builders used only to inject unsupported/corrupt test data."""
import json
from stash_ai.evidence.identity import marker_segment_signature as current_signature


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


def index_result(store, scene, model, kind, segments, scope=None, *, metadata_only=False):
    """Index a validated effective view, including a frame-free threshold plan."""
    from stash_ai.evidence.thresholds import indexed_signature
    inference = next((segment for segment in reversed(segments)
                      if segment["meta"].get("origin") == "inference"), None)
    store.connection.execute("INSERT INTO result_index VALUES (?,?,?,?,?,?) "
        "ON CONFLICT(scope,scene,model,kind) DO UPDATE SET "
        "populated=excluded.populated,signature=excluded.signature",
        (scope or store.scope, str(scene), model, kind,
         int(any(s["populated"] if metadata_only else bool(s["items"]) for s in segments)),
         indexed_signature(publication_record(inference, kind)) if inference is not None else None))


def marker_segment_signature(segment):
    return current_signature({k: v for k, v in segment.items()
                              if k not in ('scene_threshold_override', 'marker_threshold_override')})


def publication_record(segment, kind):
    from stash_ai.storage.thresholds import overlay as current_overlay
    key, _ = THRESHOLD_KEYS[kind]
    raw = {k: v for k, v in segment.items() if k not in ('scene_threshold_override', 'marker_threshold_override')}
    return current_overlay(raw, kind, segment.get(key))
