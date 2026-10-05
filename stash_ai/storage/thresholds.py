"""One optional publication threshold, separate from immutable inference evidence."""
THRESHOLD_KEYS = {
    'scene': ('scene_threshold_override', 'scene_threshold'),
    'markers': ('marker_threshold_override', 'marker_threshold'),
}


def validate(value):
    from stash_ai.evidence.thresholds import _score
    if not _score(value):
        raise ValueError('Invalid result threshold; run --db-check.')
    return value


def overlay(record, kind, threshold):
    if threshold is not None:
        validate(threshold)
        key, meta_key = THRESHOLD_KEYS[kind]
        record.pop(key, None)
        if threshold != record['meta'][meta_key]:
            record[key] = threshold
    return record


def separate(record, kind):
    key, _ = THRESHOLD_KEYS[kind]
    other = THRESHOLD_KEYS['markers' if kind == 'scene' else 'scene'][0]
    if other in record:
        raise ValueError('Threshold override does not match result kind.')
    threshold = record.get(key)
    if threshold is not None:
        validate(threshold)
    return {field: value for field, value in record.items() if field != key}, threshold
