"""One effective publication threshold, separate from captured provenance."""
PUBLICATION_THRESHOLD = 'publication_threshold'
THRESHOLD_KEYS = {'scene': 'scene_threshold', 'markers': 'marker_threshold'}


def validate(value):
    from stash_ai.evidence.thresholds import _score
    if not _score(value):
        raise ValueError('Invalid result threshold; run --db-check.')
    return value


def effective(record):
    key = 'scene_threshold' if record['mode'] == 'scene' else 'marker_threshold'
    return record.get(PUBLICATION_THRESHOLD, record['meta'][key])


def overlay(record, kind, threshold):
    record.pop(PUBLICATION_THRESHOLD, None)
    if threshold is not None and validate(threshold) != record['meta'][THRESHOLD_KEYS[kind]]:
        record[PUBLICATION_THRESHOLD] = threshold
    return record


def separate(record, kind):
    threshold = record.get(PUBLICATION_THRESHOLD)
    if threshold is not None:
        validate(threshold)
    return {k: v for k, v in record.items() if k != PUBLICATION_THRESHOLD}, threshold
