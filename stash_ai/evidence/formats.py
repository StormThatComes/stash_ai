"""The active inference contract. Archived publication proofs are separate."""
import math

ANALYSIS_FORMAT = 5


def current(meta):
    return type(meta.get('analysis_format')) is int and meta['analysis_format'] == ANALYSIS_FORMAT


def validate_current(segment):
    """Reject obsolete inference rather than guessing its missing provenance."""
    if not isinstance(segment, dict) or not isinstance(segment.get('meta'), dict):
        raise ValueError('Invalid active inference record; run --db-update.')
    meta = segment['meta']
    if meta.get('origin') != 'inference':
        raise ValueError('Active results require current inference evidence; run --db-update and inference again.')
    if not current(meta):
        raise ValueError('Obsolete active inference evidence; run --db-update and inference again.')
    if (segment.get('mode') not in ('scene', 'frames')
            or any(not isinstance(meta.get(key), str) or not meta[key]
                   for key in ('inference_signature', 'frame_input_signature'))):
        raise ValueError('Current inference evidence requires both input and inference signatures.')
    if 'marker_tag_blocklist' in meta:
        raise ValueError('Current inference evidence cannot contain the retired marker blocklist.')
    required_scores = ['marker_score_floor', 'scene_support_floor', 'scene_threshold', 'marker_threshold']
    if segment['mode'] == 'scene':
        required_scores.append('scene_score_floor')
        has_frames = 'frames' in segment if 'items' in segment else segment.get('has_frames', False)
        if not has_frames:
            raise ValueError('Current scene evidence requires retained scene frames.')
        for item in segment.get('items', ()):
            if (type(item.get('score')) not in (int, float) or not math.isfinite(item['score'])
                    or not 0 <= item['score'] <= 1 or 'statistics' not in item
                    or (item['statistics'] is not None and not isinstance(item['statistics'], dict))):
                raise ValueError('Current scene evidence requires raw scores and explicit statistics.')
            validate_statistics(item['statistics'])
    if any(type(meta.get(key)) not in (int, float) or not math.isfinite(meta[key])
           or not 0 <= meta[key] <= 1 for key in required_scores):
        raise ValueError('Current inference evidence requires publication thresholds and retention floors.')


STATISTIC_FIELDS = frozenset(('hits', 'high_hits', 'max_score', 'sampled_frames',
    'sampled_seconds', 'support_seconds', 'sample_fraction', 'coverage_fraction',
    'longest_hits', 'longest_seconds', 'peak_time'))


def validate_statistics(stats):
    """Null is retained negative evidence; a positive statistic is complete."""
    if stats is None:
        return
    if (not isinstance(stats, dict) or set(stats) != STATISTIC_FIELDS
            or any(type(stats[key]) is not int or stats[key] < 0
                   for key in ('hits', 'high_hits', 'longest_hits', 'sampled_frames'))
            or any(type(stats[key]) not in (int, float) or not math.isfinite(stats[key]) or stats[key] < 0
                   for key in STATISTIC_FIELDS - {'hits', 'high_hits', 'longest_hits', 'sampled_frames'})
            or any(stats[key] > 1 for key in ('max_score', 'sample_fraction', 'coverage_fraction'))
            or stats['high_hits'] > stats['hits'] or stats['longest_hits'] > stats['hits']
            or stats['hits'] > stats['sampled_frames']):
        raise ValueError('Current scene statistics must be complete and internally consistent; run inference.')


def validate_archived(segment):
    """Validate pinned proof storage, including retired pre-v5 inference."""
    version = segment['meta'].get('analysis_format')
    if version is not None and (type(version) is not int or not 1 <= version <= ANALYSIS_FORMAT):
        raise ValueError('Unsupported archived inference evidence format.')
    if current(segment['meta']):
        validate_current(segment)
