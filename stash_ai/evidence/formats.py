"""Current retained evidence contract; historical records keep their own rules."""
import math

ANALYSIS_FORMAT = 4


def current(meta):
    return type(meta.get('analysis_format')) is int and meta['analysis_format'] == ANALYSIS_FORMAT


def validate_current(segment):
    """Validate current provenance without inventing missing historical evidence."""
    meta = segment['meta']
    version = meta.get('analysis_format')
    if version is not None and (type(version) is not int or not 1 <= version <= ANALYSIS_FORMAT):
        raise ValueError('Unsupported inference evidence format; update stash_ai.')
    if not current(meta):
        return
    if (meta.get('origin') != 'inference' or segment['mode'] not in ('scene', 'frames')
            or any(not isinstance(meta.get(key), str) or not meta[key]
                   for key in ('inference_signature', 'frame_input_signature'))):
        raise ValueError('Current inference evidence requires both input and inference signatures.')
    if 'marker_tag_blocklist' in meta:
        raise ValueError('Current inference evidence cannot contain the retired marker blocklist.')
    required_scores = ['marker_score_floor', 'scene_threshold', 'marker_threshold']
    if segment['mode'] == 'scene':
        required_scores.append('scene_score_floor')
        has_frames = 'frames' in segment if 'items' in segment else segment.get('has_frames', False)
        if type(meta.get('scene_frames_format')) is not int or meta['scene_frames_format'] != 1 or not has_frames:
            raise ValueError('Current scene evidence requires retained scene frames.')
        for item in segment.get('items', ()):
            if (type(item.get('score')) not in (int, float) or not math.isfinite(item['score'])
                    or not 0 <= item['score'] <= 1 or 'statistics' not in item
                    or (item['statistics'] is not None and not isinstance(item['statistics'], dict))):
                raise ValueError('Current scene evidence requires raw scores and explicit statistics.')
    if any(type(meta.get(key)) not in (int, float) or not math.isfinite(meta[key])
           or not 0 <= meta[key] <= 1 for key in required_scores):
        raise ValueError('Current inference evidence requires publication thresholds and retention floors.')
