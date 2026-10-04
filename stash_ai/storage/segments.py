"""Immutable borrowed segments with memoized digest and frame-group identity."""
from copy import deepcopy
from stash_ai.signatures import digest


class StoredSegment(dict):
    """Internally borrowed immutable record; private caches never enter JSON.

    Public detached copies are ordinary mutable dicts, without cached digests.
    """
    def __deepcopy__(self, memo):
        result = {}
        memo[id(self)] = result
        result.update((deepcopy(k, memo), deepcopy(v, memo)) for k, v in self.items())
        return result


def segment_digest(segment):
    if isinstance(segment, StoredSegment):
        if not hasattr(segment, '_digest'):
            segment._digest = digest(segment)
        return segment._digest
    return digest(segment)


def frame_group(segment):
    if isinstance(segment, StoredSegment) and hasattr(segment, '_frame_group'):
        return segment._frame_group
    value = digest({'input': segment['meta'].get('frame_input_signature'),
        'times': [[float(row['start']), float(row['end'])] for row in segment['items']]})
    if isinstance(segment, StoredSegment):
        segment._frame_group = value
    return value
