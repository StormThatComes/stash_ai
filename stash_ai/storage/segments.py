"""Deeply immutable borrowed JSON; detached public copies remain mutable."""
from copy import deepcopy
from stash_ai.signatures import digest


def _immutable(*args, **kwargs):
    raise TypeError('Borrowed evidence is immutable; request a detached copy.')


def freeze(value):
    if isinstance(value, (FrozenDict, FrozenList)):
        return value
    if isinstance(value, dict):
        return FrozenDict(value)
    if isinstance(value, list):
        return FrozenList(value)
    if isinstance(value, tuple):
        return tuple(freeze(item) for item in value)
    return value


class FrozenDict(dict):
    __slots__ = ('_frozen',)
    def __init__(self, *args, **kwargs):
        # Reinitializing an existing borrowed object is another mutation.
        if hasattr(self, '_frozen'):
            _immutable()
        dict.__init__(self, ((key, freeze(value)) for key, value in dict(*args, **kwargs).items()))
        self._frozen = True

    __setitem__ = __delitem__ = clear = pop = popitem = setdefault = update = __ior__ = _immutable

    def __deepcopy__(self, memo):
        result = {}
        memo[id(self)] = result
        result.update((deepcopy(k, memo), deepcopy(v, memo)) for k, v in self.items())
        return result

    def __reduce__(self):
        return type(self), (dict(self),)


class FrozenList(list):
    __slots__ = ('_frozen',)
    def __init__(self, values=()):
        if hasattr(self, '_frozen'):
            _immutable()
        list.__init__(self, (freeze(value) for value in values))
        self._frozen = True

    __setitem__ = __delitem__ = append = extend = insert = pop = remove = clear = reverse = sort = __iadd__ = __imul__ = _immutable

    def __deepcopy__(self, memo):
        result = []
        memo[id(self)] = result
        result.extend(deepcopy(value, memo) for value in self)
        return result

    def __reduce__(self):
        return type(self), (list(self),)


class StoredSegment(FrozenDict):
    """Borrowed immutable record; private memo attributes never enter JSON."""
    __slots__ = ('_digest', '_frame_group', '_evidence_digest', '_scene_projected', '_validated_current')


def segment_digest(segment):
    if isinstance(segment, StoredSegment):
        if not hasattr(segment, '_digest'):
            segment._digest = digest(segment)
        return segment._digest
    return digest(segment)


def frame_group(segment):
    if isinstance(segment, StoredSegment) and hasattr(segment, '_frame_group'):
        return segment._frame_group
    value = digest({'input': segment['meta']['frame_input_signature'],
        'times': [[float(row['start']), float(row['end'])] for row in segment['items']]})
    if isinstance(segment, StoredSegment):
        segment._frame_group = value
    return value
