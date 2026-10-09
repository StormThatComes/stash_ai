"""Shared bounded ordering for identical immutable frames and source context."""
from stash_ai.runtime.cache import RamCache
from stash_ai.storage.segments import FrozenList


class FrameMemo:
    def __init__(self, cache=None):
        self.cache = cache if cache is not None else RamCache(2 * 2**20)
        self.token = object()
        self.sorts = 0

    def ordered(self, tags, source, floor, exclusions, validate):
        # Only immutable payloads may retain successful validation/order.
        if not isinstance(tags, FrozenList):
            return _ordered(tags, validate, floor=floor, exclusions=exclusions)
        key = ('frame-order', self.token, source, id(tags), floor, frozenset(exclusions), validate)
        entry = self.cache.get(key)
        if entry is not None and entry[0] is tags:
            return entry[1]
        self.sorts += 1
        values = tuple(_ordered(tags, validate, floor=floor, exclusions=exclusions))
        self.cache.put(key, (tags, values))  # retain tags to prevent id recycling
        return values

    def close(self):
        self.cache.discard_matching(lambda key: isinstance(key, tuple) and
                                    key[:2] == ('frame-order', self.token))


def _ordered(tags, validate, *, floor=0, exclusions=()):
    from .labels import raw_label
    values = []
    for tag, score in tags:
        if tag in exclusions:
            continue
        label = raw_label(tag) if validate else tag
        if score >= floor:
            values.append((label, score))
    return sorted(values, key=lambda item: (-item[1], item[0]))


def eligible(tags, floor, exclusions=(), limit=0, *, validate=False, memo=None, source=None):
    # Filter before sorting, including untrusted/non-memoized external frames.
    # The memo key describes the entire qualification represented by its order.
    values = (memo.ordered(tags, source, floor, exclusions, validate) if memo is not None
              else _ordered(tags, validate, floor=floor, exclusions=exclusions))
    return list(values[:limit] if limit else values)


def selections(tags, high, floor, cores=(), tags_excluded=(), limit=0, *, validate=False, memo=None, source=None):
    ordered = eligible(tags, min(high, floor), tags_excluded, validate=validate, memo=memo, source=source)
    primary = [(tag, score) for tag, score in ordered if score >= floor and tag not in cores]
    extras = [(tag, score) for tag, score in ordered if score >= high]
    return (primary[:limit], extras[:limit]) if limit else (primary, extras)
