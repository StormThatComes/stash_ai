"""Reusable interval queries over compatible cached observations."""
from bisect import bisect_left
from dataclasses import replace
from stash_ai.evidence.interval_geometry import append_interval
from stash_ai.evidence.temporal.extras import qualified_extras


class ObservationIndex:
    """One timeline and reusable interval queries shared by all primary labels."""
    def __init__(self, events, context, group, cache=None):
        from stash_ai.runtime.cache import RamCache
        self.events, self.context, self.group = events, context, group
        self.starts = [event['start'] for event in events]
        self.cache = cache if cache is not None else RamCache(256 * 2**20)
        self.token = object()
        self.scans = 0
        self.timelines = {}
        self.cache_keys = set()
        self.disjoint = all(a['end'] <= b['start'] for a, b in zip(events, events[1:]))

    def support(self, start, end, policy, allowed=None):
        allowed = frozenset(allowed) if allowed is not None else None
        key = ('extras', self.token, start, end, policy.extra_min_hits, policy.short_marker_extras, allowed)
        cached = self.cache.get(key)
        if cached is not None:
            return cached
        self.scans += 1
        first, last = bisect_left(self.starts, start), bisect_left(self.starts, end)
        if last <= first:
            return {}
        if self.disjoint:
            from stash_ai.evidence.timeline import Timeline
            if allowed not in self.timelines:
                self.timelines[allowed] = Timeline(self.events, self.context, allowed)
            support = self.timelines[allowed].support(first, last, end, policy, self.group)
        else:
            support = self.scan_support(first, last, end, policy, allowed)
        from stash_ai.evidence.timeline import support_size
        self.cache.put(key, support, retained_bytes=support_size(support))
        self.cache_keys.add(key)
        return support

    def qualified(self, start, end, policy, allowed=None):
        allowed = frozenset(allowed) if allowed is not None else None
        key = ('qualified_extras', self.token, start, end, policy.extra_min_hits,
               policy.extra_min_fraction, policy.short_marker_extras, policy.extra_short_min_hits, allowed)
        cached = self.cache.get(key)
        if cached is None:
            cached = qualified_extras(self.support(start, end, policy, allowed), end-start, policy)
            self.cache.put(key, cached)
            self.cache_keys.add(key)
        return cached

    def close(self):
        # These token-scoped queries cannot be reused by a later scene/index.
        # Result spans retain the proofs they need; release the cache references.
        for key in self.cache_keys:
            self.cache.discard(key)
        self.cache_keys.clear()
        self.timelines.clear()

    def scan_support(self, first, last, end, policy, allowed):
        """Exact fallback for overlapping legacy/custom observation timelines."""
        support = {}
        for index in range(first, last):
            event = self.events[index]
            time, stop = event['start'], min(event['end'], end)
            for tag, values in event['extras'].items():
                sources = values if allowed is None else {s: score for s, score in values.items() if s in allowed}
                if not sources:
                    continue
                item = support.setdefault(tag, dict(intervals=[], sample_times=[], hits=0))
                item['hits'] += 1
                append_interval(item['intervals'], time, stop)
                if len(item['sample_times']) < policy.extra_min_hits:
                    item['sample_times'].append(time)
                if policy.short_marker_extras:
                    item['run_group'] = self.group
                    runs = item.setdefault('runs', [])
                    if runs and runs[-1][1] == index and time <= runs[-1][3] + .001:
                        runs[-1][1], runs[-1][3] = index + 1, max(runs[-1][3], stop)
                    else:
                        runs.append([index, index + 1, time, stop])
                for source, score in sources.items():
                    if source is None:
                        continue
                    own = item.setdefault('sources', {}).setdefault(source,
                        dict(intervals=[], sample_times=[], score=score, signatures=self.context[source]))
                    own['score'] = max(own['score'], score)
                    append_interval(own['intervals'], time, stop)
                    if len(own['sample_times']) < policy.extra_min_hits:
                        own['sample_times'].append(time)
        return support


def qualify_marker_extras(finished, extra_frames, policy, identity, extra_group, extra_cache):
    """Attach interval-query proofs only to retained marker candidates."""
    # Inspect every observation inside the finished interval, including a short
    # permitted gap in primary detections. Compute this only for retained markers.
    from stash_ai.signatures import digest
    events = {}
    for start, end, extras in extra_frames:
        event = events.setdefault(start, dict(start=start, end=end, extras={}))
        event['end'] = max(event['end'], end)
        for label, score in extras:
            event['extras'].setdefault(label, {})[None] = max(score, event['extras'].get(label, {}).get(None, score))
    index = ObservationIndex([events[t] for t in sorted(events)], {}, extra_group or digest(
        {'input': identity, 'frames': extra_frames}), extra_cache)
    qualified = []
    try:
        for marker in finished:
            evidence = {label: value for label, value in index.support(marker.start, marker.end, policy).items() if label != marker.tag}
            extras = [item for item in index.qualified(marker.start, marker.end, policy) if item[0] != marker.tag]
            qualified.append(replace(marker, extra_tags=tuple(item[0] for item in extras), extra_evidence=tuple(extras),
                                     extra_support=evidence))
    finally:
        index.close()
    return qualified
