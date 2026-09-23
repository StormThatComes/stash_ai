"""Fuse cached VTT observations, retaining model scores and input provenance."""
from __future__ import annotations

from bisect import bisect_left, bisect_right
from collections import defaultdict
from copy import deepcopy

from stash_taggers import SOURCES
from stash_temporal import append_interval, marker_floor, qualified_extras


class ObservationIndex:
    """One timeline and reusable interval queries shared by all primary labels."""
    def __init__(self, events, context, group, cache=None):
        from stash_runtime import RamCache
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
            from stash_timeline import Timeline
            if allowed not in self.timelines:
                self.timelines[allowed] = Timeline(self.events, self.context, allowed)
            support = self.timelines[allowed].support(first, last, end, policy, self.group)
        else:
            support = self.scan_support(first, last, end, policy, allowed)
        from stash_timeline import support_size
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


class FrameIndex:
    """Verify groups once; observations are prepared lazily and reused."""
    def __init__(self, app, records, policies, signatures=None, cache=None, tag_policies=None):
        from stash_merge import marker_segment_signature
        from stash_exclusions import policies_or_empty
        self.app, self.policies, self.cache = app, policies, cache
        self.tag_policies = policies_or_empty(tag_policies)
        self.signatures = signatures if signatures is not None else {
            id(segment): marker_segment_signature(segment) for segments in records.values() for segment in segments}
        self.groups = make_groups(records, self.signatures)
        self.by_signature = {sig: key for key, group in self.groups.items() for _, _, sig in group['members']}
        self.indexes = {}

    def __enter__(self):
        return self

    def __exit__(self, *args):
        for index in self.indexes.values():
            index.close()

    def get(self, key):
        if key not in self.indexes:
            group = self.groups[key]
            self.indexes[key] = ObservationIndex(observations(self.app, group, self.policies, self.tag_policies), group['context'], key, self.cache)
        return self.indexes[key]


def make_groups(records, signatures=None):
    from stash_merge import marker_segment_signature
    from stash_progress import digest
    groups = {}
    for source in SOURCES:
        for segment in records.get(source, []):
            meta = segment["meta"]
            identity = meta.get("frame_input_signature")
            if segment["mode"] != "frames" or not identity or meta.get("marker_threshold") is None:
                continue
            # A signature plus the actual timeline: shifted/missing cues never
            # become matching observations merely because metadata was copied.
            from stash_performance import frame_group
            key = frame_group(segment)
            signature = signatures[id(segment)] if signatures is not None else marker_segment_signature(segment)
            group = groups.setdefault(key, {"members": [], "context": defaultdict(set)})
            if signature not in group["context"][source]:
                group["members"].append((source, segment, signature))
                group["context"][source].add(signature)
    for group in groups.values():
        group["members"].sort(key=lambda item: (list(SOURCES).index(item[0]), item[2]))
        group["context"] = {source: sorted(signatures) for source, signatures in group["context"].items()}
    return dict(sorted(groups.items()))


def observations(app, group, core_policies, tag_policies=None):
    """One event per timestamp; duplicate runs of a model count only once."""
    from stash_merge import raw_label
    from stash_exclusions import policies_or_empty
    tag_policies = policies_or_empty(tag_policies)
    times = {}
    for source, segment, signature in group["members"]:
        from stash_thresholds import marker_threshold
        high = marker_threshold(segment, getattr(app, "MARKER_THRESHOLDS", None))
        floor = marker_floor(app.TEMPORAL_POLICY, high)
        for row in segment["items"]:
            start, end = row["start"], row["end"]
            if start < 0 or end <= start:
                raise ValueError("Invalid cached VTT interval for shared observations.")
            event = times.setdefault(start, {"start": start, "end": end, "cores": {}, "extras": {}})
            event["end"] = max(event["end"], end)
            ordered = sorted(((raw_label(tag), score) for tag, score in row["tags"]
                              if tag not in app.MARKER_TAG_BLOCKLIST and tag not in tag_policies[source]),
                             key=lambda item: (-item[1], item[0]))
            cores = [(tag, score) for tag, score in ordered if score >= floor and tag not in core_policies[source]]
            extras = [(tag, score) for tag, score in ordered if score >= high]
            if app.MARKER_MAX_TAGS_PER_FRAME:
                cores = cores[:app.MARKER_MAX_TAGS_PER_FRAME]
                extras = extras[:app.MARKER_MAX_TAGS_PER_FRAME]
            for tag, score in cores:
                evidence = event["cores"].setdefault(tag, {}).setdefault(source,
                    {"score": score, "strong": False, "signatures": set()})
                evidence["score"] = max(evidence["score"], score)
                evidence["strong"] |= score >= high
                evidence["signatures"].add(signature)
            for tag, score in extras:
                sources = event["extras"].setdefault(tag, {})
                sources[source] = max(sources.get(source, score), score)
    return [times[start] for start in sorted(times)]


def shared_runs(app, events, group, key, options):
    """Qualify across models before minimum-hit rejection; never average scores."""
    policy = app.TEMPORAL_POLICY
    active, finished = {}, []
    by_signature = {signature: segment for _, segment, signature in group["members"]}

    def finish(tag):
        run = active.pop(tag)
        if run["hits"] < app.MARKER_MIN_HITS or len(run["sources"]) < 2:
            return
        sources = [source for source in SOURCES if source in run["sources"]]
        anchor = sources[0]
        own = run["sources"][anchor]
        signature = sorted(own["signatures"])[0]
        source_evidence = {source: {**evidence, "signatures": sorted(evidence["signatures"])}
                           for source, evidence in run["sources"].items()}
        finished.append(dict(tag=tag, start=run["start"], end=run["end"],
            source=anchor, score=own["score"], hits=own["hits"], strong_hits=own["strong_hits"],
            derived="shared_frames", shared_hits=run["hits"], shared_strong_hits=run["strong_hits"],
            source_evidence=source_evidence, support_sources=sources,
            support_context=group["context"], frame_context=group["context"], frame_group=key,
            evidence_signature=signature, meta=by_signature[signature]["meta"], build_options=options,
            extra_tags=[], extra_evidence=[], extra_support={}))

    for event in events:
        start, end = event["start"], event["end"]
        for tag, sources in event["cores"].items():
            if tag in active and start - active[tag]["end"] > app.MARKER_MERGE_GAP_SECONDS + 0.001:
                finish(tag)
            strong = any(evidence["strong"] for evidence in sources.values())
            seed = strong or len(sources) >= policy.consensus_min_models
            if tag not in active:
                if not seed:
                    continue
                active[tag] = dict(start=start, end=start, last_seed_end=end, hits=0, strong_hits=0, sources={})
            run = active[tag]
            if seed:
                run["last_seed_end"] = max(run["last_seed_end"], end)
            supported_end = min(end, run["last_seed_end"] + policy.marker_max_weak_seconds)
            if supported_end <= start:
                finish(tag)
                continue
            run["end"] = max(run["end"], supported_end)
            run["hits"] += 1
            run["strong_hits"] += int(strong)
            for source, evidence in sources.items():
                own = run["sources"].setdefault(source,
                    dict(score=evidence["score"], hits=0, strong_hits=0, intervals=[], signatures=set()))
                own["score"] = max(own["score"], evidence["score"])
                own["hits"] += 1
                own["strong_hits"] += int(evidence["strong"])
                own["signatures"].update(evidence["signatures"])
                append_interval(own["intervals"], start, supported_end)
    for tag in list(active):
        finish(tag)
    return finished


def collect_extras(app, span, events, group, allowed=None):
    """Collect high detections inside this contribution, including extra-only models."""
    policy = app.TEMPORAL_POLICY
    index = events if isinstance(events, ObservationIndex) else ObservationIndex(events, group['context'], span.get('frame_group'))
    support = {tag: value for tag, value in index.support(span['start'], span['end'], policy, allowed).items() if tag != span['tag']}
    span["extra_support"] = support
    span["frame_context"] = group["context"]
    # Final qualification is performed over the full union by merged_extras.
    extras = [item for item in index.qualified(span['start'], span['end'], policy, allowed) if item[0] != span['tag']]
    span["extra_tags"] = [item[0] for item in extras]
    span["extra_evidence"] = extras


def enrich(app, records, policies, spans, signatures=None, index=None, metrics=None):
    """Add missing shared intervals and attach source-aware extras from compatible inputs."""
    from stash_temporal import build_options
    policy = app.TEMPORAL_POLICY
    if not policy.enabled or not (policy.shared_marker_frames or policy.shared_marker_extras):
        return spans
    index = index if index is not None else FrameIndex(app, records, policies, signatures)
    groups, by_signature = index.groups, index.by_signature
    options = {**build_options(app), "core_exclusions": {k: sorted(v) for k, v in policies.items()},
               "tag_exclusions": {k: sorted(v) for k, v in index.tag_policies.items()},
               "blocklist": sorted(app.MARKER_TAG_BLOCKLIST)}
    covered = defaultdict(list)
    for span in spans:
        key = by_signature.get(span.get("evidence_signature"))
        if key:
            span["frame_group"] = key
        if span["end"] is not None:
            covered[span["tag"]].append((span["start"], span["end"]))
    for tag, intervals in covered.items():
        merged = []
        for start, end in sorted(intervals):
            append_interval(merged, start, end)
        covered[tag] = merged
    candidates = []
    if policy.shared_marker_frames:
        for key, group in groups.items():
            if len(group["context"]) < 2:
                continue
            for span in shared_runs(app, index.get(key).events, group, key, options):
                if not any(a <= span["start"] and b >= span["end"] for a,b in covered[span["tag"]]):
                    candidates.append(span)
        candidates.sort(key=lambda span: (-span["shared_strong_hits"], -span["shared_hits"],
                                         span["start"], span["tag"], span["frame_group"]))
        if app.MAX_MARKERS_PER_SCENE:
            candidates = candidates[:app.MAX_MARKERS_PER_SCENE]
        if metrics is not None:
            metrics['shared_marker_runs'] += len(candidates)
            metrics['shared_marker_extensions'] += sum(any(a < span['end'] and b > span['start']
                for a,b in covered[span['tag']]) for span in candidates)
        spans.extend(candidates)
    for span in spans:
        key = span.get("frame_group")
        if key and (policy.shared_marker_extras or span.get("derived") == "shared_frames"):
            # Even with cross-model extras disabled, a newly shared interval can
            # collect extras supplied by its own primary contributors.
            group = groups[key]
            collect_extras(app, span, index.get(key), group, None if policy.shared_marker_extras else span['support_sources'])
    return spans


def refresh_clipped_evidence(app, span, records, policies, tag_policies=None):
    """Recompute scores and sample proofs within cleanup's published bounds."""
    groups = make_groups(records)
    key = span.get("frame_group")
    if key not in groups:
        return True
    group = groups[key]
    events = observations(app, group, policies, tag_policies)
    if span.get("derived") == "shared_frames":
        evidence, hits, strong_hits = {}, set(), set()
        for source, own in span["source_evidence"].items():
            ranges = [[max(a, span["start"]), min(b, span["end"])] for a, b in own["intervals"]
                      if min(b, span["end"]) > max(a, span["start"])]
            samples = []
            ends = [b for _, b in ranges]
            for event in events:
                item = event["cores"].get(span["tag"], {}).get(source)
                index = bisect_right(ends, event["start"])
                if item is not None and index < len(ranges) and event["end"] > ranges[index][0]:
                    samples.append((event, item))
            if not samples:
                continue
            evidence[source] = dict(score=max(item["score"] for _, item in samples),
                hits=len(samples), strong_hits=sum(item["strong"] for _, item in samples), intervals=ranges,
                signatures=sorted({signature for _, item in samples for signature in item["signatures"]}))
            hits.update(event["start"] for event, _ in samples)
            strong_hits.update(event["start"] for event, item in samples if item["strong"])
        sources = [source for source in SOURCES if source in evidence]
        if not sources:
            return False
        anchor = sources[0]
        own = evidence[anchor]
        signature = own["signatures"][0]
        meta = next(segment["meta"] for _, segment, sig in group["members"] if sig == signature)
        span.update(source_evidence=evidence, support_sources=sources, source=anchor,
                    score=own["score"], hits=own["hits"], strong_hits=own["strong_hits"],
                    evidence_signature=signature, meta=meta, shared_hits=len(hits), shared_strong_hits=len(strong_hits))
    if app.TEMPORAL_POLICY.shared_marker_extras or span.get("derived") == "shared_frames":
        if not app.TEMPORAL_POLICY.shared_marker_extras:
            allowed = set(span["support_sources"])
            events = [{**event, "extras": {tag: {source: score for source, score in values.items() if source in allowed}
                      for tag, values in event["extras"].items() if any(source in allowed for source in values)}}
                      for event in events]
        collect_extras(app, span, events, group)
    return True


def prune_blocked_extras(span, blocked):
    if not blocked or not (blocked & (set(span.get('extra_tags', [])) | set(span.get('extra_support', {})) |
                                    {item[0] for item in span.get('extra_evidence', [])})):
        return span
    result = deepcopy(span)
    result["extra_tags"] = [tag for tag in result.get("extra_tags", []) if tag not in blocked]
    if "extra_support" in result:
        result["extra_support"] = {tag: evidence for tag, evidence in result["extra_support"].items() if tag not in blocked}
    result["extra_evidence"] = [item for item in result.get("extra_evidence", []) if item[0] not in blocked]
    return result
