"""Shared marker qualification, source-aware enrichment and clipped replay."""
from bisect import bisect_right
from collections import defaultdict
from copy import deepcopy
from contextlib import closing
from stash_ai.models import SOURCES
from stash_ai.evidence.interval_geometry import append_interval
from .frames import make_groups, observations
from .observations import ObservationIndex


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
        signature = own["signature"]
        source_evidence = {source: dict(evidence)
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
                    dict(score=evidence["score"], hits=0, strong_hits=0, intervals=[], signature=evidence["signature"]))
                own["score"] = max(own["score"], evidence["score"])
                own["hits"] += 1
                own["strong_hits"] += int(evidence["strong"])
                append_interval(own["intervals"], start, supported_end)
    for tag in list(active):
        finish(tag)
    return finished


def collect_extras(app, span, index: ObservationIndex, allowed=None):
    """Collect high detections inside this contribution, including extra-only models."""
    policy = app.TEMPORAL_POLICY
    support = {tag: value for tag, value in index.support(span['start'], span['end'], policy, allowed).items() if tag != span['tag']}
    span["extra_support"] = support
    span["frame_context"] = index.context
    # Final qualification is performed over the full union by merged_extras.
    extras = [item for item in index.qualified(span['start'], span['end'], policy, allowed) if item[0] != span['tag']]
    span["extra_tags"] = [item[0] for item in extras]
    span["extra_evidence"] = extras


def enrich(app, policies, spans, index, metrics=None):
    """Add missing shared intervals and attach source-aware extras from compatible inputs."""
    from stash_ai.evidence.temporal.policy import build_options
    policy = app.TEMPORAL_POLICY
    if not (policy.shared_marker_frames or policy.shared_marker_extras):
        return spans
    groups, by_signature = index.groups, index.by_signature
    options = {**build_options(app, {source: record for group in index.groups.values() for source, record, _ in group['members']}), "core_exclusions": {k: sorted(v) for k, v in policies.items()},
               "tag_exclusions": {k: sorted(v) for k, v in index.tag_policies.items()}}
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
            collect_extras(app, span, index.get(key), None if policy.shared_marker_extras else span['support_sources'])
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
                signature=samples[0][1]["signature"])
            hits.update(event["start"] for event, _ in samples)
            strong_hits.update(event["start"] for event, item in samples if item["strong"])
        sources = [source for source in SOURCES if source in evidence]
        if not sources:
            return False
        anchor = sources[0]
        own = evidence[anchor]
        signature = own["signature"]
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
        with closing(ObservationIndex(events, group['context'], key)) as extras_index:
            collect_extras(app, span, extras_index)
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
