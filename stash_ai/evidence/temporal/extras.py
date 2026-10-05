"""Qualify extra labels over compatible observations and full interval unions."""
from collections import defaultdict
from .policy import TemporalPolicy
from stash_ai.evidence.interval_geometry import append_interval


def longest_extra_run(evidence):
    """Union consecutive index runs only within the same verified timeline."""
    groups = defaultdict(list)
    for item in evidence:
        if item.get('run_group') is not None:
            groups[item['run_group']].extend(item.get('runs', []))
    longest = 0
    for runs in groups.values():
        current = None
        for first, last, start, end in sorted(runs):
            if current is not None and first <= current[1] and start <= current[3] + .001:
                current[1], current[3] = max(current[1], last), max(current[3], end)
            else:
                current = [first, last, start, end]
            longest = max(longest, current[1] - current[0])
    return longest


def qualified_extras(evidence, duration, policy):
    accepted = []
    for label, item in evidence.items():
        hits = item['hits']
        fraction = min(1.0, sum(b-a for a,b in item['intervals']) / duration) if duration > 0 else 0
        short = policy.short_marker_extras and longest_extra_run([item]) >= policy.extra_short_min_hits
        if hits >= policy.extra_min_hits and (fraction + 1e-12 >= policy.extra_min_fraction or short):
            accepted.append((label, hits, fraction))
    return sorted(accepted, key=lambda item: (-item[2], -item[1], item[0]))


def merged_extras(component):
    """Qualify extras over the whole union; duplicate model coverage counts once."""
    records = component["contributors"]
    from stash_ai.evidence.formats import current
    for record in records:
        if current(record.get('meta', {})) and record.get('derived') != 'occurrence' and (
                'extra_support' not in record or not record.get('build_options')):
            raise ValueError('Current marker contributions require temporal extra_support evidence.')
    settings = next((r["build_options"]["policy"] for r in records if r.get("build_options")), None)
    if settings is None:
        return
    from stash_ai.evidence.historical_markers import HistoricalPolicy, frozen_policy
    options = next(r['build_options'] for r in records if r.get('build_options'))
    policy = frozen_policy(options)
    if isinstance(policy, HistoricalPolicy):
        return
    known, legacy, intervals, times = set(), set(), defaultdict(list), defaultdict(set)
    source_evidence = defaultdict(lambda: defaultdict(list))
    support_records = defaultdict(list)
    seen_support = set()
    for record in records:
        if "extra_support" not in record:
            from stash_ai.evidence.historical_evidence import plain_extras
            legacy.update(plain_extras(record))
            continue
        for tag, evidence in record["extra_support"].items():
            # Several source spans can share exactly the same indexed proof.
            # Its intervals, sample times and source records are set-like data.
            identity = (tag, id(evidence))
            if identity in seen_support:
                continue
            seen_support.add(identity)
            known.add(tag)
            intervals[tag].extend(evidence["intervals"])
            times[tag].update(evidence["sample_times"])
            support_records[tag].append(evidence)
            for source, values in evidence.get("sources", {}).items():
                source_evidence[tag][source].append(values)
    duration = component["end"] - component["start"] if component["end"] is not None else 0.0
    accepted, statistics = set(legacy), {}
    def bounded_ranges(intervals):
        ranges = []
        for start, end in sorted(intervals):
            start, end = max(start, component["start"]), min(end, component["end"])
            if end > start:
                append_interval(ranges, start, end)
        return ranges

    for tag in sorted(known - {component["tag"]}):
        ranges = bounded_ranges(intervals[tag])
        seconds = sum(end-start for start,end in ranges)
        fraction = min(1.0, seconds / duration) if duration > 0 else 0.0
        # Each source retains only enough distinct sample times to prove min_hits.
        # This prevents a frame-by-primary-by-extra tensor in the result cache.
        hits_at_least = sum(component["start"] <= time < component["end"] for time in times[tag])
        run_hits = longest_extra_run(support_records[tag]) if policy.short_marker_extras else 0
        coverage = fraction + 1e-12 >= policy.extra_min_fraction
        short = policy.short_marker_extras and run_hits >= policy.extra_short_min_hits
        qualifies = hits_at_least >= policy.extra_min_hits and (coverage or short)
        if qualifies:
            accepted.add(tag)
        statistics[tag] = dict(support_seconds=seconds, coverage_fraction=fraction,
                               hits_at_least=hits_at_least, qualified=qualifies)
        if policy.short_marker_extras:
            statistics[tag].update(longest_run_hits=run_hits,
                                  qualification=('coverage' if coverage else 'short_episode') if qualifies else None)
        if source_evidence[tag]:
            statistics[tag]["source_evidence"] = {}
            for source, records in sorted(source_evidence[tag].items()):
                own_ranges = bounded_ranges([interval for record in records for interval in record["intervals"]])
                own_times = {time for record in records for time in record["sample_times"]
                             if component["start"] <= time < component["end"]}
                if own_ranges:
                    statistics[tag]["source_evidence"][source] = dict(
                        score=max(record["score"] for record in records),
                        support_seconds=sum(b-a for a,b in own_ranges), hits_at_least=len(own_times),
                        signatures=sorted({signature for record in records for signature in record["signatures"]}))
    component["extra_tags"] = sorted(accepted - {component["tag"]})
    component["extra_evidence"] = statistics
