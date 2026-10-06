"""Additional scene qualification using positive evidence already in the cache."""
from collections import defaultdict
from dataclasses import asdict

from stash_ai.models import SOURCES
from stash_ai.evidence.interval_geometry import append_interval


def _shared_members(group, scene_records):
    """Frames are derived directly from these current scene records."""
    members, context = [], {}
    for source, segment, signature in group['members']:
        meta = segment['meta']
        scores = {entry['tag']: [entry['score']] for entry in scene_records[source]['items']}
        context[source] = signature
        members.append((source, segment, signature, meta['scene_threshold'], meta['marker_score_floor'],
                        defaultdict(list, scores)))
    return members, context


def _shared_timeline(members, policies, policy):
    from stash_ai.evidence.labels import raw_label
    timeline = {}
    for source, segment, signature, threshold, _, aggregates in members:
        for row in segment['items']:
            start, end = float(row['start']), float(row['end'])
            if start < 0 or end <= start:
                raise ValueError('Invalid cached VTT interval for shared scene evidence.')
            event = timeline.setdefault(start, dict(start=start, end=end, tags={}))
            event['end'] = max(event['end'], end)
            for label, score in row['tags']:
                label = raw_label(label)
                if label in policies[source] or score < threshold * policy.scene_support_ratio:
                    continue
                own = event['tags'].setdefault(label, {}).setdefault(source,
                    dict(max_score=score, high=False, signature=signature, aggregates=[]))
                own['max_score'] = max(own['max_score'], score)
                own['high'] |= score >= threshold
                own['aggregates'].extend(aggregates[label])
    return [timeline[position] for position in sorted(timeline)]


def _shared_statistics(events, existing):
    stats = {}
    for position, event in enumerate(events):
        for tag, sources in event['tags'].items():
            if tag in existing:
                continue
            item = stats.setdefault(tag, dict(hits=0, high_hits=0, run=0, longest=0, last=-2,
                end=event['start'], intervals=[], sources={}, high_sources=set()))
            item['run'] = item['run'] + 1 if item['last'] == position-1 and event['start'] <= item['end'] + .001 else 1
            item['last'], item['end'] = position, event['end']
            item['longest'] = max(item['longest'], item['run'])
            item['hits'] += 1
            item['high_hits'] += int(any(own['high'] for own in sources.values()))
            append_interval(item['intervals'], event['start'], event['end'])
            for source, values in sources.items():
                own = item['sources'].setdefault(source, dict(hits=0, high_hits=0, max_score=values['max_score'],
                    score=None, signature=values['signature']))
                own['hits'] += 1
                own['high_hits'] += int(values['high'])
                own['max_score'] = max(own['max_score'], values['max_score'])
                if values['aggregates']:
                    own['score'] = max([own['score']] + values['aggregates']) if own['score'] is not None else max(values['aggregates'])
                if values['high']:
                    item['high_sources'].add(source)
    return stats


def _shared_candidates(group_key, members, scene_context, stats, events,
                       policy, options, incomplete):
    candidates = []
    for tag, item in stats.items():
        if item['high_hits'] >= policy.scene_min_hits and len(item['high_sources']) >= policy.consensus_min_models:
            reason, sources = 'shared_repeated_high', item['high_sources']
        elif (item['hits'] >= policy.scene_repeat_hits and item['longest'] >= policy.scene_repeat_run and
              item['hits'] / len(events) >= policy.scene_repeat_fraction and
              len(item['sources']) >= policy.consensus_min_models):
            reason, sources = 'shared_persistent_support', item['sources']
        else:
            continue
        sources = [source for source in SOURCES if source in sources]
        anchor = sources[0]
        evidence = {source: dict(item['sources'][source])
                    for source in sources}
        meta = next(segment['meta'] for source, segment, signature, *_ in members
                    if source == anchor and signature == evidence[anchor]['signature'])
        candidates.append(dict(tag=tag, source=anchor, score=evidence[anchor]['score'], meta=meta,
            qualification=reason, derived='shared_scene_frames', source_evidence=evidence,
            support_sources=sources, frame_group=group_key,
            scene_context=scene_context,
            build_options=options,
            joint_statistics=dict(hits=item['hits'], high_hits=item['high_hits'], longest_hits=item['longest'],
                sampled_frames=len(events), sample_fraction=item['hits']/len(events), cache_partial=incomplete)))
    return candidates


def _shared_scene_contributions(app, index, scene_records, policies, existing, metrics):
    """Never average model scores or interpret unretained detections as zero.

    Sparse positive evidence can prove a minimum count/run even when weaker
    observations were not retained. An incomplete cache cannot prove absence.
    Frame observations come from the same scene records as aggregate scores.
    """
    from stash_ai.evidence.identity import marker_segment_signature
    from stash_ai.evidence.cache_views import scene_threshold
    policy = app.TEMPORAL_POLICY
    if not policy.shared_scene_tags:
        return []

    existing = {entry['tag'] for entry in existing}
    options = dict(policy=asdict(policy),
                   scene_thresholds={marker_segment_signature(record): scene_threshold(record) for record in scene_records.values()},
                   scene_exclusions={source: sorted(values) for source, values in policies.items()})
    candidates, usable, partial = [], False, False

    for group_key, group in index.groups.items():
        members, scene_context = _shared_members(group, scene_records)
        if len(scene_context) < policy.consensus_min_models:
            continue
        usable = True
        incomplete = any(floor > threshold * policy.scene_support_ratio + 1e-12
                         for _, _, _, threshold, floor, _ in members)
        partial |= incomplete
        events = _shared_timeline(members, policies, policy)
        stats = _shared_statistics(events, existing)
        candidates.extend(_shared_candidates(
            group_key, members, scene_context, stats, events,
            policy, options, incomplete))

    candidates.sort(key=lambda item: (-item['joint_statistics']['high_hits'], -item['joint_statistics']['hits'],
                                     item['tag'], item['frame_group']))
    if metrics is not None:
        metrics['shared_scene_tags'] += len({item['tag'] for item in candidates})
        metrics['scene_cache_partial'] += int(partial)
        metrics['scene_cache_missing'] += int(not usable and len(scene_records) >= policy.consensus_min_models)
    return candidates

def cleanup_scene_support(app, store, scene_id, hits, selected, policies):
    """Revoke only published sources, under their frozen rules and input records."""
    from stash_ai.signatures import digest
    from stash_ai.evidence.frozen_replay import PublishedRecords, snapshot_context, frozen_policies, frozen_records
    from stash_ai.storage.snapshots import require_rebased
    require_rebased(hits)
    cleaned, rebuilt = [], {}
    published = PublishedRecords(store, scene_id)
    for hit in hits:
        if hit.get('derived') != 'shared_scene_frames':
            if hit['source'] not in selected or hit['tag'] not in policies[hit['source']]:
                cleaned.append(hit)
            continue
        if not any(source in selected and hit['tag'] in policies[source] for source in hit['support_sources']):
            cleaned.append(hit)
            continue
        key = digest([hit['build_options'], hit['scene_context']])
        if key not in rebuilt:
            matching = published.matching('scene', hit['scene_context'], error=
                'Shared scene sources changed since publication; run --merge-only before cleanup.')
            options = hit['build_options']
            context = snapshot_context(app, options, 'scene')
            from stash_ai.evidence.cache_views import views
            context, matching = views(context, frozen_records(options, 'scene', matching))
            frozen = frozen_policies(options, 'scene_exclusions', selected, policies)
            rebuilt[key] = shared_scene_contributions(context, matching, frozen, cache=store.cache)
        for candidate in rebuilt[key]:
            if candidate['tag'] == hit['tag'] and candidate not in cleaned:
                cleaned.append(candidate)
    return cleaned


def shared_scene_contributions(app, scene_records, policies, existing=(), metrics=None, *, cache=None):
    from stash_ai.evidence.cache_views import frame_records
    from stash_ai.evidence.frames import FrameIndex
    with FrameIndex(app, frame_records(scene_records), {s: frozenset() for s in SOURCES}, cache=cache, tag_policies=policies) as index:
        return _shared_scene_contributions(app, index, scene_records, policies, existing, metrics)
