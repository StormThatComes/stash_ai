"""Replay and clip published evidence when cleanup revokes source votes."""
from __future__ import annotations

import json
from dataclasses import dataclass

import stash_ai.evidence.source_markers as merge_source_markers
from stash_ai.evidence.frozen_replay import PublishedRecords, snapshot_context, frozen_policies


@dataclass
class MarkerReplay:
    candidates: list
    context: object
    core_policies: dict
    records: dict
    tag_policies: dict


class EvidenceReplay:
    """Lazily index one scene's proofs and cache their frozen-policy replay."""

    def __init__(self, app, store, scene_id, selected, core_policies, tag_policies):
        self.app, self.store, self.scene_id = app, store, scene_id
        self.selected, self.core_policies, self.tag_policies = selected, core_policies, tag_policies
        self.published = PublishedRecords(store, scene_id)
        self.rebuilt = {}

    def _rebuild(self, options, matching):
        context = snapshot_context(self.app, options, 'markers')
        policies = frozen_policies(options, 'core_exclusions', self.selected, self.core_policies)
        frozen_tags = frozen_policies(options, 'tag_exclusions', self.selected, self.tag_policies)
        # Replay old proofs with frozen exclusions; clipping cannot admit new votes.
        candidates = merge_source_markers.marker_contributions(
            context, matching, policies, tag_policies=frozen_tags,
            allow_unknown_blocklist=True, allow_unknown_retention=True,
            frozen_exclusions=frozen_tags)
        return MarkerReplay(candidates, context, policies, matching, frozen_tags)

    def candidates(self, span):
        required = span.get('frame_context') or {
            **span.get('support_context', {}), span['source']: [span['evidence_signature']]}
        matching = self.published.matching('markers', required, error=
            'Consensus sources changed since publication; run --merge-only before cleanup.')
        key = json.dumps([span['build_options'], required], sort_keys=True)
        if key not in self.rebuilt:
            self.rebuilt[key] = self._rebuild(span['build_options'], matching)
        return self.rebuilt[key]


def _needs_replay(span, selected, core_policies, tag_policies, extra_labels):
    core_affected = any(source in selected and (span['tag'] in core_policies[source] or
                        span['tag'] in tag_policies[source])
                        for source in span.get('support_sources', []))
    extra_sources = set(span.get('frame_context', {})) | {span['source']}
    extras_affected = any(extra_labels & tag_policies[source] for source in extra_sources & selected)
    return core_affected or extras_affected


def _clip_candidate(span, item, replay: MarkerReplay):
    from stash_ai.evidence.shared_markers import refresh_clipped_evidence

    if item['tag'] != span['tag']:
        return None
    if span.get('derived') != 'shared_frames' and (item['source'] != span['source'] or item.get('derived')):
        return None
    start, end = max(span['start'], item['start']), min(span['end'], item['end'])
    if end <= start:
        return None
    clipped = {**item, 'start': start, 'end': end}
    if (start, end) != (item['start'], item['end']):
        clipped.update(hits=None, strong_hits=None, extra_evidence=[], cleanup_clipped=True)
        if clipped.get('derived') == 'shared_frames':
            clipped.update(shared_hits=None, shared_strong_hits=None)
        # Keep extras bounded by the published range; rebuild exact frame proofs.
        clipped['extra_tags'] = sorted(set(item.get('extra_tags', [])) & set(span.get('extra_tags', [])))
        clipped['extra_support'] = {label: dict(
            intervals=[[max(a, start), min(b, end)] for a, b in evidence['intervals'] if min(b, end) > max(a, start)],
            sample_times=[t for t in evidence['sample_times'] if start <= t < end])
            for label, evidence in item.get('extra_support', {}).items()}
        if not refresh_clipped_evidence(replay.context, clipped, replay.records, replay.core_policies, replay.tag_policies):
            return None
    return clipped


def cleanup_support(app, store, scene_id, spans, selected, core_policies, tag_policies=None):
    """Re-evaluate revoked votes, bounded by the already-published source ranges."""
    from stash_ai.evidence.shared_markers import prune_blocked_extras
    from stash_ai.exclusions import policies_or_empty

    tag_policies = policies_or_empty(tag_policies)
    replay = EvidenceReplay(app, store, scene_id, selected, core_policies, tag_policies)
    cleaned = []
    for span in spans:
        if span.get('derived') == 'occurrence':
            from stash_ai.evidence.occurrences import cleanup_occurrence
            point = cleanup_occurrence(app, store, scene_id, span, selected, core_policies, tag_policies)
            if point is not None:
                cleaned.append(point)
            continue
        own_blocked = tag_policies[span['source']] if span['source'] in selected else frozenset()
        if span.get('derived') != 'shared_frames' and span['source'] in selected and (
                span['tag'] in core_policies[span['source']] or span['tag'] in own_blocked):
            continue
        extra_labels = set(span.get('extra_tags', [])) | set(span.get('extra_support', {}))
        if not _needs_replay(span, selected, core_policies, tag_policies, extra_labels):
            cleaned.append(span)
            continue
        if not span.get('build_options') or not span.get('evidence_signature'):
            # Imported ranges have no frame proofs; remove only captured extras.
            cleaned.append(prune_blocked_extras(span, own_blocked))
            continue
        prepared = replay.candidates(span)
        for item in prepared.candidates:
            clipped = _clip_candidate(span, item, prepared)
            if clipped is None:
                continue
            # Cleanup only revokes extras; normal merge may admit additional labels.
            new_extras = (set(clipped.get('extra_tags', [])) | set(clipped.get('extra_support', {}))) - extra_labels
            clipped = prune_blocked_extras(clipped, new_extras)
            if clipped not in cleaned:
                cleaned.append(clipped)
    return cleaned
