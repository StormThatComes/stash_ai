"""Replay and clip published evidence when cleanup revokes source votes."""
from __future__ import annotations

import json
from dataclasses import replace

import stash_ai.evidence.identity as merge_identity
import stash_ai.evidence.source_markers as merge_source_markers
from stash_ai.models import SOURCES


class EvidenceReplay:
    """Lazily index one scene's proofs and cache their frozen-policy replay."""

    def __init__(self, app, store, scene_id, selected, core_policies, tag_policies):
        self.app, self.store, self.scene_id = app, store, scene_id
        self.selected, self.core_policies, self.tag_policies = selected, core_policies, tag_policies
        self.records = None
        self.available_by_source, self.rebuilt = {}, {}

    def _matching_records(self, required):
        if self.records is None:
            self.records = self.store.records(self.scene_id, 'markers', borrow=True)
        matching = {}
        for source, signatures in required.items():
            if source not in self.available_by_source:
                self.available_by_source[source] = {
                    merge_identity.marker_segment_signature(segment): segment
                    for segment in self.records.get(source, [])}
            available = self.available_by_source[source]
            if not set(signatures) <= set(available):
                raise RuntimeError('Consensus sources changed since publication; run --merge-only before cleanup.')
            matching[source] = [available[signature] for signature in signatures]
        return matching

    def _rebuild(self, options, matching):
        from stash_ai.evidence.temporal.policy import TemporalPolicy

        context = replace(self.app)
        context.TEMPORAL_POLICY = TemporalPolicy.from_snapshot(options['policy'])
        context.MARKER_THRESHOLDS = options.get('marker_thresholds', {})
        for name, value in options['settings'].items():
            setattr(context, name, value)
        # Preserve the retired option only as part of this published proof.
        legacy_blocklist = frozenset(options.get('blocklist', []))
        policies = {source: frozenset(options['core_exclusions'][source]) |
                    (self.core_policies[source] if source in self.selected else frozenset())
                    for source in SOURCES}
        frozen_tags = {source: frozenset(options.get('tag_exclusions', {}).get(source, [])) | legacy_blocklist |
                       (self.tag_policies[source] if source in self.selected else frozenset())
                       for source in SOURCES}
        # Replay old proofs with frozen exclusions; clipping cannot admit new votes.
        candidates = merge_source_markers.marker_contributions(
            context, matching, policies, tag_policies=frozen_tags,
            allow_unknown_blocklist=True, allow_unknown_retention=True,
            legacy_blocklist=legacy_blocklist)
        return candidates, context, policies, matching, frozen_tags

    def candidates(self, span):
        required = span.get('frame_context') or {
            **span.get('support_context', {}), span['source']: [span['evidence_signature']]}
        matching = self._matching_records(required)
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


def _clip_candidate(span, item, context, matching, policies, frozen_tags):
    from stash_ai.evidence.shared_markers import refresh_clipped_evidence

    if item['tag'] != span['tag']:
        return None
    if span.get('derived') != 'shared_frames' and (item['source'] != span['source'] or item.get('derived')):
        return None
    start, end = max(span['start'], item['start']), min(span['end'], item['end'])
    if end <= start:
        return None
    clipped = {**item, 'start': start, 'end': end}
    if span['build_options'].get('blocklist'):
        # Later cleanup needs these frozen bans to validate the same legacy frames.
        clipped['build_options'] = {**item['build_options'], 'blocklist': span['build_options']['blocklist']}
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
        if not refresh_clipped_evidence(context, clipped, matching, policies, frozen_tags):
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
        candidates, context, policies, matching, frozen_tags = replay.candidates(span)
        for item in candidates:
            clipped = _clip_candidate(span, item, context, matching, policies, frozen_tags)
            if clipped is None:
                continue
            # Cleanup only revokes extras; normal merge may admit additional labels.
            new_extras = (set(clipped.get('extra_tags', [])) | set(clipped.get('extra_support', {}))) - extra_labels
            clipped = prune_blocked_extras(clipped, new_extras)
            if clipped not in cleaned:
                cleaned.append(clipped)
    return cleaned
