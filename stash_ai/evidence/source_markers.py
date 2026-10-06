"""Build validated per-source markers before cross-model enrichment."""
from .contracts import RecordsBySource, MarkerContribution
from stash_ai.domain import Cue, FramePrediction
from stash_ai.runtime.timing import stage
from .labels import raw_label


def marker_contributions(app, records: RecordsBySource, core_policies, *, index=None, metrics=None, cache=None,
                         tag_policies) -> list[MarkerContribution]:
    from stash_ai.evidence.frames import FrameIndex
    if index is not None:
        if index.tag_policies != tag_policies or index.policies != core_policies:
            raise ValueError("Marker evidence and frame index require the same exclusion policies.")
        return _marker_contributions(app, records, core_policies, index, metrics)
    with FrameIndex(app, records, core_policies, cache=cache, tag_policies=tag_policies) as index:
        return _marker_contributions(app, records, core_policies, index, metrics)


def _marker_contributions(app, records, core_policies, index, metrics):
    from stash_ai.evidence.temporal.markers import consensus_seeds, build_markers as temporal_markers
    from stash_ai.evidence.temporal.policy import build_options
    from stash_ai.exclusions import core_policies_with_global
    from stash_ai.evidence.thresholds import validate_marker_cache
    validate_marker_cache(app, records)
    tag_policies = index.tag_policies
    effective_cores = core_policies_with_global(core_policies, tag_policies)
    report = index.report
    with stage(report, 'marker_seeds'):
        seeds = consensus_seeds(records, effective_cores, app.TEMPORAL_POLICY,
                                app.MARKER_MAX_TAGS_PER_FRAME)
    signatures = index.signatures
    spans = []
    with stage(report, 'marker_build'):
        for model, segment in records.items():
            if segment["mode"] == "frames":
                frames = [FramePrediction(Cue(frame["start"], frame["end"], "", (0, 0, 1, 1)),
                           tuple((raw_label(tag), score) for tag, score in frame["tags"]
                                 if tag not in tag_policies[model]))
                          for frame in segment["items"]]
                from stash_ai.evidence.thresholds import marker_threshold
                kwargs = dict(threshold=marker_threshold(segment), seeds=seeds,
                              identity=segment["meta"]["frame_input_signature"])
                defer_extras = app.TEMPORAL_POLICY.shared_marker_extras
                markers, _ = temporal_markers(app, frames, effective_cores[model], **kwargs,
                              collect_extras=not defer_extras,
                              extra_group=index.by_signature[signatures[id(segment)]],
                              extra_cache=index.cache)
                options = {**build_options(app, records), "core_exclusions": {k: sorted(v) for k, v in core_policies.items()},
                           "tag_exclusions": {k: sorted(v) for k, v in tag_policies.items()}}
                values = [dict(tag=marker.tag, start=marker.start, end=marker.end, score=marker.score,
                               hits=marker.hits, extra_tags=list(marker.extra_tags),
                               extra_evidence=list(marker.extra_evidence), strong_hits=marker.strong_hits,
                               support_sources=list(marker.support_sources), extra_support=marker.extra_support,
                               build_options=options, evidence_signature=signatures[id(segment)],
                               support_context={source: signatures[id(records[source])]
                                   for source in marker.support_sources}) for marker in markers]
            else:
                raise ValueError("Unsupported stored marker evidence.")
            for item in values:
                if item["tag"] not in effective_cores[model]:
                    from stash_ai.evidence.shared_markers import prune_blocked_extras
                    spans.append(prune_blocked_extras({**item, "source": model, "meta": segment["meta"]},
                                                     tag_policies[model]))
    from stash_ai.evidence.shared_markers import enrich
    with stage(report, 'marker_fusion'):
        return enrich(app, core_policies, spans, index, metrics)
