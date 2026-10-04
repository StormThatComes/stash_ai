"""Build validated per-source markers before cross-model enrichment."""
from stash_ai.domain import Cue, FramePrediction
from stash_ai.runtime import stage
from .labels import raw_label


def marker_contributions(app, records, core_policies, *, index=None, metrics=None, cache=None, tag_policies=None,
                         allow_unknown_blocklist=False, allow_unknown_retention=False,
                         legacy_blocklist=frozenset()):
    from stash_ai.evidence.frames import FrameIndex
    from stash_ai.evidence.thresholds import marker_context
    app = marker_context(app, records)
    if index is not None:
        return _marker_contributions(app, records, core_policies, index, metrics, tag_policies,
                                     allow_unknown_blocklist=allow_unknown_blocklist,
                                     allow_unknown_retention=allow_unknown_retention,
                                     legacy_blocklist=legacy_blocklist)
    with FrameIndex(app, records, core_policies, cache=cache, tag_policies=tag_policies) as index:
        return _marker_contributions(app, records, core_policies, index, metrics, tag_policies,
                                     allow_unknown_blocklist=allow_unknown_blocklist,
                                     allow_unknown_retention=allow_unknown_retention,
                                     legacy_blocklist=legacy_blocklist)


def _marker_contributions(app, records, core_policies, index, metrics, tag_policies, *,
                          allow_unknown_blocklist=False, allow_unknown_retention=False,
                          legacy_blocklist=frozenset()):
    from stash_ai.evidence.temporal.markers import consensus_seeds, build_markers as temporal_markers
    from stash_ai.evidence.temporal.policy import build_options
    from stash_ai.exclusions import policies_or_empty, core_policies_with_global
    from stash_ai.evidence.thresholds import validate_marker_cache
    validate_marker_cache(app, records, allow_unknown_blocklist=allow_unknown_blocklist,
                          allow_unknown_retention=allow_unknown_retention,
                          legacy_blocklist=legacy_blocklist)
    tag_policies = policies_or_empty(tag_policies if tag_policies is not None else getattr(index, 'tag_policies', None))
    effective_cores = core_policies_with_global(core_policies, tag_policies)
    report = getattr(index, 'report', None)
    with stage(report, 'marker_seeds'):
        seeds = consensus_seeds(records, effective_cores, app.TEMPORAL_POLICY,
                                thresholds=app.MARKER_THRESHOLDS)
    signatures = index.signatures
    spans = []
    with stage(report, 'marker_build'):
        for model, segments in records.items():
            for segment in segments:
                if segment["mode"] == "frames":
                    frames = [FramePrediction(Cue(frame["start"], frame["end"], "", (0, 0, 1, 1)),
                               tuple((raw_label(tag), score) for tag, score in frame["tags"]
                                     if tag not in tag_policies[model]))
                              for frame in segment["items"]]
                    from stash_ai.evidence.thresholds import marker_threshold
                    kwargs = dict(threshold=marker_threshold(segment, app.MARKER_THRESHOLDS), seeds=seeds,
                                  identity=segment["meta"].get("frame_input_signature"))
                    defer_extras = app.TEMPORAL_POLICY.shared_marker_extras and signatures[id(segment)] in index.by_signature
                    markers, _ = (temporal_markers(app, frames, effective_cores[model], **kwargs,
                                  collect_extras=not defer_extras,
                                  extra_group=index.by_signature.get(signatures[id(segment)], signatures[id(segment)]),
                                  extra_cache=index.cache) if app.TEMPORAL_POLICY.enabled else
                                  app.build_markers(frames, effective_cores[model], **kwargs))
                    options = {**build_options(app), "core_exclusions": {k: sorted(v) for k, v in core_policies.items()},
                               "tag_exclusions": {k: sorted(v) for k, v in tag_policies.items()},
                               "blocklist": []}
                    values = [dict(tag=marker.tag, start=marker.start, end=marker.end, score=marker.score,
                                   hits=marker.hits, extra_tags=list(marker.extra_tags),
                                   **(dict(extra_evidence=list(marker.extra_evidence), strong_hits=marker.strong_hits,
                                           support_sources=list(marker.support_sources), extra_support=marker.extra_support, build_options=options,
                                           evidence_signature=signatures[id(segment)],
                                           support_context={source: sorted({signatures[id(s)] for s in records[source]
                                               if s["meta"].get("frame_input_signature") == segment["meta"].get("frame_input_signature")})
                                               for source in marker.support_sources})
                                      if app.TEMPORAL_POLICY.enabled else {})) for marker in markers]
                elif segment["mode"] == "spans":
                    values = segment["items"]
                else:
                    raise ValueError("Unsupported stored marker evidence.")
                for item in values:
                    if item["tag"] not in effective_cores[model]:
                        from stash_ai.evidence.shared_markers import prune_blocked_extras
                        spans.append(prune_blocked_extras({**item, "source": model, "meta": segment["meta"]},
                                                         tag_policies[model]))
    from stash_ai.evidence.shared_markers import enrich
    with stage(report, 'marker_fusion'):
        return enrich(app, records, core_policies, spans, signatures, index, metrics)
