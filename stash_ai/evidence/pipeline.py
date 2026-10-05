"""Coordinate scene, marker and occurrence qualification from retained results."""
from contextlib import nullcontext
from stash_ai.runtime.timing import stage
from .labels import raw_label
from .contracts import SceneContribution, MarkerContribution
from .source_markers import marker_contributions


def evidence(app, store, scene_id, core_policies, scene_policies, metrics=None, *, selected_sources=None) -> tuple[list[SceneContribution], list[MarkerContribution]]:
    """Retain source markers and enrich them with compatible shared observations.

    selected_sources is an analysis-only view used for counterfactual reports:
    omitted sources are removed before every fusion/consensus/occurrence stage.
    """
    from stash_ai.evidence.qualification import scene_qualification
    from stash_ai.evidence.frames import FrameIndex
    from stash_ai.evidence.scenes import shared_scene_contributions
    report = store.report
    scene_hits, spans = [], []
    need_scene = app.APPLY_SCENE_TAGS or (app.CREATE_MARKERS
                                        and app.TEMPORAL_POLICY.occurrence_markers)
    scene_records = store.records(scene_id, 'scene', borrow=True) if need_scene else {}
    if selected_sources is not None:
        selected_sources = frozenset(selected_sources)
        scene_records = {source: record for source, record in scene_records.items()
                         if source in selected_sources}
    from stash_ai.evidence.cache_views import views, frame_records
    app, scene_records = views(app, scene_records)
    for model, segment in scene_records.items():
        eligible = []
        for entry in segment["items"]:
            label = raw_label(entry["tag"])
            reason = scene_qualification(entry, segment["meta"],
                                  segment["meta"].get("scene_threshold", 0.0), app.TEMPORAL_POLICY)
            if label not in scene_policies[model] and reason:
                eligible.append({**entry, "source": model, "meta": segment["meta"],
                                 **({"qualification": reason} if entry.get("statistics") else {})})
        # Apply the scene cap once per current model record.
        eligible.sort(key=lambda item: (-(item["score"] if item["score"] is not None else -1), item["tag"]))
        scene_hits.extend(eligible[:app.MAX_TAGS] if app.MAX_TAGS else eligible)
    records = store.records(scene_id, 'markers', borrow=True)
    if selected_sources is not None:
        records = {source: record for source, record in records.items()
                   if source in selected_sources}
    from stash_ai.evidence.thresholds import marker_context
    app = marker_context(app, records)
    if report is not None:
        inputs = records.values()
        report.record_counts(input_models=len(records), input_segments=len(inputs),
            frame_rows=sum(len(s['items']) for s in inputs if s['mode'] == 'frames'),
            retained_detections=sum(len(row['tags']) for s in inputs if s['mode'] == 'frames' for row in s['items']))
    with stage(report, 'frame_index'):
        index = FrameIndex(app, records, core_policies, cache=store.cache, tag_policies=scene_policies, report=report)
    with index:
        if app.CREATE_MARKERS:
            spans = marker_contributions(app, records, core_policies, index=index, metrics=metrics, tag_policies=scene_policies)
        if need_scene:
            with stage(report, 'scene_fusion'):
                scene_frames = frame_records(scene_records)
                with FrameIndex(app, scene_frames, core_policies, cache=store.cache, tag_policies=scene_policies) as scene_index:
                    scene_hits.extend(shared_scene_contributions(app, scene_index, scene_records, scene_policies, scene_hits, metrics))
    if app.CREATE_MARKERS:
        from stash_ai.evidence.occurrences import occurrence_contributions
        report = store.report
        with report.measure('occurrence_compute') if report is not None else nullcontext():
            spans.extend(occurrence_contributions(app, scene_hits, frame_records(scene_records), spans, core_policies, scene_policies, metrics))
    return sorted(scene_hits, key=lambda item: (item["tag"], item["source"])), spans
