"""Read-only tag explanations and effective-policy diagnostics."""
from collections import Counter
import json

from stash_ai.evidence import pipeline as merge
import stash_ai.evidence.intervals as merge_intervals
import stash_ai.evidence.labels as merge_labels
from stash_ai.storage.store import StateStore, scope_for
from stash_ai.storage.results import KINDS


def _effective_scene_thresholds(app, threshold):
    """Explain the actual scene cutoffs used by temporal qualification."""
    result = {"effective_scene_threshold": threshold}
    if type(threshold) not in (int, float):
        return result
    policy = app.TEMPORAL_POLICY
    ratio = policy.scene_support_ratio if policy.enabled else 1.0
    result.update(
        temporal_enabled=policy.enabled,
        scene_support_ratio=ratio,
        effective_support_floor=threshold * ratio,
    )
    if policy.enabled:
        result["strong_peak_threshold"] = (
            threshold + (1.0 - threshold) * policy.scene_strong_margin
        )
    return result


def _record_thresholds(app, kind, segment):
    """Expose stored policy separately from the effective replay policy."""
    meta = segment.get("meta", {})
    if kind == "scene":
        from stash_ai.evidence.identity import SCENE_OVERRIDE
        from stash_ai.evidence.cache_views import scene_threshold
        result = {
            "stored_scene_threshold": meta.get("scene_threshold"),
            "scene_threshold_override": segment.get(SCENE_OVERRIDE),
            **_effective_scene_thresholds(app, scene_threshold(segment)),
        }
        return result
    from stash_ai.evidence.temporal.policy import marker_floor
    from stash_ai.evidence.identity import OVERRIDE
    from stash_ai.evidence.thresholds import marker_threshold
    effective = marker_threshold(segment)
    result = {
        "stored_marker_threshold": meta.get("marker_threshold"),
        "marker_threshold_override": segment.get(OVERRIDE),
        "effective_marker_threshold": effective,
    }
    if type(effective) in (int, float):
        policy = app.TEMPORAL_POLICY
        result.update(
            temporal_enabled=policy.enabled,
            marker_continue_ratio=policy.marker_continue_ratio if policy.enabled else 1.0,
            effective_support_floor=marker_floor(policy, effective),
        )
    return result


def _explain_scene_contributor(app, hit, recorded_scene_sources):
    value = dict(hit)
    thresholds = _effective_scene_thresholds(
        app, hit.get("meta", {}).get("scene_threshold"))
    source_records = recorded_scene_sources.get(hit.get("source"), {}).get("records", ())
    signature = hit.get("meta", {}).get("inference_signature")
    matching = [
        record["thresholds"] for record in source_records
        if record.get("meta", {}).get("inference_signature") == signature
    ]
    if len(matching) == 1:
        thresholds["stored_scene_threshold"] = matching[0].get("stored_scene_threshold")
        thresholds["scene_threshold_override"] = matching[0].get("scene_threshold_override")
    value["thresholds"] = thresholds
    return value


def explain(app, args, core_policies, scene_policies):
    label = merge_labels.raw_label(args.explain_tag.removeprefix(merge_labels.AI.prefix))
    # Inspection always opens a read-only snapshot, even without --dry-run.
    with StateStore(args.state_db, scope_for(app.STASH_URL), True,
                    cache_bytes=int(args.ram_cache_gb * 2**30),
                    memory_budget_bytes=int(args.ram_budget_gb * 2**30)) as store:
        reports = []
        for scene_id in dict.fromkeys(args.scene_id):
            diagnostics = Counter()
            hits, spans = merge.evidence(app, store, scene_id, core_policies, scene_policies, diagnostics)
            matching = [span for span in spans if span["tag"] == label]
            recorded_sources = {}
            for kind in KINDS:
                sources = {}
                for source, segments in store.records(scene_id, kind).items():
                    matches = []
                    for segment in segments:
                        items = ([dict(start=frame["start"], end=frame["end"], score=score)
                                  for frame in segment["items"] for tag, score in frame["tags"] if tag == label]
                                 if segment["mode"] == "frames" else [entry for entry in segment["items"] if entry["tag"] == label])
                        if items:
                            matches.append({
                                "meta": segment["meta"],
                                "thresholds": _record_thresholds(app, kind, segment),
                                "detections": items,
                            })
                    if matches:
                        sources[source] = {"excluded": label in scene_policies[source] or (kind == 'markers' and label in core_policies[source]),
                                           "records": matches}
                recorded_sources[kind] = sources
            reports.append({"scene_id": str(scene_id), "tag": merge_labels.AI.prefix + label, 'diagnostics': dict(diagnostics),
                            "scene_contributors": [
                                _explain_scene_contributor(app, hit, recorded_sources["scene"])
                                for hit in hits if hit["tag"] == label
                            ],
                            "markers": merge_intervals.union_intervals(matching), "recorded_sources": recorded_sources,
                            "as_marker_extra": [dict(tag=merge_labels.AI.prefix + marker["tag"], start=marker["start"], end=marker["end"],
                                                     evidence=marker["extra_evidence"][label])
                                                for marker in merge_intervals.union_intervals(spans) if label in marker.get("extra_evidence", {})],
                            "pending_sync": any(store.snapshot(scene_id, kind, borrow=True)[1] is not None for kind in KINDS)})
    print(json.dumps(reports, ensure_ascii=False, indent=2, allow_nan=False))
    return 0
