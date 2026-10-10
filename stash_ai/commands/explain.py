"""Read-only tag explanations and effective-policy diagnostics."""
from collections import Counter
import json

from stash_ai.evidence import pipeline as merge
import stash_ai.evidence.intervals as merge_intervals
import stash_ai.evidence.labels as merge_labels
from stash_ai.storage.store import StateStore, scope_for
from stash_ai.storage.results import KINDS
from stash_ai.db.read_source import consistent_source

REPORT_FORMAT = 2


def _effective_scene_thresholds(app, threshold):
    """Explain the actual scene cutoffs used by temporal qualification."""
    result = {"effective_scene_threshold": threshold}
    policy = app.TEMPORAL_POLICY
    ratio = policy.scene_support_ratio
    result.update(
        temporal_enabled=True,
        scene_support_ratio=ratio,
        effective_support_floor=threshold * ratio,
    )
    result["strong_peak_threshold"] = (
        threshold + (1.0 - threshold) * policy.scene_strong_margin
    )
    return result


def _record_thresholds(app, kind, segment):
    """Expose stored policy separately from the effective replay policy."""
    meta = segment["meta"]
    if kind == "scene":
        from stash_ai.storage.thresholds import PUBLICATION_THRESHOLD
        from stash_ai.evidence.cache_views import scene_threshold
        result = {
            "stored_scene_threshold": meta["scene_threshold"],
            "publication_threshold": segment.get(PUBLICATION_THRESHOLD),
            **_effective_scene_thresholds(app, scene_threshold(segment)),
        }
        return result
    from stash_ai.evidence.temporal.policy import marker_floor
    from stash_ai.storage.thresholds import PUBLICATION_THRESHOLD
    from stash_ai.evidence.thresholds import marker_threshold
    effective = marker_threshold(segment)
    result = {
        "stored_marker_threshold": meta["marker_threshold"],
        "publication_threshold": segment.get(PUBLICATION_THRESHOLD),
        "effective_marker_threshold": effective,
    }
    policy = app.TEMPORAL_POLICY
    result.update(
        temporal_enabled=True,
        marker_continue_ratio=policy.marker_continue_ratio,
        effective_support_floor=marker_floor(policy, effective),
    )
    return result


def _explain_scene_contributor(app, hit, recorded_scene_sources):
    value = dict(hit)
    thresholds = _effective_scene_thresholds(
        app, hit["meta"]["scene_threshold"])
    record = recorded_scene_sources.get(hit.get("source"), {}).get("record")
    signature = hit["meta"]["inference_signature"]
    if record and record["meta"]["inference_signature"] == signature:
        thresholds["stored_scene_threshold"] = record["thresholds"].get("stored_scene_threshold")
        thresholds["publication_threshold"] = record["thresholds"].get("publication_threshold")
    value["thresholds"] = thresholds
    return value


def explain(app, args, core_policies, scene_policies):
    label = merge_labels.raw_label(args.explain_tag.removeprefix(merge_labels.AI.prefix))
    # Inspection always opens a read-only snapshot, even without --dry-run.
    with consistent_source(args.state_db), StateStore(args.state_db, scope_for(app.STASH_URL), True,
                    cache_bytes=int(args.ram_cache_gb * 2**30),
                    memory_budget_bytes=int(args.ram_budget_gb * 2**30)) as store:
        reports = []
        for scene_id in dict.fromkeys(args.scene_id):
            diagnostics = Counter()
            hits, spans = merge.evidence(app, store, scene_id, core_policies, scene_policies, diagnostics)
            markers = merge_intervals.union_intervals(spans)
            recorded_sources = {}
            for kind in KINDS:
                sources = {}
                for source, segment in store.records(scene_id, kind, borrow=True).items():
                    items = ([dict(start=frame["start"], end=frame["end"], score=score)
                              for frame in segment["items"] for tag, score in frame["tags"] if tag == label]
                             if segment["mode"] == "frames" else [entry for entry in segment["items"] if entry["tag"] == label])
                    if items:
                        sources[source] = {
                            "excluded": label in scene_policies[source] or (kind == 'markers' and label in core_policies[source]),
                            "record": {
                                "meta": segment["meta"],
                                "thresholds": _record_thresholds(app, kind, segment),
                                "detections": items,
                            },
                        }
                recorded_sources[kind] = sources
            reports.append({"format": REPORT_FORMAT, "scene_id": str(scene_id), "tag": merge_labels.AI.prefix + label, 'diagnostics': dict(diagnostics),
                            "scene_contributors": [
                                _explain_scene_contributor(app, hit, recorded_sources["scene"])
                                for hit in hits if hit["tag"] == label
                            ],
                            "markers": [marker for marker in markers if marker["tag"] == label], "recorded_sources": recorded_sources,
                            "as_marker_extra": [dict(tag=merge_labels.AI.prefix + marker["tag"], start=marker["start"], end=marker["end"],
                                                     evidence=marker["extra_evidence"][label])
                                                for marker in markers if label in marker.get("extra_evidence", {})],
                            "pending_sync": any(store.snapshot(scene_id, kind, borrow=True)[1] is not None for kind in KINDS)})
        from stash_ai.storage.preview import check_source
        check_source(store)
    print(json.dumps(reports, ensure_ascii=False, indent=2, allow_nan=False))
    return 0
