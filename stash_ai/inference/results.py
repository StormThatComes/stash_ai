"""Capture raw predictions and check completion without publication side effects."""
from datetime import datetime, timezone
from stash_ai.config import vocabulary_digest
from stash_ai.models import ModelSource
from stash_ai.publication import markers as marker_ops
import hashlib
import json


def now():
    return datetime.now(timezone.utc).isoformat()


def capture_outputs(app, config, vocabulary, cues, predictions, frames, analysis=None):
    from stash_ai.evidence.retention import marker_cache_floor
    cue_data = [[cue.start, cue.end, cue.image, list(cue.rect)] for cue in cues]
    meta = {"origin": "inference", "captured_at": now(), "repo": config.source.repo,
            "revision": config.source.revision, "vocabulary_sha256": vocabulary_digest(vocabulary.labels),
            "input_cues_sha256": hashlib.sha256(json.dumps(cue_data, ensure_ascii=False).encode()).hexdigest(),
            "scene_threshold": config.scene_threshold, "marker_threshold": config.marker_threshold,
            "aggregation": app.AGGREGATION, "top_k_frames": app.TOP_K_FRAMES}
    if app.TEMPORAL_POLICY.enabled:
        meta.update(analysis_format=3,
                    scene_support_floor=config.scene_threshold * app.TEMPORAL_POLICY.scene_support_ratio)
    if app.TEMPORAL_POLICY.enabled or config.marker_cache_floor is not None:
        meta['marker_score_floor'] = marker_cache_floor(config, app.TEMPORAL_POLICY)
    outputs = {}
    if app.APPLY_SCENE_TAGS:
        outputs["scene"] = {"mode": "scene", "meta": meta,
                            "items": [dict(tag=tag, score=score,
                                           **({"statistics": analysis["scene_statistics"][tag]} if analysis and
                                              tag in analysis.get("scene_statistics", {}) else {})) for tag, score in predictions]}
        if analysis is not None and 'scene_frames' in analysis:
            outputs['scene']['frames'] = analysis['scene_frames']
            outputs['scene']['meta'] = {**meta, 'scene_frames_format': 1,
                'scene_score_floor': analysis['scene_score_floor']}
    if app.CREATE_MARKERS:
        # Empty legacy metadata proves no labels were discarded before caching.
        outputs["markers"] = {"mode": "frames", "meta": {
            **meta, "marker_tag_blocklist": []}, "items": [
            dict(start=frame.cue.start, end=frame.cue.end, tags=list(frame.tags)) for frame in frames]}
    return outputs


def has_outputs(app, store, scene_id, model):
    from stash_ai.storage.progress import completed, output_kinds
    return completed(store, scene_id, model, output_kinds(app))



def outputs_already_present(app, scene: dict, source: ModelSource) -> bool:
    tagged = any(tag["name"].startswith(source.prefix) for tag in scene.get("tags") or [])
    marked = any(marker_ops.owned_marker(marker, source) for marker in scene.get("scene_markers") or [])
    return (not app.APPLY_SCENE_TAGS or tagged) and (not app.CREATE_MARKERS or marked)
