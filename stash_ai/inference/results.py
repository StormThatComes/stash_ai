"""Capture raw predictions and check completion without publication side effects."""
from datetime import datetime, timezone
from stash_ai.config import vocabulary_digest
import hashlib
import json
from .input_identity import cue_manifest
from stash_ai.evidence.contracts import RetainedRecord


def now():
    return datetime.now(timezone.utc).isoformat()


def capture_outputs(app, config, vocabulary, cues, predictions, frames, analysis) -> dict[str, RetainedRecord]:
    from stash_ai.evidence.retention import marker_cache_floor
    cue_data = cue_manifest(cues)
    meta = {"origin": "inference", "captured_at": now(), "repo": config.source.repo,
            "revision": config.source.revision, "vocabulary_sha256": vocabulary_digest(vocabulary.labels),
            "input_cues_sha256": hashlib.sha256(json.dumps(cue_data, ensure_ascii=False).encode()).hexdigest(),
            "scene_threshold": config.scene_threshold, "marker_threshold": config.marker_threshold,
            "aggregation": app.AGGREGATION, "top_k_frames": app.TOP_K_FRAMES}
    meta.update(analysis_format=3,
                scene_support_floor=config.scene_threshold * app.TEMPORAL_POLICY.scene_support_ratio)
    meta['marker_score_floor'] = marker_cache_floor(config, app.TEMPORAL_POLICY)
    outputs = {}
    if app.APPLY_SCENE_TAGS:
        outputs["scene"] = {"mode": "scene", "meta": {
                                **meta, 'scene_frames_format': 1,
                                'scene_score_floor': analysis['scene_score_floor']},
                            "frames": analysis['scene_frames'],
                            "items": [dict(tag=tag, score=score,
                                           **({"statistics": analysis["scene_statistics"][tag]} if
                                              tag in analysis['scene_statistics'] else {})) for tag, score in predictions]}
    if app.CREATE_MARKERS:
        # An empty list proves no labels were discarded before caching.
        outputs["markers"] = {"mode": "frames", "meta": {
            **meta, "marker_tag_blocklist": []}, "items": [
            dict(start=frame.cue.start, end=frame.cue.end, tags=list(frame.tags)) for frame in frames]}
    return outputs
