"""Capture raw predictions and check completion without publication side effects."""
from datetime import datetime, timezone
from stash_ai.config import vocabulary_digest
import hashlib
import json
from .input_identity import cue_manifest
from stash_ai.evidence.contracts import InferenceRecord
from stash_ai.evidence.formats import ANALYSIS_FORMAT


def now():
    return datetime.now(timezone.utc).isoformat()


def capture_outputs(app, config, vocabulary, cues, predictions, frames, analysis, *,
                    inference_signature, frame_input_signature) -> dict[str, InferenceRecord]:
    from stash_ai.evidence.retention import marker_cache_floor
    cue_data = cue_manifest(cues)
    meta = {"origin": "inference", "captured_at": now(),
            "inference_signature": inference_signature, "frame_input_signature": frame_input_signature, "repo": config.source.repo,
            "revision": config.source.revision, "vocabulary_sha256": vocabulary_digest(vocabulary.labels),
            "input_cues_sha256": hashlib.sha256(json.dumps(cue_data, ensure_ascii=False).encode()).hexdigest(),
            "scene_threshold": config.scene_threshold, "marker_threshold": config.marker_threshold,
            "aggregation": app.AGGREGATION, "top_k_frames": app.TOP_K_FRAMES}
    meta.update(analysis_format=ANALYSIS_FORMAT,
                scene_support_floor=config.scene_threshold * app.TEMPORAL_POLICY.scene_support_ratio)
    meta['marker_score_floor'] = marker_cache_floor(config, app.TEMPORAL_POLICY)
    outputs = {}
    if app.APPLY_SCENE_TAGS:
        outputs["scene"] = {"mode": "scene", "meta": {
                                **meta, 'scene_frames_format': 1,
                                'scene_score_floor': analysis['scene_score_floor']},
                            "frames": analysis['scene_frames'],
                            "items": [dict(tag=tag, score=score,
                                           statistics=analysis["scene_statistics"].get(tag)) for tag, score in predictions]}
    if app.CREATE_MARKERS:
        outputs["markers"] = {"mode": "frames", "meta": dict(meta), "items": [
            dict(start=frame.cue.start, end=frame.cue.end, tags=list(frame.tags)) for frame in frames]}
    from stash_ai.evidence.formats import validate_current
    for segment in outputs.values():
        validate_current(segment)
    return outputs
