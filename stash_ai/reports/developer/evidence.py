"""Content-free result shape and compression statistics."""
from collections import Counter
import json
import zlib

from .distribution import Distribution
from .common import _safe_model, _safe_kind, _safe_counter


def _decode_result_blob(blob):
    raw = zlib.decompress(blob)
    return raw, json.loads(raw)


def _group_bucket(groups, model, kind):
    key = (_safe_model(model), _safe_kind(kind))
    if key not in groups:
        groups[key] = {
            "rows": 0,
            "decode_errors": 0,
            "compressed_bytes": Distribution(),
            "json_bytes": Distribution(),
            "compression_pct": Distribution(),
            "segments_per_row": Distribution(),
            "items_per_segment": Distribution(),
            "frame_detections": Distribution(),
            "scene_candidates": Distribution(),
            "scene_observation_frames": Distribution(),
            "scene_observations": Distribution(),
            "span_items": Distribution(),
            "segment_modes": Counter(),
            "origins": Counter(),
            "scene_frame_format": Counter(),
            "analysis_format": Counter(),
        }
    return groups[key]


def _result_evidence_report(db, tables):
    if "results" not in tables:
        return []

    groups = {}
    for model, kind, blob in db.execute("SELECT model,kind,data FROM results"):
        bucket = _group_bucket(groups, model, kind)
        bucket["rows"] += 1
        bucket["compressed_bytes"].add(len(blob))
        try:
            raw, segments = _decode_result_blob(blob)
        except (ValueError, TypeError, UnicodeDecodeError, json.JSONDecodeError, zlib.error):
            bucket["decode_errors"] += 1
            continue

        bucket["json_bytes"].add(len(raw))
        if raw:
            bucket["compression_pct"].add(round(len(blob) * 100 / len(raw)))
        if not isinstance(segments, list):
            bucket["decode_errors"] += 1
            continue

        bucket["segments_per_row"].add(len(segments))
        for segment in segments:
            if not isinstance(segment, dict):
                bucket["decode_errors"] += 1
                continue
            mode = segment.get("mode")
            safe_mode = mode if mode in ("scene", "frames", "spans") else "other"
            bucket["segment_modes"][safe_mode] += 1

            meta = segment.get("meta")
            if isinstance(meta, dict):
                origin = meta.get("origin")
                bucket["origins"][origin if origin in ("inference", "legacy") else "other"] += 1
                scene_format = meta.get("scene_frames_format")
                if isinstance(scene_format, int):
                    bucket["scene_frame_format"]["1" if scene_format == 1 else "other"] += 1
                analysis_format = meta.get("analysis_format")
                if isinstance(analysis_format, int):
                    bucket["analysis_format"][str(analysis_format) if analysis_format in (1, 2, 3, 4) else "other"] += 1

            items = segment.get("items")
            if not isinstance(items, list):
                bucket["decode_errors"] += 1
                continue
            bucket["items_per_segment"].add(len(items))

            if safe_mode == "frames":
                for frame in items:
                    tags = frame.get("tags") if isinstance(frame, dict) else None
                    bucket["frame_detections"].add(len(tags) if isinstance(tags, list) else 0)
            elif safe_mode == "scene":
                bucket["scene_candidates"].add(len(items))
                frames = segment.get("frames")
                if isinstance(frames, list):
                    bucket["scene_observation_frames"].add(len(frames))
                    for frame in frames:
                        tags = frame.get("tags") if isinstance(frame, dict) else None
                        bucket["scene_observations"].add(len(tags) if isinstance(tags, list) else 0)
            elif safe_mode == "spans":
                bucket["span_items"].add(len(items))

    result = []
    for (model, kind), bucket in sorted(groups.items()):
        result.append({
            "model": model,
            "kind": kind,
            "rows": bucket["rows"],
            "decode_errors": bucket["decode_errors"],
            "compressed_bytes": bucket["compressed_bytes"].summary(total=True),
            "json_bytes": bucket["json_bytes"].summary(total=True),
            "compressed_to_json_pct": bucket["compression_pct"].summary(),
            "segments_per_row": bucket["segments_per_row"].summary(),
            "items_per_segment": bucket["items_per_segment"].summary(),
            "frame_detections_per_frame": bucket["frame_detections"].summary(),
            "scene_candidates_per_segment": bucket["scene_candidates"].summary(),
            "scene_observation_frames_per_segment": bucket["scene_observation_frames"].summary(),
            "scene_observations_per_frame": bucket["scene_observations"].summary(),
            "span_items_per_segment": bucket["span_items"].summary(),
            "segment_modes": _safe_counter(bucket["segment_modes"], {"scene", "frames", "spans", "other"}),
            "origins": _safe_counter(bucket["origins"], {"inference", "legacy", "other"}),
            "scene_frame_formats": _safe_counter(bucket["scene_frame_format"]),
            "analysis_formats": _safe_counter(bucket["analysis_format"]),
        })
    return result
