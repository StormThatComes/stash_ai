"""Scene qualification, isolated peaks and compatible-model disagreement."""
from collections import defaultdict
from .signals import _score, _candidate, _flag


def _scene_signals(app, store, scene, hits, rows, options, scene_policies):
    from stash_ai.evidence.identity import marker_segment_signature
    from stash_ai.evidence.cache_views import views
    from stash_ai.evidence.qualification import scene_qualification

    _, projected = views(app, store.records(scene, "scene", borrow=True))
    # Index retained labels once per scene, rather than scanning the vocabulary
    # for every accepted tag. Absence from this map is never a negative vote.
    peers = defaultdict(list)
    metadata = {}
    for model, segment in projected.items():
        meta = segment["meta"]
        metadata[model, marker_segment_signature(segment)] = meta
        identity = meta.get("frame_input_signature")
        if identity:
            for entry in segment["items"]:
                peers[identity, entry["tag"]].append((model, meta, entry))

    for hit in hits:
        shared = hit.get("derived") == "shared_scene_frames"
        sources = hit.get("source_evidence", {}) if shared else {hit["source"]: hit}
        for model, own in sources.items():
            row = _candidate(rows, model, hit["tag"])
            if shared:
                metas = [metadata[model, signature]
                         for signature in hit["scene_context"].get(model, ())
                         if (model, signature) in metadata]
                thresholds = {meta.get("scene_threshold") for meta in metas}
                threshold = next(iter(thresholds)) if len(thresholds) == 1 else None
                score, stats = own.get("max_score"), None
            else:
                threshold = hit["meta"].get("scene_threshold")
                stats = hit.get("statistics")
                score = stats.get("max_score") if stats else hit.get("score")
            cutoff = threshold
            if not shared and hit.get("qualification") == "strong_peak" and _score(threshold):
                cutoff += (1 - threshold) * app.TEMPORAL_POLICY.scene_strong_margin
            detail = dict(kind="scene", score=score, threshold=cutoff,
                          qualification=hit.get("qualification", "aggregate"))
            if stats and "peak_time" in stats:
                detail["time"] = stats["peak_time"]
            if _score(score) and _score(cutoff) and score <= cutoff + options.margin + 1e-12:
                _flag(row, "weak_or_borderline", detail)
            if (not shared and hit.get("qualification") == "strong_peak" and stats
                    and stats.get("high_hits") == 1 and stats.get("sampled_frames", 0) >= 2):
                _flag(row, "isolated_peak", {**detail, "high_hits": 1,
                                             "sampled_frames": stats["sampled_frames"]})
            if shared:
                continue
            identity = hit["meta"].get("frame_input_signature")
            if not identity:
                continue
            comparisons = defaultdict(list)
            for peer, meta, entry in peers.get((identity, hit["tag"]), ()):
                if (peer != model and hit["tag"] not in scene_policies[peer]
                        and _score(entry.get("score")) and _score(meta.get("scene_threshold"))):
                    comparisons[peer].append((meta, entry))
            for peer, values in comparisons.items():
                if all(scene_qualification(entry,
                                    meta["scene_threshold"], app.TEMPORAL_POLICY) is None
                       for meta, entry in values):
                    meta, entry = max(values, key=lambda value: value[1]["score"])
                    _flag(row, "model_disagreement", {
                        **detail, "peer_model": peer, "peer_score": entry["score"],
                        "peer_threshold": meta["scene_threshold"]})
