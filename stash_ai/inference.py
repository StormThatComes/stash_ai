"""Media preparation and local scene inference with explicit runtime settings."""
from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from io import BytesIO
from typing import Any, Iterator
from urllib.parse import urljoin

from stash_ai.transport import absolute_stash_url
from stash_ai.domain import Cue, FramePrediction
from stash_ai.vtt import format_time, parse_vtt_cues


def evenly_sample(items: list, count: int) -> list:
    if count <= 0 or len(items) <= count:
        return items
    if count == 1:
        return [items[len(items) // 2]]
    return [items[round(i * (len(items) - 1) / (count - 1))] for i in range(count)]


def fetch_cues(app, stash, paths: dict) -> tuple[list[Cue], int]:
    if not paths.get("vtt"):
        raise ValueError("No VTT path; generate Stash sprites before running scene marker analysis.")
    content = stash.get_media(absolute_stash_url(paths["vtt"], base_url=app.STASH_URL))
    cues = parse_vtt_cues(content.decode("utf-8-sig"))
    # Resolve before sampling: a sample containing only one sheet from a
    # multi-sheet VTT must not be mistaken for a single-sheet scene.
    single_sheet = len({cue.image for cue in cues}) == 1
    cues = [
        Cue(
            cue.start,
            cue.end,
            absolute_stash_url(paths["sprite"], base_url=app.STASH_URL)
            if single_sheet and paths.get("sprite")
            else urljoin(absolute_stash_url(paths["vtt"], base_url=app.STASH_URL), cue.image),
            cue.rect,
        )
        for cue in cues
    ]
    return evenly_sample(cues, app.MAX_FRAMES_PER_SCENE), len(cues)


def iter_sprite_frames(stash, paths: dict, cues: list[Cue]) -> Iterator[tuple[Cue, Any]]:
    from PIL import Image

    sheets: OrderedDict[str, Any] = OrderedDict()
    try:
        for cue in cues:
            url = cue.image
            if url not in sheets:
                with Image.open(BytesIO(stash.get_media(url))) as opened:
                    sheets[url] = opened.convert("RGB")
                if len(sheets) > 2:
                    _, old = sheets.popitem(last=False)
                    old.close()
            sheets.move_to_end(url)
            sprite = sheets[url]
            x, y, width, height = cue.rect
            if x + width > sprite.width or y + height > sprite.height:
                raise ValueError(f"VTT rectangle exceeds sprite bounds at {format_time(cue.start)}.")
            frame = sprite.crop((x, y, x + width, y + height))
            try:
                yield cue, frame
            finally:
                frame.close()
    finally:
        for sprite in sheets.values():
            sprite.close()


def prepare_image(app, image: Any, target_size: int) -> Any:
    import torchvision.transforms.functional as TVF
    from PIL import Image

    side = max(image.size)
    with Image.new("RGB", (side, side), (255, 255, 255)) as padded:
        padded.paste(image, ((side - image.width) // 2, (side - image.height) // 2))
        with padded.resize((target_size, target_size), Image.Resampling.BICUBIC) as resized:
            tensor = TVF.pil_to_tensor(resized).float().div_(255.0)
    return TVF.normalize(tensor, mean=app.JOYTAG_CLIP_MEAN, std=app.JOYTAG_CLIP_STD)


def select_scores(scores: list[float], labels: list[str], threshold: float, limit: int) -> list[tuple[str, float]]:
    import numpy as np

    if isinstance(scores, np.ndarray):
        # Match Python float comparisons exactly at the threshold, including
        # float32 values just below a float64 policy boundary.
        indexes = np.flatnonzero(scores.astype(np.float64, copy=False) >= threshold)
        matches = [(labels[int(i)], float(scores[i])) for i in indexes]
    else:
        matches = [
            (label, float(score))
            for label, score in zip(labels, scores)
            if score >= threshold
        ]
    matches.sort(key=lambda item: (-item[1], item[0]))
    return matches[:limit] if limit else matches


@dataclass
class _PredictionState:
    processed: int
    top_scores: Any
    frame_predictions: list[FramePrediction]
    scene_frames: list[dict]
    accumulator: Any
    retain_scene: bool
    scene_floor: float
    marker_floor: float
    policy: Any


def _restore_prediction_state(app, tagger, cues, *, raw_scene, checkpoint, report):
    import numpy as np
    from stash_ai.temporal import SceneAccumulator, marker_floor
    from stash_ai.retention import scene_cache_floor, marker_cache_floor

    settings, labels = tagger.settings, tagger.labels
    policy = app.TEMPORAL_POLICY
    retain_scene = raw_scene and app.APPLY_SCENE_TAGS
    scene_floor = scene_cache_floor(settings, policy)
    # Separate mode has no persistent evidence; use the publication floor there.
    retained_marker_floor = (marker_cache_floor(settings, policy) if raw_scene
                             else marker_floor(policy, settings.marker_threshold))
    processed, top_scores, saved = (
        checkpoint.restore()
        if checkpoint
        else (0, np.empty((0, len(labels)), np.float32), [])
    )
    accumulator = (
        SceneAccumulator(
            settings.scene_threshold,
            policy,
            checkpoint.analysis if checkpoint else None,
        )
        if policy.enabled and app.APPLY_SCENE_TAGS
        else None
    )
    if accumulator is not None and accumulator.frames != processed:
        raise ValueError("Checkpoint scene statistics do not match completed cues; use --restart.")
    if checkpoint and (app.CREATE_MARKERS or retain_scene) and len(saved) != processed:
        raise ValueError("Checkpoint frame count does not match completed cues.")

    frame_predictions = (
        [
            FramePrediction(
                cues[index],
                tuple((tag, score) for tag, score in frame["tags"]),
            )
            for index, frame in enumerate(saved)
        ]
        if app.CREATE_MARKERS
        else []
    )
    scene_frames = (
        [
            dict(start=cues[i].start, end=cues[i].end, tags=row["scene_tags"])
            for i, row in enumerate(saved)
        ]
        if retain_scene
        else []
    )
    if processed and report is not None:
        report.counts["checkpoint_resumes"] += 1
        report.detail(
            f"  Resume {settings.source.key}: {processed}/{len(cues)} frames already processed."
        )

    return _PredictionState(
        processed,
        top_scores,
        frame_predictions,
        scene_frames,
        accumulator,
        retain_scene,
        scene_floor,
        retained_marker_floor,
        policy,
    )


def _run_prediction_batches(app, tagger, stash, paths, cues, state, *,
                            checkpoint, prefetch, prepare_workers):
    import numpy as np
    from stash_ai.progress import prepared_batches
    from stash_ai.models import validate_scores

    settings, labels = tagger.settings, tagger.labels
    keep = 1 if app.AGGREGATION == "max" else min(app.TOP_K_FRAMES, len(cues))
    batches = prepared_batches(
        app,
        tagger,
        stash,
        paths,
        cues[state.processed:],
        prefetch,
        prepare_workers,
    )
    try:
        for prepared, batch_cues in batches:
            result = validate_scores(
                tagger.predict_batch(prepared),
                len(prepared),
                labels,
                settings.source.title,
            )
            combined = np.concatenate((state.top_scores, result), axis=0)
            count = min(keep, len(combined))
            state.top_scores = np.partition(
                combined, len(combined) - count, axis=0
            )[-count:]

            batch_frames = []
            if app.CREATE_MARKERS:
                for cue, row in zip(batch_cues, result):
                    matches = select_scores(
                        row,
                        labels,
                        state.marker_floor,
                        0,
                    )
                    batch_frames.append(FramePrediction(cue, tuple(matches)))

            batch_scene = (
                [
                    dict(
                        start=cue.start,
                        end=cue.end,
                        tags=select_scores(row, labels, state.scene_floor, 0),
                    )
                    for cue, row in zip(batch_cues, result)
                ]
                if state.retain_scene
                else []
            )
            state.scene_frames.extend(batch_scene)
            if state.accumulator is not None:
                state.accumulator.add(result, batch_cues)
            state.frame_predictions.extend(batch_frames)
            state.processed += len(batch_cues)

            if checkpoint:
                checkpoint.record(
                    state.processed,
                    state.top_scores,
                    batch_frames,
                    analysis=state.accumulator.state() if state.accumulator is not None else None,
                    scene_frames=batch_scene if state.retain_scene else None,
                )
    finally:
        # Save the last fully completed batch on Ctrl+C/inference/preparation errors.
        try:
            if checkpoint:
                checkpoint.flush()
        finally:
            batches.close()

    if state.processed != len(cues) or not len(state.top_scores):
        raise RuntimeError("Incomplete inference; scene sync was not started.")


def _finalize_predictions(app, tagger, scene_exclusions, state, *, raw_scene, analysis):
    import numpy as np
    from stash_ai.temporal import scene_reason

    settings, labels = tagger.settings, tagger.labels
    scene_scores = np.sort(state.top_scores, axis=0)[::-1].mean(axis=0)
    statistics = (
        state.accumulator.statistics(labels)
        if state.accumulator is not None
        else {}
    )
    if analysis is not None:
        analysis["scene_statistics"] = statistics
        if state.retain_scene:
            analysis.update(
                scene_frames=state.scene_frames,
                scene_score_floor=state.scene_floor,
            )

    if state.retain_scene and getattr(settings, 'scene_cache_floor', None) is not None:
        # Keep exact aggregate scores for every retained label, including those
        # below today's support floor. Temporal statistics still use today's
        # publication policy; future rethreshold rebuilds them from the frames.
        candidates = {tag for row in state.scene_frames for tag, _ in row['tags']}
        predictions = [(tag, float(scene_scores[i])) for i, tag in enumerate(labels)
                       if tag in candidates]
        predictions.sort(key=lambda item: (-item[1], item[0]))
    elif state.accumulator is not None:
        predictions = [
            (labels[index], float(scene_scores[index]))
            for index in state.accumulator.items
        ]
        predictions = [
            (tag, score)
            for tag, score in predictions
            if raw_scene
            or (
                tag not in scene_exclusions
                and scene_reason(
                    score,
                    statistics[tag],
                    settings.scene_threshold,
                    state.policy,
                )
            )
        ]
        predictions.sort(key=lambda item: (-item[1], item[0]))
    elif state.retain_scene:
        candidates = {
            tag
            for row in state.scene_frames
            for tag, _ in row["tags"]
        }
        predictions = [
            (tag, float(scene_scores[i]))
            for i, tag in enumerate(labels)
            if tag in candidates
        ]
        predictions.sort(key=lambda item: (-item[1], item[0]))
    else:
        predictions = [
            (tag, score)
            for tag, score in select_scores(
                scene_scores,
                labels,
                settings.scene_threshold,
                0,
            )
            if raw_scene or tag not in scene_exclusions
        ]

    if app.MAX_TAGS and not raw_scene:
        predictions = predictions[:app.MAX_TAGS]
    if scene_exclusions and not raw_scene:
        state.frame_predictions = [
            FramePrediction(
                frame.cue,
                tuple(
                    (tag, score)
                    for tag, score in frame.tags
                    if tag not in scene_exclusions
                ),
            )
            for frame in state.frame_predictions
        ]
    return predictions, state.frame_predictions


def predict_scene(app, tagger: Any, stash, paths: dict, cues: list[Cue],
                  scene_exclusions: frozenset[str] = frozenset(), *, raw_scene: bool = False,
                  checkpoint=None, prefetch: int = 0, analysis: dict | None = None,
                  report=None, prepare_workers: int = 1
                  ) -> tuple[list[tuple[str, float]], list[FramePrediction]]:
    """Run local inference while preserving resumable checkpoint semantics."""
    if not cues:
        raise ValueError("Cannot predict a scene without VTT cues.")
    state = _restore_prediction_state(
        app,
        tagger,
        cues,
        raw_scene=raw_scene,
        checkpoint=checkpoint,
        report=report,
    )
    _run_prediction_batches(
        app,
        tagger,
        stash,
        paths,
        cues,
        state,
        checkpoint=checkpoint,
        prefetch=prefetch,
        prepare_workers=prepare_workers,
    )
    return _finalize_predictions(
        app,
        tagger,
        scene_exclusions,
        state,
        raw_scene=raw_scene,
        analysis=analysis,
    )
