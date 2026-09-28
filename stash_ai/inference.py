"""Media preparation and local scene inference.

The compatibility surface remains in stash_ai.cli. Functions that depend on
runtime CLI settings receive the live app module so tests and callers can keep
overriding CLI configuration and helper functions.
"""
from __future__ import annotations

from collections import OrderedDict
from io import BytesIO
from typing import Any, Iterator
from urllib.parse import urljoin

from stash_ai.client import absolute_stash_url
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
    return app.evenly_sample(cues, app.MAX_FRAMES_PER_SCENE), len(cues)


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


def predict_scene(app, tagger: Any, stash, paths: dict, cues: list[Cue],
                  scene_exclusions: frozenset[str] = frozenset(), *, raw_scene: bool = False,
                  checkpoint=None, prefetch: int = 0, analysis: dict | None = None,
                  report=None, prepare_workers: int = 1
                  ) -> tuple[list[tuple[str, float]], list[FramePrediction]]:
    import numpy as np
    from stash_ai.progress import prepared_batches
    from stash_ai.taggers import validate_scores
    from stash_ai.temporal import SceneAccumulator, marker_floor, scene_reason

    if not cues:
        raise ValueError("Cannot predict a scene without VTT cues.")

    settings, labels = tagger.settings, tagger.labels
    policy = app.TEMPORAL_POLICY
    retain_scene = raw_scene and app.APPLY_SCENE_TAGS
    scene_floor = settings.scene_threshold * (
        policy.scene_support_ratio if policy.enabled else 1.0
    )
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

    keep = 1 if app.AGGREGATION == "max" else min(app.TOP_K_FRAMES, len(cues))
    batches = prepared_batches(
        app,
        tagger,
        stash,
        paths,
        cues[processed:],
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
            combined = np.concatenate((top_scores, result), axis=0)
            count = min(keep, len(combined))
            next_top = np.partition(combined, len(combined) - count, axis=0)[-count:]

            batch_frames = []
            if app.CREATE_MARKERS:
                for cue, row in zip(batch_cues, result):
                    matches = app.select_scores(
                        row,
                        labels,
                        marker_floor(policy, settings.marker_threshold),
                        0,
                    )
                    matches = [
                        (tag, score)
                        for tag, score in matches
                        if tag not in app.MARKER_TAG_BLOCKLIST
                    ]
                    batch_frames.append(FramePrediction(cue, tuple(matches)))

            batch_scene = (
                [
                    dict(
                        start=cue.start,
                        end=cue.end,
                        tags=app.select_scores(row, labels, scene_floor, 0),
                    )
                    for cue, row in zip(batch_cues, result)
                ]
                if retain_scene
                else []
            )
            scene_frames.extend(batch_scene)
            if accumulator is not None:
                accumulator.add(result, batch_cues)
            top_scores = next_top
            frame_predictions.extend(batch_frames)
            processed += len(batch_cues)
            if checkpoint:
                checkpoint.record(
                    processed,
                    top_scores,
                    batch_frames,
                    analysis=accumulator.state() if accumulator is not None else None,
                    scene_frames=batch_scene if retain_scene else None,
                )
    finally:
        # Save the last fully completed batch on Ctrl+C/inference/preparation errors.
        try:
            if checkpoint:
                checkpoint.flush()
        finally:
            batches.close()

    if processed != len(cues) or not len(top_scores):
        raise RuntimeError("Incomplete inference; scene sync was not started.")

    scene_scores = np.sort(top_scores, axis=0)[::-1].mean(axis=0)
    statistics = accumulator.statistics(labels) if accumulator is not None else {}
    if analysis is not None:
        analysis["scene_statistics"] = statistics
        if retain_scene:
            analysis.update(scene_frames=scene_frames, scene_score_floor=scene_floor)

    if accumulator is not None:
        predictions = [
            (labels[index], float(scene_scores[index]))
            for index in accumulator.items
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
                    policy,
                )
            )
        ]
        predictions.sort(key=lambda item: (-item[1], item[0]))
    elif retain_scene:
        candidates = {tag for row in scene_frames for tag, _ in row["tags"]}
        predictions = [
            (tag, float(scene_scores[i]))
            for i, tag in enumerate(labels)
            if tag in candidates
        ]
        predictions.sort(key=lambda item: (-item[1], item[0]))
    else:
        predictions = [
            (tag, score)
            for tag, score in app.select_scores(
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
        frame_predictions = [
            FramePrediction(
                frame.cue,
                tuple(
                    (tag, score)
                    for tag, score in frame.tags
                    if tag not in scene_exclusions
                ),
            )
            for frame in frame_predictions
        ]
    return predictions, frame_predictions
