"""Scene predictions, retained frame scores and resumable batch execution."""
from __future__ import annotations

from .scene_predictions import retained_scores, candidate_scores

from dataclasses import dataclass
from typing import Any, TYPE_CHECKING

from stash_ai.domain import Cue, FramePrediction

if TYPE_CHECKING:
    import numpy as np


def select_scores(scores: np.ndarray, labels: list[str], threshold: float, limit: int) -> list[tuple[str, float]]:
    import numpy as np

    # Match Python float comparisons exactly at the threshold, including
    # float32 values just below a float64 policy boundary.
    indexes = np.flatnonzero(scores.astype(np.float64, copy=False) >= threshold)
    matches = [(labels[int(i)], float(scores[i])) for i in indexes]
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


def _restore_prediction_state(app, tagger, cues, *, checkpoint, report):
    import numpy as np
    from stash_ai.evidence.temporal.statistics import SceneAccumulator
    from stash_ai.evidence.retention import scene_cache_floor, marker_cache_floor

    settings, labels = tagger.settings, tagger.labels
    policy = app.TEMPORAL_POLICY
    from stash_ai.storage.progress import output_kinds
    retain_scene = 'scene' in output_kinds(app)
    scene_floor = scene_cache_floor(settings, policy)
    retained_marker_floor = marker_cache_floor(settings, policy)
    processed, top_scores, saved = (
        checkpoint.restore()
        if checkpoint
        else (0, np.empty((0, len(labels)), np.float32), [])
    )
    if checkpoint and len(saved) == processed and any(
            frame['start'] != cue.start or frame['end'] != cue.end
            for frame, cue in zip(saved, cues[:processed])):
        from stash_ai.inference.checkpoints import clear_checkpoint
        with checkpoint.store.connection:
            clear_checkpoint(checkpoint.store, checkpoint.scene, checkpoint.model)
        checkpoint.analysis = None
        checkpoint.processed = checkpoint.saved_processed = checkpoint.batches = 0
        checkpoint.pending.clear()
        checkpoint.top_scores = None
        if report is not None:
            report.counts['checkpoint_timeline_resets'] += 1
            report.detail('Checkpoint timeline is inconsistent; rebuilding only this scene/model.')
        return _restore_prediction_state(app, tagger, cues, checkpoint=checkpoint, report=report)
    accumulator = (
        SceneAccumulator(
            settings.scene_threshold,
            policy,
            checkpoint.analysis if checkpoint else None,
        )
        if retain_scene
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
    if report is not None:
        report.work('checkpoint_resumed' if processed else 'fresh')

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
                            checkpoint, prefetch, prepare_workers, report):
    import numpy as np
    from stash_ai.inference.preparation import prepared_batches
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
        report=report,
    )
    try:
        for prepared, batch_cues in batches:
            from contextlib import nullcontext
            timer = (report.measure('inference_batch', model=settings.source.key, units=len(prepared), unit='frames')
                     if report is not None else nullcontext())
            with timer:
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


def _finalize_predictions(app, tagger, state, *, analysis):
    import numpy as np

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

    if state.retain_scene and settings.scene_cache_floor is not None:
        # Explicit retention keeps exact aggregate scores even below today's
        # support floor. Future rethreshold rebuilds statistics from frames.
        predictions = retained_scores(scene_scores, labels, state.scene_frames)
    elif state.accumulator is not None:
        predictions = candidate_scores(scene_scores, labels, state.accumulator.items)
    else:
        predictions = select_scores(scene_scores, labels, settings.scene_threshold, 0)

    return predictions, state.frame_predictions


def predict_scene(app, tagger: Any, stash, paths: dict, cues: list[Cue],
                  *, checkpoint=None, prefetch: int = 0, analysis: dict | None = None,
                  report=None, prepare_workers: int = 1
                  ) -> tuple[list[tuple[str, float]], list[FramePrediction]]:
    """Return raw retained evidence; publication applies exclusions and limits."""
    if not cues:
        raise ValueError("Cannot predict a scene without VTT cues.")
    state = _restore_prediction_state(
        app,
        tagger,
        cues,
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
        report=report,
    )
    return _finalize_predictions(
        app,
        tagger,
        state,
        analysis=analysis,
    )
