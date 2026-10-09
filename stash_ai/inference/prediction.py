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
    if checkpoint:
        checkpoint.keep = 1 if app.AGGREGATION == 'max' else app.TOP_K_FRAMES
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
        checkpoint.processed = checkpoint.saved_processed = checkpoint.batches = checkpoint.resumed_frames = 0
        checkpoint.pending.clear()
        checkpoint.top_scores = None
        if report is not None:
            report.counts['checkpoint_timeline_resets'] += 1
            report.detail('Checkpoint timeline is inconsistent; rebuilding only this scene/model.')
        processed, top_scores, saved = 0, np.empty((0, len(labels)), np.float32), []
        checkpoint.restore_reason = 'timeline_inconsistent'
        checkpoint._notify_restore()
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
                            checkpoint, prefetch, prepare_workers, report,
                            inference_executor, batch_results=None, prepared_inputs=None,
                            transform_executor=None, result_prefetch=32):
    import numpy as np
    from stash_ai.inference.batches import predicted_batches

    settings, labels = tagger.settings, tagger.labels
    keep = 1 if app.AGGREGATION == "max" else min(app.TOP_K_FRAMES, len(cues))
    batches = batch_results if batch_results is not None else predicted_batches(
        app,
        tagger,
        stash,
        paths,
        cues[state.processed:],
        prefetch=prefetch,
        prepare_workers=prepare_workers,
        report=report,
        inference_executor=inference_executor,
        prepared_inputs=prepared_inputs,
        transform_executor=transform_executor,
        result_prefetch=result_prefetch,
    )
    try:
        for result, batch_cues in batches:
            from contextlib import nullcontext
            timer = (report.measure('inference_postprocess', model=settings.source.key, units=len(batch_cues), unit='frames')
                     if report is not None else nullcontext())
            with timer:
                combined = np.concatenate((state.top_scores, result), axis=0)
                count = min(keep, len(combined))
                state.top_scores = np.partition(
                    combined, len(combined) - count, axis=0
                )[-count:]

                # One ordering per model row, then apply each retention floor.
                batch_frames, batch_scene = [], []
                floors = ([state.marker_floor] if app.CREATE_MARKERS else []) + ([state.scene_floor] if state.retain_scene else [])
                for cue, row in zip(batch_cues, result):
                    matches = select_scores(row, labels, min(floors), 0) if floors else []
                    if app.CREATE_MARKERS:
                        batch_frames.append(FramePrediction(cue, tuple(
                            item for item in matches if item[1] >= state.marker_floor)))
                    if state.retain_scene:
                        batch_scene.append(dict(start=cue.start, end=cue.end,
                            tags=[item for item in matches if item[1] >= state.scene_floor]))
                state.scene_frames.extend(batch_scene)
                if state.accumulator is not None:
                    state.accumulator.add(result, batch_cues)
                state.frame_predictions.extend(batch_frames)
                state.processed += len(batch_cues)
                if report is not None:
                    report.counts['inference_processed_frames'] += len(batch_cues)

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
        from stash_ai.runtime.cleanup import cleanup
        cleanup(([checkpoint.flush] if checkpoint else []) + [batches.close], report=report)

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
                  report=None, prepare_workers: int = 1, inference_executor=None,
                  state=None, batch_results=None, prepared_inputs=None,
                  transform_executor=None, before_finalize=None, result_prefetch: int = 32
                  ) -> tuple[list[tuple[str, float]], list[FramePrediction]]:
    """Return raw retained evidence; publication applies exclusions and limits."""
    if not cues:
        raise ValueError("Cannot predict a scene without VTT cues.")
    state = state if state is not None else _restore_prediction_state(
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
        inference_executor=inference_executor,
        batch_results=batch_results,
        prepared_inputs=prepared_inputs,
        transform_executor=transform_executor,
        result_prefetch=result_prefetch,
    )
    if before_finalize is not None:
        before_finalize()
    from contextlib import nullcontext
    timer = (report.measure('inference_scene_finalize', model=tagger.settings.source.key,
                            units=1, unit='scenes') if report is not None else nullcontext())
    with timer:
        return _finalize_predictions(app, tagger, state, analysis=analysis)
