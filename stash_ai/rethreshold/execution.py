"""Run complete preflight before applying saved-evidence threshold changes."""
from stash_ai.runtime.reasons import EvidenceError, Reason, PendingPublication
from collections import Counter
from itertools import batched

from tqdm import tqdm

from . import scenes as scene_updates
from .targets import threshold_failure_reason, threshold_targets
from .plans import RecordAccess, PendingPolicy, ThresholdRequest, write_plan



def _plan_target(app, store, scene, config, available, pending_scenes, *, scenes, markers):
    if scenes:
        prepared = scene_updates.plan(
            app, store, scene, config,
            request=ThresholdRequest(
                kinds=tuple(kind for kind in ('scene', 'markers') if kind in available),
                access=RecordAccess.METADATA, pending=PendingPolicy.CHECKED_BY_EXECUTOR,
                marker_check_scene=False if markers else None))
        blocked = bool(prepared)
    else:
        prepared = scene_updates.plan(
            app, store, scene, config, request=ThresholdRequest(
                kinds=('markers',), access=RecordAccess.METADATA,
                pending=PendingPolicy.CHECKED_BY_EXECUTOR))
        # Marker-only preflight rejects pending publications even for no-ops.
        blocked = True
    if blocked and str(scene) in pending_scenes:
        message = ('Finish the pending merge or use --abandon-pending before changing thresholds.'
                   if scenes else 'Finish the pending merge or use --abandon-pending first.')
        raise PendingPublication(message)
    return prepared


def run(app, store, scene_ids, settings, report, *, scenes=True, markers=False):
    """Preflight all targets; apply per pair only when every check succeeds."""
    if not scenes and not markers:
        raise ValueError('At least one output kind is required for rethreshold.')
    if scenes:
        requested = ('scene', 'markers') if markers else ('scene',)
        mode = 'scene + marker thresholds' if markers else 'scene thresholds'
    else:
        requested, mode = ('markers',), 'marker thresholds'
    issue_phase = 'scene thresholds' if scenes else 'marker thresholds'
    updated_key = 'scene_thresholds_updated' if scenes else 'marker_thresholds_updated'
    targets, skipped, selected = threshold_targets(store, scene_ids, settings, requested)
    errors, reasons = 0, Counter()
    prepared_targets = []
    pending_scenes = store.pending_publication_scenes(scene for scene, _, _ in targets)
    from stash_ai.storage.metadata import backfill_batch
    bar = tqdm(total=len(targets), desc=f'Check {mode}', unit='pair')
    try:
        for group in batched(targets, 128):
            with backfill_batch(store), store.metadata.batch(
                    (scene, config.source.key) for scene, config, _ in group):
                for scene, config, available in group:
                    try:
                        prepared = _plan_target(app, store, scene, config, available, pending_scenes,
                                                scenes=scenes, markers=markers)
                        prepared_targets.append((scene, config.source.key, prepared))
                    except (ValueError, RuntimeError) as exc:
                        report.issue(issue_phase, config.source.key, scene, exc, str(exc))
                        reasons[threshold_failure_reason(exc)] += 1
                        errors += 1
                    bar.update(1)
    finally:
        bar.close()
    if errors:
        report.record_threshold_preflight(
            len(targets), errors, reasons, applied=0,
            selected=selected, skipped_no_evidence=skipped)
        return dict(errors=errors)
    changed = 0
    bar = tqdm(total=len(prepared_targets), desc=f'Apply {mode}', unit='pair')
    try:
        for group in batched(prepared_targets, 128):
            with store.connection:
                for scene, model, prepared in group:
                    write_plan(store, scene, model, prepared)
                    changed += bool(prepared)
                    bar.update(1)
    finally:
        bar.close()
    report.record_threshold_preflight(
        len(targets), 0, applied=changed,
        selected=selected, skipped_no_evidence=skipped)
    return {updated_key: changed}

