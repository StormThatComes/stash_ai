"""Run complete preflight before applying saved-evidence threshold changes."""
from collections import Counter

from tqdm import tqdm

from . import markers as marker_updates, scenes as scene_updates
from .targets import threshold_failure_reason, threshold_targets


def _check_target(app, store, scene, config, available, pending_scenes, *, scenes, markers):
    if scenes:
        outputs = scene_updates.plan(
            app, store, scene, config,
            scenes='scene' in available, markers='markers' in available,
            direct=True, marker_check_scene=False if markers else None,
            pending_checked=True, metadata_only=True)
        blocked = bool(outputs)
    else:
        marker_updates.validate(app, store, scene, config, direct=True, metadata_only=True)
        # Marker-only preflight historically rejects pending publications even
        # when the requested threshold is already current.
        blocked = True
    if blocked and str(scene) in pending_scenes:
        message = ('Finish the pending merge or use --abandon-pending before changing thresholds.'
                   if scenes else 'Finish the pending merge or use --abandon-pending first.')
        raise RuntimeError(message)


def _apply_target(app, store, scene, config, available, *, scenes, markers):
    if not scenes:
        return marker_updates.apply(
            app, store, scene, config, direct=True, pending_checked=True, metadata_only=True)
    outputs = scene_updates.plan(
        app, store, scene, config,
        scenes='scene' in available, markers='markers' in available,
        direct=True, marker_check_scene=False if markers else None,
        pending_checked=True, metadata_only=True)
    scene_updates.apply(store, scene, config.source.key, outputs, metadata_only=True)
    return bool(outputs)


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
    pending_scenes = store.pending_publication_scenes()
    from stash_ai.state_metadata import backfill_batch
    with backfill_batch(store):
        for scene, config, available in tqdm(targets, desc=f'Check {mode}', unit='pair'):
            try:
                _check_target(app, store, scene, config, available, pending_scenes,
                              scenes=scenes, markers=markers)
            except (ValueError, RuntimeError) as exc:
                report.issue(issue_phase, config.source.key, scene, exc, str(exc))
                reasons[threshold_failure_reason(exc)] += 1
                errors += 1
    if errors:
        report.record_threshold_preflight(
            len(targets), errors, reasons, applied=0,
            selected=selected, skipped_no_evidence=skipped)
        return dict(errors=errors)
    changed = 0
    for scene, config, available in tqdm(targets, desc=f'Apply {mode}', unit='pair'):
        changed += _apply_target(app, store, scene, config, available, scenes=scenes, markers=markers)
    report.record_threshold_preflight(
        len(targets), 0, applied=changed,
        selected=selected, skipped_no_evidence=skipped)
    return {updated_key: changed}
