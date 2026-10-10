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
    from .targets import iter_threshold_groups
    from .plans import ThresholdPlan
    from stash_ai.storage.metadata import backfill_batch
    from contextlib import closing
    import sqlite3
    import json

    scene_list = tuple(str(scene) for scene in scene_ids)
    selected = len(scene_list) * len(settings)
    checked, errors, reasons = 0, 0, Counter()
    pending_scenes = store.pending_publication_scenes(scene_list)
    # SQLite owns all validated plans until preflight succeeds globally. Unlike
    # a Python list, only a small page cache and one scene group remain in RAM.
    with closing(sqlite3.connect('')) as spool:
        spool.execute('PRAGMA cache_size=-2048')
        spool.execute('PRAGMA temp_store=FILE')
        spool.execute('CREATE TABLE prepared (id INTEGER PRIMARY KEY, scene TEXT, model TEXT, outputs TEXT)')
        bar = tqdm(total=selected, desc=f'Check {mode}', unit='pair')
        try:
            for group, scene_count in iter_threshold_groups(store, scene_list, settings, requested):
                with backfill_batch(store), store.metadata.batch(
                        (scene, config.source.key) for scene, config, _ in group):
                    for scene, config, available in group:
                        checked += 1
                        try:
                            prepared = _plan_target(app, store, scene, config, available, pending_scenes,
                                                    scenes=scenes, markers=markers)
                            if prepared:
                                spool.execute('INSERT INTO prepared (scene,model,outputs) VALUES (?,?,?)',
                                    (scene, config.source.key,
                                     json.dumps(prepared.outputs, allow_nan=False, separators=(',', ':'))))
                        except (ValueError, RuntimeError) as exc:
                            report.issue(issue_phase, config.source.key, scene, exc, str(exc))
                            reasons[threshold_failure_reason(exc)] += 1
                            errors += 1
                spool.commit()
                # Includes pairs skipped for lack of evidence.
                bar.update(scene_count * len(settings))
        finally:
            bar.close()
        skipped = selected - checked
        if errors:
            report.record_threshold_preflight(checked, errors, reasons, applied=0,
                                              selected=selected, skipped_no_evidence=skipped)
            return dict(errors=errors)
        changed = 0
        count = spool.execute('SELECT COUNT(*) FROM prepared').fetchone()[0]
        bar = tqdm(total=count, desc=f'Apply {mode}', unit='pair')
        try:
            cursor = spool.execute('SELECT scene,model,outputs FROM prepared ORDER BY id')
            for group in batched(cursor, 128):
                with store.connection:
                    for scene, model, serialized in group:
                        prepared = ThresholdPlan(json.loads(serialized), RecordAccess.METADATA)
                        write_plan(store, scene, model, prepared)
                        changed += 1
                        bar.update(1)
        finally:
            bar.close()
    report.record_threshold_preflight(checked, 0, applied=changed,
                                      selected=selected, skipped_no_evidence=skipped)
    return {updated_key: changed}

