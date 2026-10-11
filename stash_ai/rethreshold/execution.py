"""Run complete preflight before applying saved-evidence threshold changes."""
from stash_ai.runtime.console import progress as console_bar
from stash_ai.runtime.console import activity, phase, status
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


def run(app, store, scene_ids, settings, report, *, scenes=True, markers=False, spool_limit_bytes=0):
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
    from stash_ai.runtime.cleanup import closing
    from .spool import SpoolUsage
    import sqlite3
    import json
    from time import perf_counter

    scene_list = tuple(str(scene) for scene in scene_ids)
    selected = len(scene_list) * len(settings)
    checked, errors, reasons = 0, 0, Counter()
    pending_scenes = store.pending_publication_scenes(scene_list)
    report.preview = bool(getattr(store, 'dry_run', False))
    models = {config.source.key: Counter(selected=len(scene_list)) for config in settings}
    report.workload['rethreshold_models'] = models
    inspected, planned = 0, 0
    def show():
        status(report, preflight=f'scanned={inspected}/{selected} checked={checked} planned={planned} no_evidence={inspected-checked} errors={errors}',
               blocker='BLOCKED: ' + ', '.join(f'{name}={count}' for name, count in reasons.most_common(3)) if errors else None,
               applied=0, **{'model_' + key: f'checked={row["checked"]} planned={row["planned"]} no_evidence={row["no_evidence"]} errors={row["errors"]}'
                             for key, row in models.items()})
    # SQLite owns all validated plans until preflight succeeds globally. Unlike
    # a Python list, only a small page cache and one scene group remain in RAM.
    with closing(sqlite3.connect(''), report=report, name='release rethreshold spool') as spool:
        spool.execute('PRAGMA cache_size=-2048')
        spool.execute('PRAGMA temp_store=FILE')
        spool.execute('CREATE TABLE prepared (id INTEGER PRIMARY KEY, scene TEXT, model TEXT, outputs TEXT)')
        usage = SpoolUsage(spool, report, spool_limit_bytes)
        bar = console_bar(tqdm, total=selected, desc=f'Check {mode}', unit='pair')
        try:
            with phase(report, 'threshold preflight', errors=lambda: errors):
                show()
                for group, scene_count in iter_threshold_groups(store, scene_list, settings, requested):
                    group_checked, queued, write_seconds = Counter(), 0, 0.
                    with backfill_batch(store), store.metadata.batch(
                            (scene, config.source.key) for scene, config, _ in group):
                        for scene, config, available in group:
                            key = config.source.key
                            checked += 1
                            models[key]['checked'] += 1
                            group_checked[key] += 1
                            try:
                                with activity(report, 'plan threshold pair', model=key, scene=scene, output=mode):
                                    prepared = _plan_target(app, store, scene, config, available, pending_scenes,
                                                            scenes=scenes, markers=markers)
                                if prepared:
                                    models[key]['planned'] += 1
                                    planned += 1
                                    started = perf_counter()
                                    spool.execute('INSERT INTO prepared (scene,model,outputs) VALUES (?,?,?)',
                                        (scene, key, json.dumps(prepared.outputs, allow_nan=False, separators=(',', ':'))))
                                    write_seconds += perf_counter()-started
                                    queued += 1
                                else:
                                    models[key]['unchanged'] += 1
                            except (ValueError, RuntimeError) as exc:
                                if isinstance(exc, PendingPublication):
                                    report.publication_pending_observed = True
                                report.issue(issue_phase, key, scene, exc, str(exc))
                                reasons[threshold_failure_reason(exc)] += 1
                                errors += 1
                                models[key]['errors'] += 1
                            finally:
                                inspected += 1
                                bar.update(1)
                                show()
                    started = perf_counter()
                    spool.commit()
                    usage.flushed(queued, write_seconds+perf_counter()-started)
                    for key in models:
                        models[key]['no_evidence'] += scene_count-group_checked[key]
                    absent = scene_count * len(settings)-len(group)
                    inspected += absent
                    bar.update(absent)
                    show()
        except BaseException as error:
            usage.row['state'] = 'interrupted' if isinstance(error, KeyboardInterrupt) else type(error).__name__
            report.record_threshold_preflight(checked, errors+int(isinstance(error, Exception)), reasons, applied=0,
                                              selected=selected, skipped_no_evidence=inspected-checked)
            report.workload['rethreshold_unvisited'] = selected-inspected
            report.write('Threshold preflight stopped; applied=0 (' + type(error).__name__ + ').')
            raise
        finally:
            bar.close()
        skipped = selected - checked
        if errors:
            usage.row['state'] = 'blocked'
            report.record_threshold_preflight(checked, errors, reasons, applied=0,
                                              selected=selected, skipped_no_evidence=skipped)
            report.write(f'Threshold preflight failed; applied=0, blocked={errors}.')
            return dict(errors=errors)
        changed = 0
        count = spool.execute('SELECT COUNT(*) FROM prepared').fetchone()[0]
        usage.row['state'] = 'apply'
        preview = bool(getattr(store, 'dry_run', False))
        apply_row = dict(mode='planned' if preview else 'committed', committed=0, planned=0,
                         pending_batch=0, remaining=count, failed_batches=0)
        report.workload['rethreshold_apply'] = apply_row
        bar = console_bar(tqdm, total=count, desc=f'Apply {mode}' + (' (planned)' if preview else ''), unit='pair')
        try:
            with phase(report, 'threshold apply', mode=apply_row['mode']):
                cursor = spool.execute('SELECT scene,model,outputs FROM prepared ORDER BY id')
                for group in batched(cursor, 128):
                    apply_row['pending_batch'] = len(group)
                    status(report, apply=f'pending={len(group)} {apply_row["mode"]}={changed} remaining={count-changed}')
                    with activity(report, 'write/commit threshold batch'), store.connection:
                        for scene, model, serialized in group:
                            prepared = ThresholdPlan(json.loads(serialized), RecordAccess.METADATA)
                            write_plan(store, scene, model, prepared)
                    # A rolled-back batch must never advance confirmed work.
                    changed += len(group)
                    apply_row.update(pending_batch=0, remaining=count-changed)
                    apply_row['planned' if preview else 'committed'] = changed
                    report.counts['threshold_pairs_planned' if preview else 'threshold_pairs_committed'] += len(group)
                    bar.update(len(group))
                    status(report, apply=f'{apply_row["mode"]}={changed} remaining={count-changed}',
                           applied=0 if preview else changed)
        except BaseException:
            apply_row['failed_batches'] += 1
            apply_row['pending_batch'] = 0
            usage.row['state'] = 'apply stopped'
            raise
        finally:
            bar.close()
            report.record_threshold_preflight(checked, 0, applied=changed,
                                              selected=selected, skipped_no_evidence=skipped)
        usage.row['state'] = 'finished'
    report.record_threshold_preflight(checked, 0, applied=changed,
                                      selected=selected, skipped_no_evidence=skipped)
    return {updated_key: changed}

