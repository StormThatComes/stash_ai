"""Offline cache coverage and an advisory, targeted inference/rethreshold plan.

Coverage proves retained evidence and its internal completion index, not freshness
against current model assets or live Stash media. Existing compact metadata is
streamed; older evidence is decoded only to derive metadata in the preview DB.
"""
from __future__ import annotations

from itertools import groupby
from pathlib import Path
import sqlite3
import zlib

from tqdm import tqdm

from stash_ai.storage.results import KINDS

from stash_ai.reports.cache_assessment import assess_scene
from stash_ai.reports.cache_coverage import CoverageAccumulator
from stash_ai.reports.cache_plan import build_plan
from stash_ai.reports.io import write_json_report
from stash_ai.reports.runtime import report_context, report_store, require_database


def _keys(store, table, columns, scenes=None):
    from stash_ai.storage.selection import selected_rows
    return list(selected_rows(store,
        f'SELECT {columns} FROM {store.table(table)} WHERE scope=?', (store.scope,), scenes))


def scene_ids(store, requested):
    """Explicit IDs may have no local state; otherwise include working-state-only scenes."""
    if requested:
        return sorted(set(map(str, requested)))
    known = set()
    for table in ('results', 'result_index', 'refresh_requests', 'checkpoints',
                  'snapshots', 'recovery_publications', 'input_checks'):
        known.update(row[0] for row in _keys(store, table, 'DISTINCT scene'))
    return sorted(known)


def _metadata_rows(store, scenes):
    from stash_ai.storage.metadata import FORMAT, decode
    from stash_ai.storage.thresholds import overlay

    with store.connection:
        store.connection.execute('CREATE TEMP TABLE IF NOT EXISTS cache_report_scenes (scene TEXT PRIMARY KEY)')
        store.connection.execute('DELETE FROM cache_report_scenes')
        store.connection.executemany('INSERT INTO cache_report_scenes VALUES (?)', ((scene,) for scene in scenes))
    rows = store.connection.execute(
        'WITH metadata AS (SELECT m.* FROM cache_report_scenes AS c CROSS JOIN ' +
        store.table('result_metadata') + ' AS m WHERE m.scope=? AND m.scene=c.scene), '
        'thresholds AS (SELECT t.* FROM cache_report_scenes AS c CROSS JOIN ' +
        store.table('result_thresholds') + ' AS t WHERE t.scope=? AND t.scene=c.scene) '
        'SELECT r.scene,r.model,r.kind,m.format,m.data,t.threshold FROM cache_report_scenes AS c CROSS JOIN ' +
        store.table('results') + ' AS r LEFT JOIN metadata AS m '
        'ON m.scope=r.scope AND m.scene=r.scene AND m.model=r.model AND m.kind=r.kind '
        'LEFT JOIN thresholds AS t '
        'ON t.scope=r.scope AND t.scene=r.scene AND t.model=r.model AND t.kind=r.kind '
        'WHERE r.scope=? AND r.scene=c.scene ORDER BY r.scene,r.model,r.kind', (store.scope,) * 3)
    for scene, model, kind, version, data, threshold in rows:
        if kind not in KINDS:
            continue
        record = None
        if version == FORMAT:
            try:
                record = overlay(decode(data), kind, threshold)
            except (TypeError, ValueError):
                pass
        if record is None:
            try:
                record = store.result_metadata(scene, kind, model)
            except (TypeError, ValueError, KeyError, IndexError, zlib.error) as exc:
                yield scene, model, kind, None, f'Unreadable evidence: {type(exc).__name__}; run --db-check.'
                continue
        yield scene, model, kind, record, None


def _scene_evidence(store, scenes):
    """Stream compact metadata, retaining only the current scene's evidence."""
    grouped = iter(groupby(_metadata_rows(store, scenes), key=lambda row: row[0]))
    current = next(grouped, None)
    for scene in tqdm(scenes, desc='Cache coverage', unit='scene'):
        while current is not None and current[0] < scene:
            current = next(grouped, None)
        if current is None or current[0] != scene:
            evidence = {}
        else:
            evidence = {(model, kind): (record, error)
                        for _, model, kind, record, error in current[1]}
            current = next(grouped, None)
        yield scene, evidence


def build_report(app, store, scenes, settings, command_context=()):
    """Inspect one scene at a time; count outputs separately from scene/model pairs."""
    from stash_ai.storage.preview import check_source
    from stash_ai.storage.metadata import backfill_batch
    from stash_ai.storage.input_checks import selected as input_checks

    models = [config.source.key for config in settings]
    indexes = {(scene, model, kind): (populated, signature)
               for scene, model, kind, populated, signature in
               _keys(store, 'result_index', 'scene,model,kind,populated,signature', scenes)}
    refreshes = set(_keys(store, 'refresh_requests', 'scene,model,kind', scenes))
    checkpoints = {(scene, model): dict(processed=processed, total=total, compatibility='unverified')
                   for scene, model, processed, total in
                   _keys(store, 'checkpoints', 'scene,model,processed,total', scenes)}
    pending = store.pending_publication_scenes(scenes)
    recovery = {scene for scene, in _keys(store, 'recovery_publications', 'DISTINCT scene', scenes)}
    deferred = input_checks(store, scenes)
    outside_refresh = {scene for scene, model, kind in refreshes if model not in models and kind in KINDS}
    coverage = CoverageAccumulator(models)
    with backfill_batch(store):
        for scene, evidence in _scene_evidence(store, scenes):
            coverage.record(assess_scene(
                app, scene, settings, evidence, indexes, refreshes, checkpoints,
                cleanup_pending=store.cleanup_pending, publication_pending=scene in pending,
                outside_refresh=scene in outside_refresh, input_checks=deferred.get(scene, {})))
    check_source(store)
    recovery_scenes = set(scenes) & recovery
    plan = build_plan(models, coverage.operations, coverage.blocked_scenes,
                      recovery_scenes, command_context)
    return dict(format_version=1, mode='cache-coverage',
        semantics=dict(scene_universe='Explicit --scene-id values, or scenes known to the current local DB scope; unseen Stash scenes are not counted.',
                       freshness='Inference identity beyond thresholds/retention, model assets, vocabulary, VTT and sprites are unverified offline.',
                       coverage='Both scene and marker inference evidence, completion index, YAML thresholds and requested retention; completed empty detections count as ready.',
                       evidence_ready_pairs='Stored evidence coverage only. Known deferred input checks block fully_ready_scenes and every plan command, including checks for unselected models.',
                       integrity='Compact metadata is trusted when valid; --db-check performs a deep payload check.',
                       checkpoint='Presence/progress only; inference verifies compatibility before resuming.',
                       plan='Advisory argv arrays in execution order; never executed by this report. Running them can publish to Stash.',
                       blockers='Blockers exclude the entire scene from all plan commands. Finish interrupted publication/cleanup, '
                                'verify missing/unreadable/deleted media without requesting fresh predictions, '
                                'include reported unselected dependency models with --models, or resolve DB errors, then regenerate.'),
        **coverage.payload(len(scenes)), plan=plan,
        publication_recovery_scene_ids=sorted(recovery_scenes),
        targets=[dict(model=config.source.key, scene_threshold=config.scene_threshold,
                      marker_threshold=config.marker_threshold, scene_cache_floor=config.scene_cache_floor,
                      marker_cache_floor=config.marker_cache_floor) for config in settings])


def format_summary(report):
    totals = report['totals']
    lines = [f"Cache coverage: {totals['fully_ready_scenes']:,}/{totals['scenes']:,} locally known/requested scenes ready.",
             'model       ready pairs  reusable pairs  inference  rethreshold  blocked']
    for row in report['models']:
        lines.append(f"{row['model']:<10} {row['evidence_ready_pairs']:>11,} {row['reusable_pairs']:>15,} "
                     f"{row['inference_pairs']:>10,} {row['rethreshold_pairs']:>12,} {row['blocked_pairs']:>8,}")
    if any(row.get('publication_dependencies') for row in report['issues']):
        lines.append('Unselected cached models block publication; see issues.publication_dependencies and include them with --models.')
    lines.append('Offline coverage; current model/media freshness is unverified. Plan commands are not executed.')
    return '\n'.join(lines)


def run(app, args):

    try:
        require_database(args)
        settings = app.services.load_model_settings(args.model_config, args.models, None, True)
        local = report_context(app, args)
        context = []
        for flag, path in (('--state-db', args.state_db), ('--model-config', args.model_config),
                           ('--temporal-policy', args.temporal_policy),
                           ('--marker-core-exclusions', args.marker_core_exclusions),
                           ('--scene-tag-exclusions', args.scene_tag_exclusions)):
            context.extend((flag, str(Path(path).expanduser().resolve())))
        with report_store(app, args) as store:
            report = build_report(local, store, scene_ids(store, args.scene_id), settings, context)
        destination = write_json_report(report, args.cache_report)
        print(format_summary(report))
        print(f'Cache report written to {destination}. No Stash requests or inference were performed.')
        return 0
    except (OSError, ValueError, RuntimeError, sqlite3.Error) as exc:
        print('Cache report failed: ' + str(exc))
        return 1
