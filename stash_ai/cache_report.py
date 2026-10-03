"""Offline cache coverage and an advisory, targeted inference/rethreshold plan.

Coverage proves retained evidence and its internal completion index, not freshness
against current model assets or live Stash media. Existing compact metadata is
streamed; older evidence is decoded only to derive metadata in the preview DB.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from itertools import groupby
import json
from pathlib import Path
import sqlite3
from types import SimpleNamespace
import zlib

from tqdm import tqdm

from stash_ai.state_results import KINDS

STATUSES = (
    'ready', 'rethreshold', 'missing', 'partial', 'refresh_required',
    'legacy_only', 'insufficient', 'inconsistent',
)
REUSABLE = frozenset(('ready', 'rethreshold'))
RESTART = frozenset(('legacy_only', 'insufficient', 'inconsistent'))


def _keys(store, table, columns):
    return list(store.connection.execute(
        f'SELECT {columns} FROM {store.table(table)} WHERE scope=?', (store.scope,)))


def scene_ids(store, requested):
    """Explicit IDs may have no local state; otherwise include working-state-only scenes."""
    if requested:
        return sorted(set(map(str, requested)))
    known = set()
    for table in ('results', 'result_index', 'refresh_requests', 'checkpoints',
                  'snapshots', 'legacy_imports', 'recovery_publications'):
        known.update(row[0] for row in _keys(store, table, 'DISTINCT scene'))
    return sorted(known)


def _metadata_rows(store, scenes):
    from stash_ai.state_metadata import FORMAT, decode
    from stash_ai.state_thresholds import overlay

    with store.connection:
        store.connection.execute('CREATE TEMP TABLE IF NOT EXISTS cache_report_scenes (scene TEXT PRIMARY KEY)')
        store.connection.execute('DELETE FROM cache_report_scenes')
        store.connection.executemany('INSERT INTO cache_report_scenes VALUES (?)', ((scene,) for scene in scenes))
    rows = store.connection.execute(
        'SELECT r.scene,r.model,r.kind,m.format,m.data,t.threshold FROM ' +
        store.table('results') + ' AS r LEFT JOIN ' + store.table('result_metadata') +
        ' AS m USING(scope,scene,model,kind) LEFT JOIN ' + store.table('result_thresholds') +
        ' AS t USING(scope,scene,model,kind) WHERE r.scope=? '
        'AND r.scene IN (SELECT scene FROM cache_report_scenes) '
        'ORDER BY r.scene,r.model,r.kind', (store.scope,))
    for scene, model, kind, version, data, threshold in rows:
        if kind not in KINDS:
            continue
        segments = None
        if version == FORMAT:
            try:
                segments = overlay(decode(data), kind, threshold)
            except (TypeError, ValueError):
                pass
        if segments is None:
            try:
                segments = store.result_metadata(scene, kind, model)
            except (TypeError, ValueError, KeyError, IndexError, zlib.error) as exc:
                yield scene, model, kind, None, f'Unreadable evidence: {type(exc).__name__}; run --db-check.'
                continue
        yield scene, model, kind, segments, None


def _inspect(app, config, kind, segments, index, refresh, checkpoint, error):
    from stash_ai.scene_cache import scene_threshold, validate_scene
    from stash_ai.temporal import marker_floor
    from stash_ai.thresholds import _score, indexed_signature, marker_threshold, validate_marker_blocklist

    def result(status, reason, **details):
        return dict(status=status, reason=reason, **details)

    if error:
        return result('inconsistent', error, unreadable=True)
    if refresh:
        return result('refresh_required', 'An outstanding refresh invalidates the saved result.')
    if segments is None:
        if index is not None:
            return result('inconsistent', 'Completion index exists without evidence; run --db-check.')
        return result('partial' if checkpoint else 'missing',
                      'Checkpoint exists; compatibility is checked when inference resumes.' if checkpoint
                      else 'No saved result for this output.')
    if not segments:
        return result('inconsistent', 'A result row has no segments; this is not completed zero-detection inference.')
    inference = next((segment for segment in reversed(segments)
                      if segment['meta'].get('origin') == 'inference'), None)
    if inference is None:
        return result('legacy_only', 'Imported output does not prove completed model inference.')
    signature = inference['meta'].get('inference_signature')
    expected = (1, int(any(segment['populated'] for segment in segments)), indexed_signature(inference))
    if not isinstance(signature, str) or not signature or index != expected:
        return result('inconsistent', 'Evidence and completion index disagree; run --db-check.')

    threshold = config.scene_threshold if kind == 'scene' else config.marker_threshold
    needed = (threshold * (app.TEMPORAL_POLICY.scene_support_ratio if app.TEMPORAL_POLICY.enabled else 1.)
              if kind == 'scene' else marker_floor(app.TEMPORAL_POLICY, threshold))
    requested = getattr(config, 'scene_cache_floor' if kind == 'scene' else 'marker_cache_floor')
    required_floor = min(needed, requested) if requested is not None else needed
    floors, effective = [], []
    try:
        for segment in segments:
            meta = segment['meta']
            if kind == 'scene':
                validate_scene(app, segment, threshold, has_frames=segment['has_frames'])
                floor, saved = meta['scene_score_floor'], scene_threshold(segment)
            else:
                if segment['mode'] != 'frames' or meta.get('origin') != 'inference':
                    raise ValueError('Imported marker ranges cannot prove retained frame coverage.')
                validate_marker_blocklist(segment, config.source.key)
                floor = meta.get('marker_score_floor', meta.get('marker_threshold'))
                saved = marker_threshold(segment)
            if not _score(saved) or not _score(floor):
                raise ValueError('Invalid saved threshold or retention floor; run --db-check.')
            if floor > required_floor:
                raise ValueError(f'Retained floor {floor:g} exceeds required {required_floor:g}.')
            floors.append(floor)
            effective.append(saved)
    except (KeyError, ValueError) as exc:
        return result('insufficient', str(exc), required_floor=required_floor,
                      target_threshold=threshold)
    changed = any(value != threshold for value in effective)
    return result('rethreshold' if changed else 'ready',
                  'Retained evidence supports YAML thresholds; apply rethreshold without inference.' if changed
                  else 'Completed evidence supports the selected thresholds and retention.',
                  retained_floor_max=max(floors), required_floor=required_floor,
                  target_threshold=threshold, saved_thresholds=sorted(set(effective)))


def _publication_dependencies(app, evidence, selected):
    """Merge reads unselected cached models at their saved effective thresholds."""
    from stash_ai.scene_cache import scene_threshold, validate_scene
    from stash_ai.models import SOURCES
    from stash_ai.thresholds import validate_marker_cache

    dependencies = []
    context = SimpleNamespace(TEMPORAL_POLICY=app.TEMPORAL_POLICY)
    for (model, kind), (segments, error) in sorted(evidence.items()):
        if model in selected:
            continue
        try:
            if error:
                raise ValueError(error)
            if model not in SOURCES:
                raise ValueError('Unknown cached model; run --db-check.')
            if kind == 'markers':
                if any(segment['mode'] not in ('frames', 'spans') for segment in segments):
                    raise ValueError('Unsupported stored marker evidence; run --db-check.')
                validate_marker_cache(context, {model: segments})
            else:
                # Unchanged old aggregates remain publishable. Validate only
                # projection required by an override or changed support policy.
                for segment in segments:
                    meta = segment['meta']
                    threshold = scene_threshold(segment)
                    support_changed = (app.TEMPORAL_POLICY.enabled
                        and ('scene_support_floor' in meta or meta.get('scene_frames_format') == 1)
                        and meta.get('scene_support_floor') != threshold * app.TEMPORAL_POLICY.scene_support_ratio)
                    if threshold != meta.get('scene_threshold') or support_changed:
                        validate_scene(app, segment, threshold, has_frames=segment['has_frames'])
        except (KeyError, TypeError, ValueError) as exc:
            dependencies.append(dict(model=model, kind=kind, reason=str(exc),
                resolution=f'Include {model} with --models and regenerate the report; '
                           'resolve unreadable or invalid evidence with --db-check first.'))
    return dependencies


def build_report(app, store, scenes, settings, command_context=()):
    """Inspect one scene at a time; count outputs separately from scene/model pairs."""
    from stash_ai.preview import check_source
    from stash_ai.state_metadata import backfill_batch

    models = [config.source.key for config in settings]
    indexes = {(scene, model, kind): (inference, populated, signature)
               for scene, model, kind, inference, populated, signature in
               _keys(store, 'result_index', 'scene,model,kind,inference,populated,signature')}
    refreshes = set(_keys(store, 'refresh_requests', 'scene,model,kind'))
    checkpoints = {(scene, model): dict(processed=processed, total=total, compatibility='unverified')
                   for scene, model, processed, total in
                   _keys(store, 'checkpoints', 'scene,model,processed,total')}
    pending = store.pending_publication_scenes()
    recovery = {scene for scene, in _keys(store, 'recovery_publications', 'DISTINCT scene')}
    outside_refresh = {scene for scene, model, kind in refreshes if model not in models and kind in KINDS}
    counts = {model: {kind: Counter() for kind in KINDS} for model in models}
    pairs = {model: Counter() for model in models}
    issues, operations = [], defaultdict(list)
    complete_scenes = 0
    grouped = iter(groupby(_metadata_rows(store, scenes), key=lambda row: row[0]))
    current = next(grouped, None)
    with backfill_batch(store):
        for scene in tqdm(scenes, desc='Cache coverage', unit='scene'):
            while current is not None and current[0] < scene:
                # Advancing groupby discards unselected scenes without keeping payloads.
                current = next(grouped, None)
            if current is None or current[0] != scene:
                evidence = {}
            else:
                evidence = {(model, kind): (segments, error)
                            for _, model, kind, segments, error in current[1]}
                current = next(grouped, None)
            dependencies = _publication_dependencies(app, evidence, models)
            inspected = []
            for config in settings:
                model = config.source.key
                checkpoint = checkpoints.get((scene, model))
                outputs = {}
                for kind in KINDS:
                    segments, error = evidence.get((model, kind), (None, None))
                    outputs[kind] = _inspect(app, config, kind, segments,
                        indexes.get((scene, model, kind)), (scene, model, kind) in refreshes,
                        checkpoint, error)
                    counts[model][kind][outputs[kind]['status']] += 1
                blockers = []
                if store.cleanup_pending:
                    blockers.append('pending_global_cleanup')
                if scene in pending:
                    blockers.append('pending_publication')
                if scene in outside_refresh:
                    blockers.append('refresh_required_for_unselected_model')
                if dependencies:
                    blockers.append('unusable_evidence_for_unselected_model')
                if any(value.get('unreadable') for value in outputs.values()):
                    blockers.append('unreadable_evidence_run_db_check')
                if any(outputs[kind]['status'] in RESTART and not getattr(app,
                       'REPLACE_SCENE_TAGS' if kind == 'scene' else 'REPLACE_MARKERS') for kind in KINDS):
                    blockers.append('replacement_disabled')
                inspected.append((model, checkpoint, outputs, blockers))
            # Every generated command attempts publication. A blocked selected
            # pair therefore blocks its peers, regardless of model iteration order.
            blocking_models = sorted(model for model, _, _, blockers in inspected if blockers)
            scene_ready = True
            for model, checkpoint, outputs, blockers in inspected:
                statuses = {value['status'] for value in outputs.values()}
                if blocking_models and not blockers:
                    blockers.append('blocked_selected_model')
                reusable = statuses <= REUSABLE
                ready = statuses == {'ready'}
                pairs[model]['evidence_ready_pairs'] += ready
                pairs[model]['reusable_pairs'] += reusable
                pairs[model]['blocked_pairs'] += bool(blockers)
                scene_ready &= ready and not blockers
                if ready and not blockers:
                    continue
                row = dict(scene_id=scene, model=model, outputs=outputs, blockers=blockers)
                if dependencies:
                    row['publication_dependencies'] = dependencies
                if 'blocked_selected_model' in blockers:
                    row['blocking_models'] = blocking_models
                if checkpoint:
                    row['checkpoint'] = checkpoint
                issues.append(row)
                if blockers:
                    continue
                if not reusable:
                    # A missing/refresh output forces normal inference to check
                    # checkpoint compatibility. Otherwise force replacement of
                    # unusable completed evidence, even with an orphan checkpoint.
                    restart = bool(statuses & RESTART) and not (
                        checkpoint and statuses & {'missing', 'partial', 'refresh_required'})
                    action = 'restart_inference' if restart else 'resume_or_infer'
                    pairs[model]['inference_pairs'] += 1
                else:
                    action = 'rethreshold'
                    pairs[model]['rethreshold_pairs'] += 1
                operations[action, model].append(scene)
            complete_scenes += scene_ready
    check_source(store)
    plan = []
    blocked_scenes = {row['scene_id'] for row in issues if row['blockers']}
    for action in ('resume_or_infer', 'restart_inference', 'rethreshold'):
        for model in models:
            targets = [scene for scene in operations.get((action, model), ()) if scene not in blocked_scenes]
            for start in range(0, len(targets), 50):
                batch = targets[start:start + 50]
                argv = ['python', 'stash_ai.py', *command_context, '--output-mode', 'merged', '--models', model]
                for scene in batch:
                    argv.extend(('--scene-id', scene))
                if action == 'restart_inference':
                    argv.append('--restart')
                elif action == 'rethreshold':
                    argv.extend(('--merge-only', '--rethreshold-scenes', '--rethreshold-markers'))
                else:
                    argv.append('--resume')
                plan.append(dict(action=action, model=model, scene_ids=batch, argv=argv,
                                 uses_inference=action != 'rethreshold'))
    recovery_scenes = set(scenes) & recovery
    actionable_scenes = sorted(({scene for targets in operations.values() for scene in targets}
                               | recovery_scenes) - blocked_scenes)
    # Refresh publication after all backfills, including cross-model votes and Occurrence.
    for start in range(0, len(actionable_scenes), 50):
        batch = actionable_scenes[start:start + 50]
        argv = ['python', 'stash_ai.py', *command_context, '--output-mode', 'merged', '--merge-only']
        for scene in batch:
            argv.extend(('--scene-id', scene))
        plan.append(dict(action='merge_after_backfill', scene_ids=batch, argv=argv, uses_inference=False))
    summary = [dict(model=model, **{key: pairs[model][key] for key in (
        'evidence_ready_pairs', 'reusable_pairs', 'inference_pairs', 'rethreshold_pairs', 'blocked_pairs')},
        outputs={kind: {status: counts[model][kind][status] for status in STATUSES} for kind in KINDS})
        for model in models]
    return dict(format_version=1, mode='cache-coverage',
        semantics=dict(scene_universe='Explicit --scene-id values, or scenes known to the current local DB scope; unseen Stash scenes are not counted.',
                       freshness='Inference identity beyond thresholds/retention, model assets, vocabulary, VTT and sprites are unverified offline.',
                       coverage='Both scene and marker inference evidence, completion index, YAML thresholds and requested retention; completed empty detections count as ready.',
                       integrity='Compact metadata is trusted when valid; --db-check performs a deep payload check.',
                       checkpoint='Presence/progress only; inference verifies compatibility before resuming.',
                       plan='Advisory argv arrays in execution order; never executed by this report. Running them can publish to Stash.',
                       blockers='Blockers exclude the entire scene from all plan commands. Finish interrupted publication/cleanup, '
                                'include reported unselected dependency models with --models, or resolve DB errors, then regenerate.'),
        totals=dict(scenes=len(scenes), fully_ready_scenes=complete_scenes,
                    scene_model_pairs=len(scenes) * len(models),
                    **{key: sum(row[key] for row in summary) for key in (
                        'evidence_ready_pairs', 'reusable_pairs', 'inference_pairs', 'rethreshold_pairs', 'blocked_pairs')}),
        models=summary, issues=issues, plan=plan,
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
    from stash_ai.state import StateStore, scope_for
    from stash_ai.temporal import load_policy

    try:
        if not Path(args.state_db).expanduser().is_file():
            raise FileNotFoundError(f'State database does not exist: {args.state_db}')
        settings = app.load_model_settings(args.model_config, args.models, None, True)
        local = SimpleNamespace(**{**vars(app), 'TEMPORAL_POLICY':
                                    load_policy(app, args.temporal_policy, not args.no_temporal)})
        context = []
        for flag, path in (('--state-db', args.state_db), ('--model-config', args.model_config),
                           ('--temporal-policy', args.temporal_policy),
                           ('--marker-core-exclusions', args.marker_core_exclusions),
                           ('--scene-tag-exclusions', args.scene_tag_exclusions)):
            context.extend((flag, str(Path(path).expanduser().resolve())))
        if args.no_temporal:
            context.append('--no-temporal')
        with StateStore(args.state_db, scope_for(app.STASH_URL), True, allow_cleanup=True) as store:
            report = build_report(local, store, scene_ids(store, args.scene_id), settings, context)
        payload = json.dumps(report, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + '\n'
        destination = Path(args.cache_report).expanduser().resolve()
        destination.parent.mkdir(parents=True, exist_ok=True)
        with destination.open('x', encoding='utf-8', newline='\n') as stream:
            stream.write(payload)
        print(format_summary(report))
        print(f'Cache report written to {destination}. No Stash requests or inference were performed.')
        return 0
    except (OSError, ValueError, RuntimeError, sqlite3.Error) as exc:
        print('Cache report failed: ' + str(exc))
        return 1
