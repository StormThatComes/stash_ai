"""Finite recognition and retained-portrait orchestration; no tag runtime or hook."""
from collections import Counter
from dataclasses import asdict, dataclass, replace
import os
from urllib.parse import urlsplit

from stash_ai.runtime.cleanup import closing, cleanup
from stash_ai.runtime.console import present, status, activity
from stash_ai.runtime.report import RunReport
from .config import digest
from .database import FaceDatabase, encoded
from .models import ASSETS, ensure_assets, load_backend, namespace
from .backend import LazyBackend
from .context import policy_scope, portrait_reference_scope
from .portrait_refresh import LazyAssets, retained_targets
from .prefetch import ScenePrefetch
from .media import SpriteInput
from .scanning import scan_scene
from .clustering import matching_policy
from .rebuild import rebuild_selected
from .scene_sync import reconcile_unlinks, remove_stale_links, remove_other_model_links, reconcile_legacy_merges
from .reference_usage import usage_line
from .publication import publish_person, ensure_person
from .merge_publication import canonical_targets
from .stash import RecognitionStash
from .names import load_names, plan_names, starter
from .metadata import enrich
from .portrait_pipeline import Pipeline as PortraitPipeline
from .portrait_threads import NativeBudget
from .video import VideoFrames
from .visibility import FaceVisibility, LazyVisibility
from .expression import FaceExpression, LazyExpression, required as expression_required
from .diagnostics import snapshot as error_snapshot, render as render_errors, message as safe_error
from .progress import RunProgress, WorkProgress, frame_progress, work, tracking


def instance_scope(url):
    parts = urlsplit(url)
    if (parts.scheme not in ('http', 'https') or not parts.hostname or parts.username or
            parts.password or parts.query or parts.fragment):
        raise ValueError('STASH_URL must be an HTTP(S) base URL without credentials/query/fragment.')
    return digest([parts.scheme.lower(), parts.hostname.lower(),
                   parts.port or (443 if parts.scheme == 'https' else 80), parts.path.rstrip('/')])


@dataclass
class Services:
    client: object = RecognitionStash
    assets: object = ensure_assets
    backend: object = load_backend
    source: object = SpriteInput
    metadata_config: object = load_names
    metadata_plan: object = plan_names
    metadata: object = enrich
    decoder: object = VideoFrames
    visibility: object = FaceVisibility
    expression: object = FaceExpression


def _publish_selection(database, client, report, config, scope, namespace, selected, rebuilt, ready, counts, key, run, progress,*,guard=None):
    db = database.connection
    for item in selected:
        scene, scan = item['scene'], item['scan']
        groups = rebuilt[scan]
        accepted = sorted({g['person'] for g in groups if g['status'] == 'accepted'})
        counts['groups_withheld'] += sum(g['status'] == 'withheld' for g in groups)
        counts['observations_withheld'] += sum(g['observations'] for g in groups if g['status'] == 'withheld')
        try:
            from .publication_queue import current,acknowledge
            from .publication_queue import snapshot,validate,body_policy
            captured,desired,signature=snapshot(db,scope,namespace,scene,config)
            if captured['scan']!=scan or desired!=accepted:
                raise ValueError('Rebuilt recognition results disagree with current scene decisions.')
            queued=current(db,scope,namespace,scene)
            revision=queued['revision'] if queued and queued['scan']==scan else None
            entry=dict(captured,people=desired,signature=signature,revision=revision,
                       policy=matching_policy(config),body=body_policy(config))
            def check():
                validate(db,scope,namespace,entry,config=config)
                if guard is not None:guard(item)
            check()
            from .publication_batch import ensure_people,publish_people
            # Resolve all desired identities before unlinking old model profiles.
            with work('Prepare scene identities',len(accepted),unit='person') as prepared:
                targets=ensure_people(database,client,report,scope,scene,accepted,guard=check,progress=prepared.advance)
            with activity(report,'publish scene identities',scene=scene):
                publish_people(database,client,report,scope,scene,targets,guard=check)
            with tracking(
                targets.items(),
                'Link scene identities',
                unit='person',
            ) as tracked:
                for person, performer in tracked:
                    counts['people_linked_or_observed'] += 1
                    target = ready.setdefault(person, {'person': person, 'performer': performer, 'scans': []})
                    target['scans'].append(item)
                    evidence = db.execute("SELECT json_extract(payload,'$.reference_context') FROM candidates "
                                          "WHERE scan=? AND policy=? AND person=? AND status='accepted' LIMIT 1",
                                          (scan, matching_policy(config), person)).fetchone()
                    if evidence:
                        target.setdefault('matching', {})[scan] = {'policy': matching_policy(config),
                                                                  'reference_context': evidence[0], 'person': person}
            counts['scene_links_removed'] += remove_stale_links(database, client, report, scope, scene,
                                                                 set(targets.values()), namespace,guard=check)
            counts['other_model_links_removed'] += remove_other_model_links(
                database, client, report, scope, scene, set(targets.values()), namespace,guard=check)
            if check:check()
            with db:acknowledge(db,scope,namespace,scene,revision)
            counts['scenes_processed'] += 1
            database.progress(run, 'matching',
                              {'scene': scene, 'groups': groups, 'policy': 'cold_selected_replay'})
        except KeyboardInterrupt:
            raise
        except Exception as error:
            counts['people_failed'] += 1
            report.optional('recognition_error', report.issue, 'recognition', 'publication', scene, error, safe_error(error, key))
        progress.advance(linked=counts['people_linked_or_observed'], failed=counts['people_failed'])


def execute(args, config, *, services=None):
    from .profiling import pipeline_scope
    with pipeline_scope(config.data['execution']['profile_rebuild']):
        return _execute(args, config, services=services)


def _execute(args, config, *, services=None):
    services = services or Services()
    retry_images = bool(getattr(args, 'retry_unresolved_images', False))
    portraits_only = getattr(args, 'command', 'run') == 'portraits'
    performer_ids = list(dict.fromkeys(getattr(args, 'performer_id', None) or []))
    url = os.getenv('STASH_URL', 'http://localhost:9999')
    key = os.getenv('STASH_API_KEY', '')
    scope = instance_scope(url)
    # Optional extensions read immutable config before any Stash mutations.
    extra = services.metadata_config(config, scope) if services.metadata_config and not portraits_only else None
    from tqdm import tqdm
    report = RunReport(args.progress_style == 'verbose', tqdm.write)
    counts = Counter()
    rejection_totals={kind:Counter() for kind in ('face','body','person_evidence_unavailable')}
    references = {}
    from .sprite_cache import scope as sprite_scope
    with sprite_scope(config), policy_scope(), FaceDatabase(config.database) as database:
        client = services.client(url=url, api_key=key, timeout=config.data['execution']['timeout_seconds'])
        with closing(client, report=report, name='close recognition HTTP'), present(report, args.progress_style, status_position=2):
            client.report = report
            ui = RunProgress(report, portraits=portraits_only, metadata=bool(services.metadata))
            with ui.phase('Recognition selection'):
                ready = {}
                if portraits_only:
                    ready = retained_targets(database, config, scope, args.scene_id or None,
                                             performer_ids=performer_ids or None)
                    source = args.scene_id or sorted({s['scene'] for t in ready.values() for s in t['scans']}, key=int)
                else:
                    source = client.scene_ids() if args.all_scenes else dict.fromkeys(args.scene_id)
                identifiers = []
                seen = set()
                with tracking(
                    source,
                    'Read recognition scene IDs',
                    unit='scene',
                    position=0,
                ) as selected_ids:
                    for identifier in selected_ids:
                        if identifier not in seen:
                            identifiers.append(identifier)
                            seen.add(identifier)
            snapshot = config.snapshot()
            if portraits_only or retry_images:
                snapshot['invocation'] = {**({'command': 'portraits'} if portraits_only else {}),
                                          **({'performer_ids': performer_ids} if performer_ids else {}),
                                          **({'retry_unresolved_images': True} if retry_images else {})}
            if extra is not None:
                snapshot['names'] = extra.snapshot()
            run = database.start_run(scope, identifiers, snapshot)
            outcome, failure = 'success', None
            visibility = None
            expression = None
            backend = None
            person_collector = None
            inputs = None
            try:
                with ui.phase('Recognition assets'):
                    with work('Configure lazy model loading', 1, unit='step', position=0) as assets:
                        paths = LazyAssets(config, services.assets, report)
                        assets.advance()
                    visibility = LazyVisibility(config, paths, services.visibility)
                    expression = LazyExpression(config, paths, services.expression)
                    model_namespace = namespace(config)
                    provenance = {name: asdict(ASSETS[name]) for name in
                                  (config.data['models']['detector'], config.data['models']['recognizer'])}
                    if config.data['models']['recognizer'] == 'auraface':
                        from .aura import CONTRACT
                        provenance['preprocessing'] = CONTRACT
                    if not portraits_only:
                        with database.connection:
                            database.connection.execute('INSERT OR IGNORE INTO models VALUES (?,?)',
                                                        (model_namespace, encoded(provenance)))
                    def loaded(value):
                        database.progress(run, 'backend', {'namespace': model_namespace,
                            'embedding_device': value.device, 'detector_device': getattr(value, 'detector_device', 'cpu'),
                            'dimensions': value.dimensions, 'runtime': getattr(value, 'provenance', {})})
                    def load_models(config, paths):
                        with activity(report, 'load detection and recognition models'):
                            paths.prepare_identity()
                            return services.backend(config, paths)
                    backend = LazyBackend(config, paths, load_models, loaded=loaded)
                    from .person_collection import PersonCollector
                    person_collector = PersonCollector(config, paths)
                    runtime_services = replace(services, backend=backend.borrow)
                if not portraits_only:
                    selected = []
                    ordered = sorted(identifiers, key=int)
                    inputs = ScenePrefetch(client, config, services.source, report=report)
                    inputs.fill(ordered[:config.data['execution']['prefetch_scene_depth']])
                    with ui.phase('Recognition VTT', errors=lambda: counts['scenes_failed']), WorkProgress(report, 'Recognition scenes', len(ordered)) as scenes:
                        for position, identifier in enumerate(ordered):
                            following = ordered[position + 1] if position + 1 < len(ordered) else None
                            try:
                                with activity(report, 'prepare VTT input', scene=identifier):
                                    prepared = inputs.take(identifier)
                                with prepared:
                                    scene, source = prepared.scene, prepared.source
                                    inputs.fill(ordered[position+1:position+1+config.data['execution']['prefetch_scene_depth']])
                                    def progress(**values):
                                        database.progress(run, 'scan', values)
                                    with activity(report, 'detect and embed VTT faces', scene=identifier), frame_progress(report, identifier) as frames:
                                        scan, reused = scan_scene(database, scope, scene, config, model_namespace, source,
                                            backend.borrow, visibility=visibility, progress=progress, frame_progress=frames,
                                            on_cue=person_collector.collect_frame if config.data['sapiensid']['enabled'] else None)
                                    person_collector.flush()
                                    person_collector.enrich(database.connection, scan, source)
                                counts['scenes_reused' if reused else 'scenes_scanned'] += 1
                                from .rejection_summary import summary as rejection_summary
                                rejected=rejection_summary(database.connection,scan,config)
                                counts['observations_rejected']+=sum(rejected['face'].values())
                                counts['face_observations_rejected']+=sum(rejected['face'].values())
                                counts['body_observations_rejected']+=sum(rejected['body'].values())
                                counts['person_evidence_unavailable']+=sum(rejected['person_evidence_unavailable'].values())
                                for kind,reasons in rejected.items():rejection_totals[kind].update(reasons)
                                database.progress(run,'detection',dict(scene=identifier,scan=scan,
                                    rejected=rejected['face'],body_rejected=rejected['body'],
                                    person_evidence_unavailable=rejected['person_evidence_unavailable'],reused=reused))
                                for kind,reasons in rejected.items():
                                    if reasons:
                                        report.optional('recognition_quality',report.detail,
                                            f'Recognition quality scene {identifier} ({kind}): '+
                                            ' '.join(f'{reason}={total}' for reason,total in sorted(reasons.items())))
                                selected.append({'scene': identifier, 'scan': scan})
                            except KeyboardInterrupt:
                                raise
                            except Exception as error:
                                counts['scenes_failed'] += 1
                                report.optional('recognition_error', report.issue, 'recognition', 'scan', identifier, error,
                                                safe_error(error, key))
                            scenes.advance(scanned=counts['scenes_scanned'], reused=counts['scenes_reused'], failed=counts['scenes_failed'])
                            inputs.fill(ordered[position+1:position+1+config.data['execution']['prefetch_scene_depth']])
                            database.progress(run, 'run', dict(counts))
                    inputs.close()
                    if selected:
                        last_scene = selected[-1]['scene']
                        def reference_progress(**values):
                            values['peak_bytes'] = max(references.get('peak_bytes', 0), values['peak_bytes'])
                            references.update(scene=last_scene, recognizer=config.data['models']['recognizer'],
                                              max_samples_per_person=config.data['clustering']['max_samples_per_person'], **values)
                            status(report, references=usage_line(references), reference_people=values['people'],
                                   reference_samples=values['samples'], reference_stage=values['stage'])
                        try:
                            with ui.phase('Recognition identity rebuild'):
                                with work('Reconcile previous merges', 1, unit='step', position=0) as merges:
                                    reconcile_legacy_merges(database, client, scope, [s['scene'] for s in selected])
                                    merges.advance()
                                with tracking(
                                    selected,
                                    'Reconcile scene publications',
                                    unit='scene',
                                    position=0,
                                ) as publications:
                                    for item in publications:
                                        reconcile_unlinks(database, client, scope, item['scene'])
                                rebuilt = rebuild_selected(database, selected, config, progress=reference_progress)
                                from .decision_evidence import summary as decision_summary
                                database.progress(run, 'identity_decisions', decision_summary(
                                    database.connection, matching_policy(config), [s['scan'] for s in selected]))
                            database.progress(run, 'rebuild', {'mode': 'cold_selected_replay', 'selection': selected,
                                                             'matching_policy': matching_policy(config)})
                            with ui.phase('Recognition Stash publication', errors=lambda: counts['people_failed']), WorkProgress(report, 'Publish recognition', len(selected)) as publication:
                                _publish_selection(database, client, report, config, scope, model_namespace,
                                                   selected, rebuilt, ready, counts, key, run, publication)
                        except KeyboardInterrupt:
                            raise
                        except Exception as error:
                            counts['scenes_failed'] += len(selected)
                            report.optional('recognition_error', report.issue, 'recognition', 'rebuild', 'selected scenes', error, safe_error(error, key))
                        finally:
                            if references:
                                database.progress(run, 'references', references)
                                report.optional('recognition_references', report.detail,
                                    f'Face references scene {last_scene}: ' + usage_line(references) +
                                    f" people={references['people']} samples={references['samples']} stage={references['stage']}")
                with ui.phase('Recognition performer selection', errors=lambda: counts['metadata_failed']):
                    def blocked_metadata(person, error):
                        counts['metadata_failed'] += 1
                        counts['metadata_withheld'] += 1
                        report.optional('recognition_metadata_withheld', report.issue, 'recognition',
                                        'performer_selection', person, error, safe_error(error, key))
                    targets = canonical_targets(database, ready, on_blocked=blocked_metadata)
                if portraits_only:
                    counts['portrait_people_selected'] = len(targets)
                    counts['portrait_scenes_selected'] = len(identifiers)
                    database.progress(run, 'portrait_selection', {'target_ids': [t['performer'] for t in targets],
                                                                 'selection': [s for t in targets for s in t['scans']]})
                if services.metadata:
                    plan = {}
                    if not portraits_only:
                        with ui.phase('Recognition names preflight', errors=lambda: counts['metadata_failed']):
                            try:
                                with work('Prepare selected names', 1, unit='step', position=0) as preflight:
                                    plan = services.metadata_plan(database, client, report, config, extra, scope, targets)
                                    preflight.advance()
                                database.progress(run, 'names', {'plan': plan, 'target_ids': [t['performer'] for t in targets]})
                            except KeyboardInterrupt:
                                raise
                            except Exception as error:
                                plan = {}
                                counts['metadata_failed'] += 1
                                report.optional('recognition_names_preflight', report.issue, 'recognition', 'names_preflight',
                                                'selected performers', error, safe_error(error, key))
                    label = 'Portraits' if portraits_only else 'Names and portraits'
                    with NativeBudget(config), portrait_reference_scope(), PortraitPipeline(database,client,config,scope,targets,services.source,visibility,expression,
                        enabled=services.metadata is enrich) as preparation, ui.phase('Recognition portraits' if portraits_only else 'Recognition metadata', errors=lambda: counts['metadata_failed']), WorkProgress(report, label, len(targets), unit='person', key='people') as people:
                        for target_index,target in enumerate(targets):
                            preparation.activate(target_index)
                            with activity(report, label.lower(), person=target['performer']):
                                options = {'visibility': preparation.visibility}
                                if portraits_only:
                                    options['portraits_only'] = True
                                if expression_required(config.data['portraits']):
                                    options['expression'] = preparation.expression
                                if retry_images:
                                    options['retry_unresolved_images'] = True
                                counts.update(services.metadata(database, client, report, config, plan,
                                                               scope, target, paths, runtime_services, **options) or {})
                                preparation.promote()
                                database.progress(run, 'metadata', dict(counts))
                                people.advance(failed=counts['metadata_failed'])
                if targets and not portraits_only:
                    try:
                        with ui.phase('Recognition names starter'):
                            output = starter(client, targets, scope)
                        database.progress(run, 'name_starter', {'yaml': output, 'truncated': len(targets) > 50})
                        report.optional('recognition_name_starter', report.detail, 'Performer names starter (selected IDs, maximum 50):\n' + output)
                    except Exception as error:
                        report.optional('recognition_name_starter', report.write, 'Name starter unavailable: ' + safe_error(error, key))
                if counts['scenes_failed'] or counts['people_failed'] or counts['metadata_failed']:
                    outcome = 'partial'
                ui.finish()
                report.console.finish_rendering()
                report.optional('recognition_summary', report.write, f'Recognition {outcome}: ' +
                                ' '.join(f'{name}={value}' for name, value in sorted(counts.items())) +
                                ''.join(' | '+kind+': '+' '.join(f'{reason}={total}' for reason,total in sorted(reasons.items()))
                                        for kind,reasons in rejection_totals.items() if reasons) +
                                (' | face references=' + usage_line(references) +
                                 f" people={references['people']} samples={references['samples']} stage={references['stage']}" if references else ''))
                return 0 if outcome == 'success' else 1
            except BaseException as error:
                outcome = 'interrupted' if isinstance(error, KeyboardInterrupt) else 'failed'
                failure = safe_error(error, key)[:1000]
                raise
            finally:
                summary = {'errors': {'format': 1, 'unavailable': True}}
                def collect_errors():
                    errors = report.optional('recognition_error_snapshot', error_snapshot, report, key)
                    if errors is not None:
                        summary['errors'] = errors
                def show_errors():
                    report.optional('recognition_error_summary', render_errors, report, summary['errors'])
                def finish_history():
                    summary.update(counts=dict(counts), mutations=dict(report.counts),
                                   rejections={kind:dict(reasons) for kind,reasons in rejection_totals.items()})
                    from .profiling import pipeline_snapshot
                    timings = pipeline_snapshot()
                    if timings is not None:
                        summary['pipeline_profile'] = timings
                    if getattr(report, 'recognition_rebuild_profile', None) is not None:
                        summary['rebuild_profile'] = report.recognition_rebuild_profile
                    if references:
                        summary['references'] = dict(references)
                    database.finish_run(run, outcome, summary, failure)
                actions = [('close face visibility model', visibility.close)] if visibility is not None else []
                if inputs is not None:
                    actions.insert(0, ('close recognition input prefetch', inputs.close))
                if expression is not None:
                    actions.append(('close portrait expression model', expression.close))
                if backend is not None:
                    actions.append(('close recognition identity models', backend.close))
                if person_collector is not None:
                    actions.append(('close person identity models', person_collector.close))
                resource_error = None
                before_cleanup = report.counts['cleanup_failures']
                try:
                    cleanup(actions, report=report)
                except BaseException as error:
                    resource_error = error
                    if outcome in ('success', 'partial'):
                        outcome, failure = 'failed', safe_error(error, key)[:1000]
                finally:
                    if report.counts['cleanup_failures'] > before_cleanup and outcome == 'success':
                        outcome, failure = 'failed', 'Recognition resource cleanup failed.'
                    # History stays on its SQLite owner, even while a worker's
                    # resources must remain quarantined. Preserve any primary.
                    cleanup([('collect recognition error summary', collect_errors),
                             ('render recognition error summary', show_errors),
                             ('finish recognition history', finish_history)],
                            report=report, primary=resource_error)
                if resource_error is not None:
                    raise resource_error
