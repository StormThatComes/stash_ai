"""Finite recognition and retained-portrait orchestration; no tag runtime or hook."""
from contextlib import ExitStack
from dataclasses import asdict, dataclass, replace
from itertools import chain
import os
from urllib.parse import urlsplit

from stash_ai.runtime.cleanup import closing, cleanup
from stash_ai.runtime.console import present, status, activity, phase, skipped
from stash_ai.runtime.report import RunReport
from .config import digest
from .profiling import pipeline_stage, pipeline_timing, pipeline_snapshot
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


@pipeline_stage('publication')
def _publish_selection(database, client, report, config, scope, namespace, selected, rebuilt, ready, counts, key, run, progress,*,guard=None):
    db = database.connection
    from .publication_probe import Probe
    probe=Probe(database,client,scope,namespace)
    for position,item in enumerate(selected):
        if position%32==0:probe.prepare(selected[position:position+32],rebuilt)
        scene, scan = item['scene'], item['scan']
        counts['publication_attempted'] += 1
        try:
            groups = rebuilt[scan]
            accepted = sorted({g['person'] for g in groups if g['status'] == 'accepted'})
            counts['groups_withheld'] += sum(g['status'] == 'withheld' for g in groups)
            counts['observations_withheld'] += sum(g['observations'] for g in groups if g['status'] == 'withheld')
            from .publication_queue import current,acknowledge
            from .publication_queue import validate,body_policy
            from .publication_snapshot import Snapshot
            snapshot_reader=Snapshot(db,scope,namespace,scene,config)
            captured,desired,signature=snapshot_reader.read()
            if captured['scan']!=scan or desired!=accepted:
                raise ValueError('Rebuilt recognition results disagree with current scene decisions.')
            queued=current(db,scope,namespace,scene)
            revision=queued['revision'] if queued and queued['scan']==scan else None
            entry=dict(captured,people=desired,signature=signature,revision=revision,
                       policy=matching_policy(config),body=body_policy(config))
            def check():
                validate(db,scope,namespace,entry,config=config,snapshot_reader=snapshot_reader)
                if guard is not None:guard(item)
            check()
            from .publication_batch import ensure_people,publish_people
            targets=probe.unchanged(scene,accepted)
            unchanged=targets is not None
            if unchanged:
                counts['scenes_publication_reused'] += 1
            else:
                probe.clear()
                # Resolve all identities before unlinking old model profiles.
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
            if not unchanged:
                counts['scene_links_removed'] += remove_stale_links(database, client, report, scope, scene,
                                                                     set(targets.values()), namespace,guard=check)
                counts['other_model_links_removed'] += remove_other_model_links(
                    database, client, report, scope, scene, set(targets.values()), namespace,guard=check)
            if check:check()
            with db:acknowledge(db,scope,namespace,scene,revision)
            counts['scenes_processed'] += 1
            if unchanged:
                counts['publications_unchanged'] += 1
            database.progress(run, 'matching',
                              {'scene': scene, 'groups': groups, 'policy': 'cold_selected_replay'})
        except KeyboardInterrupt:
            raise
        except Exception as error:
            counts['people_failed'] += 1
            counts['publication_failed'] += 1
            report.optional('recognition_error', report.issue, 'recognition', 'publication', scene, error, safe_error(error, key))
        progress.advance(linked=counts['people_linked_or_observed'], failed=counts['people_failed'])


def execute(args, config, *, services=None):
    from .work_locations import invocation
    with invocation(config):
        return _run(args, config, services)


def _scene_ids(args, client):
    if args.all_scenes:
        return client.scene_ids()
    explicit = dict.fromkeys(args.scene_id or ())
    ranges = getattr(args, 'scene_range', None)
    if not ranges:
        return explicit
    # Enumerate existing IDs once: gaps and wide numeric ranges must not
    # create missing-scene requests or allocate every integer in a range.
    matching = (identifier for identifier in client.scene_ids()
                if any(start <= int(identifier) <= end for start, end in ranges))
    return chain(explicit, matching)


def _run(args, config, services):
    from .profiling import pipeline_scope
    from tqdm import tqdm
    from .run_display import RunDisplay
    report = RunReport(args.progress_style == 'verbose', tqdm.write)
    display = RunDisplay(report)
    try:
        with present(report, args.progress_style, status_position=2), pipeline_scope(config.data['execution']['profile_rebuild'],
                cuda_sample_every=config.data['execution']['profile_cuda_every']):
            _execute(args, config, services=services, display=display)
    except BaseException as error:
        display.finish('interrupted' if isinstance(error, KeyboardInterrupt) else 'failed')
        raise
    outcome = 'partial' if display.retention == 'failed' else display.recognition
    display.finish(outcome)
    return 0 if outcome == 'success' else 1


def _execute(args, config, *, services, display):
    services = services or Services()
    retry_images = bool(getattr(args, 'retry_unresolved_images', False))
    portraits_only = getattr(args, 'command', 'run') == 'portraits'
    performer_ids = list(dict.fromkeys(getattr(args, 'performer_id', None) or []))
    url = os.getenv('STASH_URL', 'http://localhost:9999')
    key = os.getenv('STASH_API_KEY', '')
    scope = instance_scope(url)
    # Optional extensions read immutable config before any Stash mutations.
    extra = services.metadata_config(config, scope) if services.metadata_config and not portraits_only else None
    report, counts = display.report, display.counts
    rejection_totals, references = display.rejections, display.references
    publication_debt = None
    from .sprite_cache import scope as sprite_scope
    from .disk_budget import scope as disk_scope
    from .inference_cache import scope as inference_scope
    from .resident_models import scope as model_scope
    from .retained_memory import scope as retained_scope
    with ExitStack() as resources_stack:
        with pipeline_timing('startup'), phase(report, 'Recognition startup'):
            with activity(report, 'Check recognition disk budget'):
                disk_budget = resources_stack.enter_context(disk_scope(config))
            for scope_factory in (retained_scope, inference_scope, model_scope, sprite_scope):
                resources_stack.enter_context(scope_factory(config))
            resources_stack.enter_context(policy_scope())
            with activity(report, 'Open recognition database'):
                database = resources_stack.enter_context(FaceDatabase(config.database))
            client = services.client(url=url, api_key=key, timeout=config.data['execution']['timeout_seconds'])
            resources_stack.enter_context(closing(client, report=report, name='close recognition HTTP'))
        from .live_resources import ConsoleResources, monitor
        resources = ConsoleResources(config, budget=disk_budget) if args.progress_style != 'off' else None
        with monitor(resources.meter if resources else None):
            if resources is not None:
                report.console.live_fields = resources.fields
            client.report = report
            ui = RunProgress(report, portraits=portraits_only, metadata=bool(services.metadata))
            with pipeline_timing('selection'), ui.phase('Recognition selection'):
                ready = {}
                if portraits_only:
                    ready = retained_targets(database, config, scope, args.scene_id or None,
                                             performer_ids=performer_ids or None)
                    source = args.scene_id or sorted({s['scene'] for t in ready.values() for s in t['scans']}, key=int)
                else:
                    source = _scene_ids(args, client)
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
            display.selected = len(identifiers)
            snapshot = config.snapshot()
            if portraits_only or retry_images:
                snapshot['invocation'] = {**({'command': 'portraits'} if portraits_only else {}),
                                          **({'performer_ids': performer_ids} if performer_ids else {}),
                                          **({'retry_unresolved_images': True} if retry_images else {})}
            if extra is not None:
                snapshot['names'] = extra.snapshot()
            run = database.start_run(scope, identifiers, snapshot)
            display.run_id = run
            outcome, failure = 'success', None
            visibility = None
            expression = None
            backend = None
            person_collector = None
            inputs = None
            from .gpu_memory import GpuMemory
            gpu = GpuMemory(config, report, database, run)
            from .adaptive_gpu import observe_batches
            resources_stack.enter_context(observe_batches(gpu.batch_change))
            display.gpu = gpu
            if resources is not None:
                resources.gpu = gpu
            model_policy = config.data['execution']['gpu_model_policy']
            sequential = model_policy == 'sequential' and config.data['sapiensid']['enabled']
            try:
                with pipeline_timing('asset_setup'), ui.phase('Recognition assets'):
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
                    display.prefetch = inputs
                    if resources:
                        resources.prefetch = inputs
                    inputs.fill(ordered[:config.data['execution']['prefetch_scene_depth']])
                    from .scan_pipeline import shared_workers
                    with shared_workers(config.data['execution']['scan_cpu_workers']), gpu.phase('vtt'), ui.phase('Recognition VTT', errors=lambda: counts['scenes_failed']), WorkProgress(report, 'Recognition scenes', len(ordered), estimate=False) as scenes:
                        from .scene_batches import Batches
                        batch_size=config.data['execution']['sequential_scene_batch_size'] if sequential else 1
                        def release_faces():
                            if backend.model is None and visibility.model is None:return
                            gpu.release('faces',[('unload face models',backend.unload),('unload face visibility',visibility.close)])
                        def release_all():
                            if backend.model is None and visibility.model is None and person_collector.model is None:return
                            gpu.release('scene',[('close person models',person_collector.close),
                                ('unload face models',backend.unload),('unload face visibility',visibility.close)])
                        batches=Batches(ordered,batch_size,release_faces,release_all)
                        with closing(batches.events(),name='close scene batches') as events:
                            for pass_kind,position,identifier in events:
                                if pass_kind != 'combined':
                                    offset = position % batch_size
                                    total = min(batch_size, len(ordered) - position + offset)
                                    status(report, batch=f'{"pose" if pass_kind == "body" else "face"} {offset + 1}/{total}')
                                cursor = initial = scan = cue_total = None
                                operation = 'pose' if pass_kind == 'body' else 'face'
                                scan_mode = None
                                def started(mode, completed):
                                    nonlocal cursor, initial, scan_mode
                                    initial = cursor = completed
                                    scan_mode = mode
                                    display.scans[mode] += 1
                                scene_work = False
                                try:
                                    with activity(report, 'prepare VTT input', scene=identifier):
                                        prepared = batches.take(pass_kind,position,identifier,inputs)
                                    with prepared:
                                        scene, source = prepared.scene, prepared.source
                                        cue_total = len(source.cues)
                                        if pass_kind!='body':inputs.fill(ordered[position+1:position+1+config.data['execution']['prefetch_scene_depth']])
                                        def progress(**values):
                                            nonlocal cursor
                                            cursor = values.get('scanned', values.get('reused', cursor))
                                            status(report, checkpoint=cursor)
                                            database.progress(run, 'scan', values)
                                        if pass_kind=='body':scan,reused=batches.results[position]
                                        else:
                                            with activity(report, 'detect and embed VTT faces', scene=identifier), frame_progress(report, identifier, started) as frames:
                                                scan, reused = scan_scene(database, scope, scene, config, model_namespace, source,
                                                    backend.borrow, visibility=visibility, progress=progress, frame_progress=frames,
                                                    on_cue=person_collector.collect_frame if config.data['sapiensid']['enabled'] and not sequential else None)
                                            if scan_mode is None:
                                                display.scans['reused' if reused else 'fresh'] += 1
                                            new_cues = 0 if reused else cue_total - (initial or 0)
                                            counts['face_cues'] += new_cues
                                            if config.data['sapiensid']['enabled'] and not sequential:
                                                counts['pose_cues'] += new_cues
                                        scene_work = not reused
                                        if pass_kind=='face':
                                            batches.results[position]=(scan,reused)
                                            continue
                                        if sequential and batch_size==1 and (backend.model is not None or visibility.model is not None):
                                            gpu.release('faces', [('unload face models', backend.unload),
                                                ('unload face visibility', visibility.close)], scene=identifier)
                                        operation = 'pose'
                                        person_collector.flush()
                                        filled = person_collector.enrich(database.connection, scan, source) or 0
                                        counts['pose_backfilled'] += filled
                                        counts['pose_cues'] += filled
                                        counts['vtt_complete'] += 1
                                    from .solver_scan_sizes import retain
                                    retain(database.connection, scan)
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
                                    if operation == 'pose':
                                        missing = None
                                        if scan is not None and cue_total is not None:
                                            missing = report.optional('pose_resume', person_collector.missing_cues,
                                                                      database.connection, scan, cue_total)
                                        display.pose_resume = identifier, missing
                                    elif cursor is not None:
                                        display.interrupted_cursor = identifier, cursor
                                    scene_work = True
                                    raise
                                except Exception as error:
                                    scene_work = True
                                    counts['scenes_failed'] += 1
                                    report.optional('recognition_error', report.issue, 'recognition', 'scan', identifier, error,
                                                    safe_error(error, key))
                                finally:
                                    if scan is None and scan_mode in ('fresh', 'resumed'):
                                        committed = max(0, cursor - initial)
                                        counts['face_cues'] += committed
                                        if config.data['sapiensid']['enabled'] and not sequential:
                                            counts['pose_cues'] += committed
                                    if sequential and batch_size==1 and (scene_work or backend.model is not None or
                                                       visibility.model is not None or person_collector.model is not None):
                                        cleanup([('release scene GPU models', lambda: gpu.release('scene', [
                                            ('close person models', person_collector.close),
                                            ('unload face models', backend.unload),
                                            ('unload face visibility', visibility.close)], scene=identifier))], report=report)
                                scenes.advance(fresh=display.scans['fresh'], resumed=display.scans['resumed'],
                                               reused=display.scans['reused'], failed=counts['scenes_failed'],
                                               face_cues=counts['face_cues'], pose_cues=counts['pose_cues'])
                                if pass_kind!='body':inputs.fill(ordered[position+1:position+1+config.data['execution']['prefetch_scene_depth']])
                                database.progress(run, 'run', dict(counts))
                    inputs.close()
                    person_collector.flush()
                    releases = [('close person models', person_collector.close)]
                    if model_policy != 'retain':
                        releases += [('unload face models', backend.unload), ('unload face visibility', visibility.close)]
                    gpu.release('vtt', releases)
                    if selected:
                        def reference_progress(**values):
                            values['peak_bytes'] = max(references.get('peak_bytes', 0), values['peak_bytes'])
                            references.update(scene=values.pop('scene',None), recognizer=config.data['models']['recognizer'],
                                              max_samples_per_person=config.data['clustering']['max_samples_per_person'], **values)
                            status(report, references=usage_line(references), reference_people=values['people'],
                                   reference_samples=values['samples'], reference_stage=values['stage'])
                        counts['publication_selected'] = len(selected)
                        reconciling = True
                        reconcile_scene = 'selected scenes'
                        try:
                            with phase(report, 'Recognition Stash reconciliation'):
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
                                        reconcile_scene = item['scene']
                                        reconcile_unlinks(database, client, scope, reconcile_scene)
                            reconciling = False
                            with gpu.phase('rebuild'), ui.phase('Recognition identity rebuild'):
                                rebuilt = rebuild_selected(database, selected, config, progress=reference_progress)
                                from .decision_evidence import summary as decision_summary
                                database.progress(run, 'identity_decisions', decision_summary(
                                    database.connection, matching_policy(config), [s['scan'] for s in selected]))
                            face_mode=getattr(database,'face_rebuild_mode','cold')
                            body_mode=getattr(database,'body_rebuild_mode','disabled')
                            hybrid_mode=getattr(database,'hybrid_rebuild_mode','disabled')
                            display.rebuild = dict(face=face_mode, body=body_mode, hybrid=hybrid_mode,
                                                   scenes=len(getattr(database, 'last_rebuild_selection', selected)))
                            phases={face_mode,body_mode,hybrid_mode}-{'disabled'}
                            mode=next(iter(phases)) if len(phases)==1 else 'partial'
                            database.progress(run, 'rebuild', {'mode': mode, 'face_mode':face_mode,'body_mode':body_mode,
                                                             'hybrid_mode':hybrid_mode,'selection':selected,
                                                             'rebuild_selection':getattr(database,'last_rebuild_selection',selected),
                                                             'matching_policy': matching_policy(config)})
                        except KeyboardInterrupt:
                            raise
                        except Exception as error:
                            rebuilt = None
                            if reconciling:
                                counts['publication_aborted'] += 1
                                report.optional('recognition_error', report.issue, 'recognition', 'publication',
                                                reconcile_scene, error, safe_error(error, key))
                            else:
                                counts['scenes_failed'] += len(selected)
                                report.optional('recognition_error', report.issue, 'recognition', 'rebuild',
                                                'selected scenes', error, safe_error(error, key))
                        finally:
                            if references:
                                database.progress(run, 'references', references)
                                report.optional('recognition_references', report.detail,
                                    ('Face references scene '+references['scene'] if references['scene'] else 'Face references global') + ': ' + usage_line(references) +
                                    f" people={references['people']} samples={references['samples']} stage={references['stage']}")
                        if rebuilt is not None:
                            try:
                                with ui.phase('Recognition Stash publication', errors=lambda: counts['people_failed']), WorkProgress(report, 'Publish recognition', len(selected)) as publication:
                                    _publish_selection(database, client, report, config, scope, model_namespace,
                                                       selected, rebuilt, ready, counts, key, run, publication)
                            except Exception as error:
                                counts['publication_aborted'] += 1
                                report.optional('recognition_error', report.issue, 'recognition', 'publication',
                                                'selected scenes', error, safe_error(error, key))
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
                    with gpu.phase('portraits'), NativeBudget(config), portrait_reference_scope(), PortraitPipeline(database,client,config,scope,targets,services.source,visibility,expression,
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
                if counts['scenes_failed'] or counts['people_failed'] or counts['metadata_failed'] or counts['publication_aborted']:
                    outcome = 'partial'
                ui.finish()
                # present() closes rendering after finally has also finished
                # resource cleanup, history and optional automatic retention.
                from .publication_queue import debt
                publication_debt = report.optional('recognition_publication_snapshot',debt,
                    database.connection,scope,model_namespace,identifiers)
                if publication_debt is not None:
                    publication_debt['selected_scene_count'] = len(identifiers)
                    publication_debt['rebuild_scene_count'] = len(getattr(database,'last_rebuild_selection',()))
                if publication_debt and (publication_debt['pending_selected'] or publication_debt['pending_outside_selection']):
                    report.optional('recognition_publication_debt',report.write,
                        'Recognition publication pending: selected='+str(publication_debt['pending_selected'])+
                        ' outside_selection='+str(publication_debt['pending_outside_selection'])+
                        '; inspect with recognition sync-identities')
                display.debt = publication_debt
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
                    if counts['publication_selected']:
                        counts['publication_not_attempted'] = counts['publication_selected'] - counts['publication_attempted']
                    summary.update(counts=dict(counts), mutations=dict(report.counts),
                                   rejections={kind:dict(reasons) for kind,reasons in rejection_totals.items()})
                    if publication_debt is not None:summary['publication_debt'] = publication_debt
                    from .profiling import pipeline_snapshot
                    timings = pipeline_snapshot()
                    if timings is not None:
                        timings['cutoff'] = 'before_auto_prune'
                        summary['pipeline_profile'] = timings
                    if getattr(report, 'recognition_rebuild_profile', None) is not None:
                        summary['rebuild_profile'] = report.recognition_rebuild_profile
                    if references:
                        summary['references'] = dict(references)
                    if inputs and inputs.taken:
                        summary['vtt_prefetch'] = dict(taken=inputs.taken, ready=inputs.ready, blocked_seconds=inputs.blocked_seconds)
                    if gpu.batch_sizes:
                        summary['cuda_batches'] = gpu.batch_sizes
                    if gpu.enabled:
                        summary['gpu_memory'] = gpu.summary()
                    database.finish_run(run, outcome, summary, failure)
                actions = [('close face visibility model', visibility.close)] if visibility is not None else []
                stop = [('close recognition input prefetch', inputs.close)] if inputs is not None else []
                if expression is not None:
                    actions.append(('close portrait expression model', expression.close))
                if backend is not None:
                    actions.append(('close recognition identity models', backend.close))
                if person_collector is not None:
                    actions.append(('close person identity models', person_collector.close))
                resource_error = None
                before_cleanup = report.counts['cleanup_failures']
                try:
                    with phase(report, 'Recognition resource cleanup'):
                        gpu.release('final', actions, stop=stop)
                except BaseException as error:
                    resource_error = error
                    if outcome in ('success', 'partial'):
                        outcome, failure = 'failed', safe_error(error, key)[:1000]
                finally:
                    if report.counts['cleanup_failures'] > before_cleanup and outcome == 'success':
                        outcome, failure = 'failed', 'Recognition resource cleanup failed.'
                    display.recognition = outcome
                    # History stays on its SQLite owner, even while a worker's
                    # resources must remain quarantined. Preserve any primary.
                    with phase(report, 'Recognition final history'):
                        cleanup([('collect recognition error summary', collect_errors),
                             ('render recognition error summary', show_errors),
                             ('finish recognition history', finish_history)],
                            report=report, primary=resource_error)
                try:
                    if resource_error is not None:
                        raise resource_error
                    if outcome == 'success' and not portraits_only:
                        from .db_retention import after_run, format_report
                        try:
                            retained = after_run(database, config, run, api_key=key)
                        except KeyboardInterrupt:
                            display.retention = 'interrupted'
                            raise
                        except Exception as error:
                            display.retention = 'failed'
                            report.optional('recognition_retention_error', report.issue, 'recognition',
                                            'retention', None, error, safe_error(error, key))
                            retained = None
                        if retained is None and display.retention == 'skipped':
                            skipped(report, 'Recognition auto prune', 'disabled or no eligible rebuild')
                        if retained is not None:
                            display.retention = 'success'
                            report.optional('recognition_retention_report', report.write, format_report(retained))
                finally:
                    # Keep the already durable recognition outcome. Retention may
                    # fail independently; only refresh telemetry after its attempt.
                    report.optional('recognition_profile_finalize', _finish_pipeline_profile,
                                    database, run, display.retention)


def _finish_pipeline_profile(database, run, retention):
    timings = pipeline_snapshot()
    if timings is None:
        return
    timings.update(cutoff='after_auto_prune_before_profile_write', retention_outcome=retention)
    with database.connection:
        database.connection.execute("UPDATE runs SET summary=json_set(summary,'$.pipeline_profile',json(?)) WHERE id=?",
                                    (encoded(timings), run))
