"""Finite selected-scene orchestration; no tag runtime or automatic hook."""
from collections import Counter
from dataclasses import asdict, dataclass
import os
from urllib.parse import urlsplit

from stash_ai.runtime.cleanup import closing, cleanup
from stash_ai.runtime.console import present, phase, status
from stash_ai.runtime.report import RunReport
from stash_ai.stash_api.transport import safe_error
from .config import digest
from .database import FaceDatabase, encoded
from .models import ASSETS, ensure_assets, load_backend, namespace
from .media import SpriteInput
from .scanning import scan_scene
from .clustering import cluster_scene
from .publication import publish_person, save_operation
from .stash import RecognitionStash
from .names import load_names, plan_names, starter
from .metadata import enrich
from .video import VideoFrames


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


def execute(args, config, *, services=None):
    services = services or Services()
    retry_images = bool(getattr(args, 'retry_unresolved_images', False))
    url = os.getenv('STASH_URL', 'http://localhost:9999')
    key = os.getenv('STASH_API_KEY', '')
    scope = instance_scope(url)
    # Optional extensions read immutable config before any Stash mutations.
    extra = services.metadata_config(config, scope) if services.metadata_config else None
    report = RunReport(False, print)
    counts = Counter()
    with FaceDatabase(config.database) as database:
        client = services.client(url=url, api_key=key, timeout=config.data['execution']['timeout_seconds'])
        with closing(client, report=report, name='close recognition HTTP'), present(report, args.progress_style):
            client.report = report
            if args.all_scenes:
                identifiers = list(dict.fromkeys(client.scene_ids()))
            else:
                identifiers = list(dict.fromkeys(args.scene_id))
            snapshot = config.snapshot()
            if retry_images:
                snapshot['invocation'] = {'retry_unresolved_images': True}
            if extra is not None:
                snapshot['names'] = extra.snapshot()
            run = database.start_run(scope, identifiers, snapshot)
            outcome, failure = 'success', None
            ready = {}
            try:
                with phase(report, 'Recognition assets'):
                    paths = services.assets(config, write=report.write)
                    model_namespace = namespace(config)
                    provenance = {name: asdict(ASSETS[name]) for name in
                                  (config.data['models']['detector'], config.data['models']['recognizer'])}
                    if config.data['models']['recognizer'] == 'auraface':
                        from .aura import CONTRACT
                        provenance['preprocessing'] = CONTRACT
                    with database.connection:
                        database.connection.execute('INSERT OR IGNORE INTO models VALUES (?,?)',
                                                    (model_namespace, encoded(provenance)))
                for identifier in identifiers:
                    status(report, scene=identifier, scenes_completed=counts['scenes_processed'], scenes_total=len(identifiers))
                    try:
                        scene = client.scene(identifier)
                        with phase(report, 'Recognition VTT', scene=identifier):
                            with services.source(client, scene, config) as source:
                                def progress(**values):
                                    database.progress(run, 'scan', values)
                                    status(report, completed=1, **values)
                                def backend():
                                    value = services.backend(config, paths)
                                    try:
                                        database.progress(run, 'backend', {'namespace': model_namespace,
                                            'embedding_device': value.device, 'detector_device': getattr(value, 'detector_device', 'cpu'),
                                            'dimensions': value.dimensions, 'runtime': getattr(value, 'provenance', {})})
                                    except BaseException:
                                        value.close()
                                        raise
                                    return value
                                scan, reused = scan_scene(database, scope, scene, config, model_namespace, source,
                                    backend, progress=progress)
                            counts['scenes_reused' if reused else 'scenes_scanned'] += 1
                        with phase(report, 'Recognition matching', scene=identifier):
                            groups = cluster_scene(database, scan, config)
                            # A different recognizer cannot infer the old binding
                            # from incompatible vectors. Retain its analysis, but
                            # withhold new profiles when this instance already has
                            # published identities in another namespace.
                            incompatible = database.connection.execute('SELECT namespace FROM people WHERE scope=? '
                                'AND namespace<>? AND performer IS NOT NULL LIMIT 1', (scope, model_namespace)).fetchone()
                            if incompatible:
                                for group in groups:
                                    bound = database.connection.execute('SELECT performer FROM people WHERE id=?',
                                                                        (group['person'],)).fetchone()
                                    if group['status'] == 'accepted' and bound is not None and not bound['performer']:
                                        group['status'] = 'withheld'
                                        group['reason'] = 'incompatible_published_namespace'
                                        save_operation(database, scope=scope, scene=identifier, person=group['person'],
                                            kind='namespace', target=None, payload={'namespace': model_namespace},
                                            precondition={'published_namespace': incompatible['namespace']}, status='withheld')
                            database.progress(run, 'matching', {'scene': identifier, 'groups': groups})
                        accepted = sorted({g['person'] for g in groups if g['status'] == 'accepted'})
                        counts['groups_withheld'] += sum(g['status'] == 'withheld' for g in groups)
                        counts['observations_withheld'] += sum(g['observations'] for g in groups if g['status'] == 'withheld')
                        for person in accepted:
                            try:
                                with phase(report, 'Recognition Stash publication', scene=identifier):
                                    performer = publish_person(database, client, report, scope, identifier, person)
                                    counts['people_linked_or_observed'] += 1
                                    target = ready.setdefault(person, {'person': person, 'performer': performer, 'scans': []})
                                    target['scans'].append({'scene': identifier, 'scan': scan})
                            except KeyboardInterrupt:
                                raise
                            except Exception as error:
                                counts['people_failed'] += 1
                                report.optional('recognition_error', report.write,
                                                'Recognition person withheld: ' + safe_error(error, key))
                        counts['scenes_processed'] += 1
                    except KeyboardInterrupt:
                        raise
                    except Exception as error:
                        counts['scenes_failed'] += 1
                        report.optional('recognition_error', report.write,
                                        f'Recognition scene {identifier} failed: ' + safe_error(error, key))
                    database.progress(run, 'run', dict(counts))
                targets = [ready[key] for key in sorted(ready)]
                if services.metadata:
                    with phase(report, 'Recognition names preflight'):
                        try:
                            plan = services.metadata_plan(database, client, report, config, extra, scope, targets)
                            database.progress(run, 'names', {'plan': plan, 'target_ids': [t['performer'] for t in targets]})
                        except KeyboardInterrupt:
                            raise
                        except Exception as error:
                            plan = {}
                            counts['metadata_failed'] += 1
                            report.optional('recognition_names_preflight', report.write,
                                            'Name batch withheld: ' + safe_error(error, key))
                    for target in targets:
                        with phase(report, 'Recognition metadata'):
                            options = {'retry_unresolved_images': True} if retry_images else {}
                            counts.update(services.metadata(database, client, report, config, plan,
                                                           scope, target, paths, services, **options) or {})
                            database.progress(run, 'metadata', dict(counts))
                if targets:
                    try:
                        output = starter(client, targets, scope)
                        database.progress(run, 'name_starter', {'yaml': output, 'truncated': len(targets) > 50})
                        report.optional('recognition_name_starter', report.write, 'Performer names starter (selected IDs, maximum 50):\n' + output)
                    except Exception as error:
                        report.optional('recognition_name_starter', report.write, 'Name starter unavailable: ' + safe_error(error, key))
                if counts['scenes_failed'] or counts['people_failed'] or counts['metadata_failed']:
                    outcome = 'partial'
                report.optional('recognition_summary', report.write, 'Recognition complete: ' +
                                ' '.join(f'{name}={value}' for name, value in sorted(counts.items())))
                return 0 if outcome == 'success' else 1
            except BaseException as error:
                outcome = 'interrupted' if isinstance(error, KeyboardInterrupt) else 'failed'
                failure = safe_error(error, key)[:1000]
                raise
            finally:
                summary = {'counts': dict(counts), 'mutations': dict(report.counts)}
                cleanup([('finish recognition history',
                          lambda: database.finish_run(run, outcome, summary, failure))], report=report)
