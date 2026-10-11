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
from .publication import publish_person
from .stash import RecognitionStash


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
    metadata_config: object = None
    metadata: object = None


def execute(args, config, *, services=None):
    services = services or Services()
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
            run = database.start_run(scope, identifiers, config.snapshot())
            outcome, failure = 'success', None
            try:
                with phase(report, 'Recognition assets'):
                    paths = services.assets(config, write=report.write)
                    model_namespace = namespace(config)
                    provenance = {name: asdict(ASSETS[name]) for name in
                                  (config.data['models']['detector'], config.data['models']['recognizer'])}
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
                                scan, reused = scan_scene(database, scope, scene, config, model_namespace, source,
                                    lambda: services.backend(config, paths), progress=progress)
                            counts['scenes_reused' if reused else 'scenes_scanned'] += 1
                        with phase(report, 'Recognition matching', scene=identifier):
                            groups = cluster_scene(database, scan, config)
                            database.progress(run, 'matching', {'scene': identifier, 'groups': groups})
                        accepted = sorted({g['person'] for g in groups if g['status'] == 'accepted'})
                        counts['groups_withheld'] += sum(g['status'] == 'withheld' for g in groups)
                        counts['observations_withheld'] += sum(g['observations'] for g in groups if g['status'] == 'withheld')
                        for person in accepted:
                            try:
                                with phase(report, 'Recognition Stash publication', scene=identifier):
                                    performer = publish_person(database, client, report, scope, identifier, person)
                                    counts['people_linked_or_observed'] += 1
                                    if services.metadata:
                                        values = services.metadata(database, client, report, config, extra,
                                                                   scope, identifier, scan, person, performer, paths)
                                        counts.update(values or {})
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
                if counts['scenes_failed'] or counts['people_failed']:
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
