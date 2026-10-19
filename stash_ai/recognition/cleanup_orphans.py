"""Explicit, fail-closed deletion of Stash recognition profiles absent from face SQLite."""
from collections import Counter
import os
import re
from pathlib import Path
from urllib.parse import urlsplit

from stash_ai.runtime.cleanup import closing
from stash_ai.runtime.console import present, phase
from stash_ai.runtime.report import RunReport
from .database import FaceDatabase
from .diagnostics import message as safe_error
from .progress import tracking
from .publication import PendingRecognition, operation_id, perform
from .run import instance_scope
from .stash import RecognitionStash, checked_id, owns

# Only exact lines issued by recognition count as app ownership. A note merely
# containing the substring is insufficient grounds to delete a performer.
_UUID = r'[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}'
TOKEN = re.compile(r'^stash-ai-recognition:' + _UUID + ':' + _UUID + r'$')


def tokens(performer):
    details = performer.get('details') or ''
    return frozenset(line for line in details.splitlines() if TOKEN.fullmatch(line))


def _delete_orphan(database, client, report, scope, row, expected):
    """Journal deletion intent first; observe uncertain outcomes without retrying."""
    performer = checked_id(row)['id']
    person = 'orphan-performer:' + performer  # deterministic, never an SQLite identity
    identifier = operation_id(scope, '', person, 'delete')
    previous = database.connection.execute(
        'SELECT status,target FROM publications WHERE id=?', (identifier,)).fetchone()
    fresh = client.lookup_performer(performer)
    if fresh is None:
        if previous and previous['status'] in ('pending', 'unknown'):
            with database.connection:
                database.connection.execute(
                    "UPDATE publications SET status='observed',error=NULL WHERE id=?", (identifier,))
        return 'absent'
    checked_id(fresh, performer)
    if tokens(fresh) != expected or not all(owns(fresh, token) for token in expected):
        raise PendingRecognition('Orphan performer ownership changed; deletion withheld.')
    if previous and previous['status'] in ('pending', 'unknown'):
        raise PendingRecognition('Earlier orphan deletion outcome is unresolved; no blind retry.')

    def delete():
        current = client.lookup_performer(performer)
        if current is None or tokens(current) != expected:
            raise PendingRecognition('Orphan ownership changed immediately before deletion.')
        if client.delete_person(performer) is not True:
            raise PendingRecognition('Stash performer deletion was not acknowledged.')
        if client.lookup_performer(performer) is not None:
            raise PendingRecognition('Stash performer deletion was not confirmed by a fresh read.')
    perform(database, client, report, scope=scope, scene='', person=person,
            kind='delete', target=performer,
            payload={'performer': performer, 'reason': 'absent_from_face_database'},
            precondition={'ownership_tokens': sorted(expected)}, callback=delete)
    return 'deleted'


def execute(args, config, *, services=None):
    """Independent read-only inventory unless --apply is explicitly provided."""
    path = Path(config.database)
    if not path.is_file():
        raise FileNotFoundError(
            'Recognition cleanup requires an existing face database; refusing an empty/new database.')
    from tqdm import tqdm
    apply = bool(getattr(args, 'apply', False))
    url, key = os.getenv('STASH_URL', 'http://localhost:9999'), os.getenv('STASH_API_KEY', '')
    scope = instance_scope(url)
    client_factory = getattr(services, 'client', RecognitionStash)
    report = RunReport(args.progress_style == 'verbose', tqdm.write)
    counts = Counter()
    with FaceDatabase(path) as database:
        client = client_factory(url=url, api_key=key, timeout=config.data['execution']['timeout_seconds'])
        with closing(client, report=report, name='close recognition cleanup HTTP'), present(report, args.progress_style):
            client.report = report
            run = database.start_run(scope, [], {
                **config.snapshot(), 'invocation': {'command': 'cleanup', 'apply': apply}})
            outcome, failure = 'success', None
            try:
                local = database.connection.execute(
                    'SELECT token,performer FROM people WHERE scope=?', (scope,)).fetchall()
                known_tokens = {row['token'] for row in local}
                bound_ids = {row['performer'] for row in local if row['performer']}
                with phase(report, 'Recognition orphan inventory'):
                    with tracking(client.recognition_performers(), 'Inspect Stash recognition performers',
                                  unit='performer') as tracked:
                        for row in tracked:
                            counts['scanned'] += 1
                            performer = checked_id(row)['id']
                            original = tokens(row)
                            if not original:
                                counts['ignored'] += 1
                                continue
                            if performer in bound_ids or original & known_tokens:
                                counts['protected'] += 1
                                continue
                            counts['orphans'] += 1
                            if not apply:
                                report.optional('recognition_orphan', report.detail,
                                                'Orphan recognition performer candidate: ' + performer)
                                continue
                            try:
                                result = _delete_orphan(database, client, report, scope, row, original)
                                counts['deleted' if result == 'deleted' else 'already_absent'] += 1
                            except KeyboardInterrupt:
                                raise
                            except Exception as error:
                                counts['failed'] += 1
                                report.optional('recognition_error', report.issue, 'recognition', 'cleanup',
                                                performer, error, safe_error(error, key))
                if counts['failed']:
                    outcome = 'partial'
                database.progress(run, 'cleanup', dict(counts))
                report.console.finish_rendering()
                report.optional('recognition_cleanup_summary', report.write,
                    'Recognition cleanup ' + ('apply' if apply else 'dry-run') + ': ' +
                    ' '.join(f'{name}={value}' for name, value in sorted(counts.items())))
                return 0 if outcome == 'success' else 1
            except BaseException as error:
                outcome = 'interrupted' if isinstance(error, KeyboardInterrupt) else 'failed'
                failure = safe_error(error, key)[:1000]
                raise
            finally:
                database.finish_run(run, outcome, {'counts': dict(counts)}, failure)
