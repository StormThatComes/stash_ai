"""Optional durable strict-replay generations; active computation stays in RAM.

Only derived grouping rows are saved. Raw evidence and current historical labels
are copied from the authoritative database again before any generation imports.
"""
import hashlib
import json
import os
from pathlib import Path
import re
import sqlite3
import stat
import threading
import uuid

from stash_ai.runtime.cancellation import current as cancellation
from stash_ai.runtime.cleanup import closing, cleanup
from stash_ai.runtime.console import current_report
from .database import FaceDatabase, encoded
from . import rebuild_cache
from .clustering import matching_policy
from .identity_merges import _merge_support_policy
from .profiling import count, measured

APPLICATION_ID = 0x52435054
FORMAT = 1
ENGINE = 'recognition-cold-replay-v10.0'
TABLES = ('people', 'members', 'candidates', 'identity_merges', 'merge_support')
ORDER = ('id', 'observation', 'id', 'source', 'source,target,observation,policy')
SCHEMA = '''
CREATE TABLE metadata(key TEXT PRIMARY KEY, value TEXT NOT NULL);
CREATE TABLE people(id TEXT PRIMARY KEY, scope TEXT NOT NULL, namespace TEXT NOT NULL,
 created REAL NOT NULL, performer TEXT, token TEXT NOT NULL UNIQUE);
CREATE TABLE members(observation TEXT PRIMARY KEY, person TEXT NOT NULL REFERENCES people(id));
CREATE TABLE candidates(id TEXT PRIMARY KEY, scan TEXT NOT NULL, policy TEXT NOT NULL,
 person TEXT, status TEXT NOT NULL, payload TEXT NOT NULL);
CREATE TABLE identity_merges(source TEXT PRIMARY KEY REFERENCES people(id),
 target TEXT NOT NULL REFERENCES people(id), created REAL NOT NULL, evidence TEXT NOT NULL,
 status TEXT NOT NULL, CHECK(source<>target));
CREATE TABLE merge_support(source TEXT NOT NULL REFERENCES people(id),
 target TEXT NOT NULL REFERENCES people(id), observation TEXT NOT NULL, policy TEXT NOT NULL,
 PRIMARY KEY(source,target,observation,policy), CHECK(source<>target));
'''
_NAME = re.compile(r'replay-([0-9a-f]{32})\.sqlite3\Z')
_MAX_META = 4*1024*1024
_BATCH_BYTES = 4*1024*1024


class ReplayClock:
    def __init__(self, value=1):
        self.value = value

    def __call__(self):
        value, self.value = self.value, self.value + 1
        return value


def _check_cancel():
    token = cancellation()
    if token is not None:
        token.check()


def _lease(path):
    value = path.lstat()
    if not stat.S_ISREG(value.st_mode) or value.st_nlink != 1:
        raise ValueError('Checkpoint files must be ordinary owned files.')
    return value.st_dev, value.st_ino, value.st_size, value.st_mtime_ns


def _json(path):
    lease = _lease(path)
    if lease[2] > _MAX_META:
        raise ValueError('Oversized checkpoint metadata.')
    value = json.loads(path.read_bytes())
    if lease != _lease(path) or not isinstance(value, dict):
        raise ValueError('Changed or malformed checkpoint metadata.')
    return value, lease


def _sync_directory(path):
    # Windows cannot open directories for fsync. File fsync + atomic replace
    # still prevents importing a partially written generation there.
    if os.name != 'nt':
        descriptor = os.open(path, os.O_RDONLY | getattr(os, 'O_DIRECTORY', 0))
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)


def _exclusive_json(path, value):
    data = encoded(value).encode()
    if len(data) > _MAX_META:
        raise ValueError('Oversized checkpoint metadata.')
    identity = None
    try:
        with path.open('xb') as output:
            opened = os.fstat(output.fileno())
            identity = opened.st_dev, opened.st_ino
            output.write(data)
            output.flush()
            os.fsync(output.fileno())
    except BaseException:
        def remove_incomplete():
            if identity is not None and path.exists():
                actual = path.lstat()
                if (actual.st_dev,actual.st_ino) == identity:
                    path.unlink()
        cleanup([('remove incomplete checkpoint metadata', remove_incomplete)])
        raise


def _schema(db):
    return [tuple(row) for row in db.execute('SELECT type,name,tbl_name,sql FROM sqlite_master ORDER BY type,name')]


def _review(db):
    if (db.execute('PRAGMA application_id').fetchone()[0] != APPLICATION_ID
            or db.execute('PRAGMA user_version').fetchone()[0] != FORMAT):
        raise ValueError('Unsupported checkpoint format.')
    with closing(sqlite3.connect(':memory:')) as expected:
        expected.executescript(SCHEMA)
        if _schema(db) != _schema(expected):
            raise ValueError('Unsupported checkpoint schema.')
    if db.execute('PRAGMA quick_check').fetchone()[0] != 'ok':
        raise ValueError('Damaged checkpoint database.')


def _header(db):
    rows = db.execute('SELECT key,value FROM metadata').fetchall()
    if len(rows) != 1 or rows[0][0] != 'state' or len(rows[0][1]) > _MAX_META:
        raise ValueError('Malformed checkpoint header.')
    value = json.loads(rows[0][1])
    if not isinstance(value, dict):
        raise ValueError('Malformed checkpoint header.')
    return value


def _rows(db, table, order):
    batch, size = [], 0
    # Never execute SQL obtained from a checkpoint; identifiers are constants.
    for row in db.execute(f'SELECT * FROM {table} ORDER BY {order}'):
        _check_cancel()
        row = tuple(row)
        length = len(encoded(row).encode())
        if length > _BATCH_BYTES:
            raise ValueError('Oversized checkpoint row.')
        if batch and (len(batch) == 256 or size + length > _BATCH_BYTES):
            yield batch
            batch, size = [], 0
        batch.append(row)
        size += length
    if batch:
        yield batch


def _checksum(db, header):
    hasher = hashlib.sha256(encoded(header).encode())
    for table, order in zip(TABLES, ORDER):
        hasher.update(table.encode())
        for batch in _rows(db, table, order):
            for row in batch:
                rebuild_cache._hash_row(hasher, row)
    return hasher.hexdigest()


def _single_signature_cues(db, added):
    """Added context cannot veto old merges without two physical signatures.

    No vector decoding is needed; malformed future vectors still fail at their
    ordinary use. Exact duplicate detections do not add contextual evidence.
    """
    for start in range(0, len(added), 256):
        scans = encoded([item['scan'] for item in added[start:start+256]])
        rows = db.execute('SELECT o.scan,o.cue,o.geometry,o.embedding FROM observations o '
            'WHERE o.scan IN (SELECT value FROM json_each(?)) AND o.rejected IS NULL '
            'ORDER BY o.scan,o.cue,o.id', (scans,))
        cue = signature = None
        for row in rows:
            _check_cancel()
            key = row['scan'], row['cue']
            physical = row['geometry'], row['embedding']
            if key != cue:
                cue, signature = key, physical
            elif signature != physical:
                return False
    return True


class Checkpoints:
    def __init__(self, database, selected, config, scope, namespace, *, phase=None, context=None):
        self.phase, self.phase_context = phase, context
        self.phase_payload = None
        self.restored_payload = None
        self.database, self.selected, self.config = database, selected, config
        self.scope, self.namespace = scope, namespace
        self.interval = config.data['execution'].get('rebuild_checkpoint_scenes', 0)
        self.limit = config.data['execution'].get('rebuild_checkpoint_bytes', 512*1024*1024)
        self.enabled = bool(self.interval)
        self.saved_cursor = 0
        self.input_hash = None
        self.manifest = self.manifest_lease = None
        if self.enabled:
            self._optional(self._initialize)

    def _optional(self, callback, *args):
        try:
            return callback(*args)
        except Exception as error:
            self.enabled = False
            count('checkpoint_failures')
            report = current_report()
            if report is not None:
                # One failure disables this run's checkpoint work. Reporting
                # failure must not turn optional recovery into a rebuild error.
                report.optional('rebuild_checkpoint', lambda: (_ for _ in ()).throw(error))
            return None

    def _initialize(self):
        source = self.database
        if (not isinstance(source, FaceDatabase) or source.lock is None
                or source._owner_thread != threading.get_ident()):
            raise ValueError('Checkpointing requires the current recognition writer owner.')
        self.owner = dict(format=FORMAT, installation=source.installation,
                          path=hashlib.sha256(str(source.path).encode()).hexdigest())
        self.directory = source.path.with_name(source.path.name + ('.rebuild-' + self.phase if self.phase else '.rebuild'))
        try:
            self.directory.mkdir(mode=0o700)
        except FileExistsError:
            if self.directory.is_symlink() or not self.directory.is_dir():
                raise ValueError('Checkpoint directory is not privately owned.')
        self.directory_lease = (self.directory.stat().st_dev, self.directory.stat().st_ino)
        marker = self.directory/'owner.json'
        if not marker.exists() and not marker.is_symlink():
            if any(self.directory.iterdir()):
                raise ValueError('Unowned checkpoint directory is not empty.')
            _exclusive_json(marker, self.owner)
            _sync_directory(self.directory)
        actual, self.owner_lease = _json(marker)
        if actual != self.owner:
            raise ValueError('Checkpoint directory belongs to another database.')
        self.current = self.directory/'current.json'
        if self.current.exists() or self.current.is_symlink():
            self.manifest, self.manifest_lease = _json(self.current)
            if (set(self.manifest) != {'owner','generation','checksum'} or self.manifest['owner'] != self.owner
                    or not _NAME.fullmatch(self.manifest['generation'])
                    or not isinstance(self.manifest['checksum'], str)):
                raise ValueError('Foreign checkpoint manifest.')
        self._prune()

    def _owned(self):
        if (self.directory.is_symlink() or
                (self.directory.stat().st_dev, self.directory.stat().st_ino) != self.directory_lease
                or _json(self.directory/'owner.json') != (self.owner, self.owner_lease)):
            raise ValueError('Checkpoint ownership changed during replay.')
        if self.manifest is None:
            if self.current.exists() or self.current.is_symlink():
                raise ValueError('Checkpoint manifest appeared during replay.')
        elif _json(self.current) != (self.manifest, self.manifest_lease):
            raise ValueError('Checkpoint manifest changed during replay.')

    def _open(self, path):
        _lease(path)
        if path.stat().st_size > self.limit:
            raise ValueError('Checkpoint exceeds the configured file budget.')
        return sqlite3.connect(path.as_uri() + '?mode=ro', uri=True)

    def _prune(self):
        active = self.manifest['generation'] if self.manifest else None
        for path in self.directory.iterdir():
            if re.fullmatch(r'\.manifest-[0-9a-f]{32}\.json', path.name):
                try:
                    value, lease = _json(path)
                    if (set(value) == {'owner','generation','checksum'} and value['owner'] == self.owner
                            and _NAME.fullmatch(value['generation']) and lease == _lease(path)):
                        path.unlink()
                except (OSError, ValueError, TypeError):
                    pass
                continue
            match = _NAME.fullmatch(path.name)
            if not match or path.name == active:
                continue
            try:
                lease = _lease(path)
                with closing(self._open(path)) as db:
                    _review(db)
                    header = _header(db)
                if header.get('owner') == self.owner and header.get('generation') == match[1] and lease == _lease(path):
                    path.unlink()
            except (OSError, ValueError, sqlite3.Error):
                pass  # Unknown files, including damaged foreign files, are untouched.

    def restore(self, replay):
        if not self.enabled or self.manifest is None:
            return 0, 1
        try:
            return self._restore(replay)
        except Exception as error:
            count('checkpoint_rejections')
            report = current_report()
            if report is not None:
                report.optional('rebuild_checkpoint_reuse', lambda: (_ for _ in ()).throw(error))
            # Invalid derived state falls back to cold replay and can be
            # replaced by a newly committed generation of the current inputs.
            return 0, 1

    @measured('checkpoint_restore')
    def _restore(self, replay):
        self._owned()
        path = self.directory/self.manifest['generation']
        lease = _lease(path)
        with closing(self._open(path)) as checkpoint:
            _review(checkpoint)
            header = _header(checkpoint)
            cursor, clock = self._compatible(header)
            checksum = _checksum(checkpoint, header)
            if checksum != self.manifest['checksum'] or lease != _lease(path):
                raise ValueError('Checkpoint integrity changed.')
            # Populate only our reviewed in-memory schema. Any validation failure
            # rolls back grouping rows; freshly copied raw evidence stays intact.
            try:
                replay.execute('BEGIN')
                for table, order in zip(TABLES, ORDER):
                    for batch in _rows(checkpoint, table, order):
                        placeholders = ','.join('?' for _ in batch[0])
                        replay.executemany(f'INSERT INTO {table} VALUES ({placeholders})', batch)
                if self.phase:
                    self._validate_phase(replay, header)
                else:
                    self._validate(replay, cursor, clock)
                replay.commit()
            except BaseException:
                cleanup([('roll back checkpoint import', replay.rollback)])
                raise
        self.restored_payload = header.get('phase_payload')
        self.saved_cursor = cursor
        count('checkpoint_resumed_scenes', cursor)
        count('checkpoint_append_reused_scenes', cursor if header['selected'] != self.selected else 0)
        return cursor, clock

    def _compatible(self, header):
        if self.phase and (header.get('phase') != self.phase or header.get('phase_context') != self.phase_context
                           or header.get('selected') != self.selected):
            raise ValueError('Late checkpoint dependencies changed.')
        previous = header.get('selected')
        cursor, clock = header.get('cursor'), header.get('next_clock')
        if (header.get('format') != FORMAT or header.get('engine') != ENGINE or header.get('owner') != self.owner
                or header.get('generation') != _NAME.fullmatch(self.manifest['generation'])[1]
                or header.get('scope') != self.scope or header.get('namespace') != self.namespace
                or header.get('policy') != matching_policy(self.config)
                or type(cursor) is not int or type(clock) is not int or clock < 1
                or not isinstance(previous, list) or not 0 < cursor <= len(previous)
                or len(previous) > len(self.selected) or previous != self.selected[:len(previous)]):
            raise ValueError('Checkpoint dependencies do not match this replay.')
        limits = header.get('limits')
        if (not isinstance(limits, dict) or any(type(limits.get(k)) is not int or v < limits[k]
                for k, v in rebuild_cache.resource_limits(self.config).items())):
            raise ValueError('Checkpoint admission limits were reduced.')
        if header.get('inputs') != rebuild_cache.input_context(
                self.database.connection, previous, self.scope, self.namespace,self.config):
            raise ValueError('Checkpoint raw input catalogue changed.')
        if not _single_signature_cues(self.database.connection, self.selected[len(previous):]):
            raise ValueError('Added scenes may change earlier co-presence decisions.')
        return cursor, clock

    def _validate(self, replay, cursor, clock):
        from .rebuild import _batches
        from .pose_policy import anchor_sql
        anchor = anchor_sql('o', self.config.data['clustering'])
        prefix = encoded([item['scan'] for item in self.selected[:cursor]])
        replay.execute('CREATE TEMP TABLE checkpoint_prefix(scan TEXT PRIMARY KEY)')
        try:
            replay.execute('INSERT INTO checkpoint_prefix SELECT value FROM json_each(?)', (prefix,))
            eligible = replay.execute('SELECT COUNT(*) FROM observations o JOIN checkpoint_prefix p '
                                     'ON p.scan=o.scan WHERE o.rejected IS NULL AND '+anchor).fetchone()[0]
            if clock-1 > eligible:
                raise ValueError('Invalid checkpoint identity clock.')
            installation = str(uuid.uuid5(uuid.NAMESPACE_URL, encoded([self.scope,self.namespace,'cold-replay'])))
            created = set()
            for row in replay.execute('SELECT * FROM people'):
                uuid.UUID(row['id'])
                if (row['scope'] != self.scope or row['namespace'] != self.namespace or row['performer'] is not None
                        or row['created'] != int(row['created']) or not 1 <= row['created'] < clock
                        or row['created'] in created
                        or row['token'] != f"stash-ai-recognition:{installation}:{row['id']}"):
                    raise ValueError('Invalid checkpoint person.')
                created.add(row['created'])
            if replay.execute('SELECT 1 FROM identity_merges m JOIN people s ON s.id=m.source '
                    'JOIN people t ON t.id=m.target WHERE t.created>=s.created LIMIT 1').fetchone():
                raise ValueError('Checkpoint aliases do not point to older identities.')
            if replay.execute('PRAGMA foreign_key_check').fetchone():
                raise ValueError('Checkpoint grouping references are invalid.')
            count_candidates = 0
            policy = matching_policy(self.config)
            for batch in _batches(replay.execute('SELECT * FROM candidates')):
                for row in batch:
                    payload = json.loads(row['payload'])
                    observation = payload.get('observation') if isinstance(payload, dict) else None
                    raw = replay.execute('SELECT o.*,m.person AS member FROM observations o '
                        'LEFT JOIN members m ON m.observation=o.id JOIN checkpoint_prefix p ON p.scan=o.scan '
                        'WHERE o.id=? AND o.rejected IS NULL AND '+anchor, (observation,)).fetchone()
                    if (raw is None or raw['scan'] != row['scan'] or row['policy'] != policy
                            or row['id'] != str(uuid.uuid5(uuid.UUID(observation),policy))
                            or payload.get('policy') != policy or payload.get('representative_limit') !=
                            self.config.data['clustering']['max_samples_per_person']
                            or row['status'] not in ('accepted','withheld')
                            or (row['status'] == 'accepted' and (row['person'] is None or raw['member'] != row['person']))
                            or (row['status'] == 'withheld' and raw['member'] is not None)):
                        raise ValueError('Checkpoint candidate coverage is invalid.')
                    count_candidates += 1
            if count_candidates != eligible:
                raise ValueError('Checkpoint candidate coverage is incomplete.')
            if replay.execute('SELECT 1 FROM members m JOIN observations o ON o.id=m.observation '
                    'LEFT JOIN checkpoint_prefix p ON p.scan=o.scan LEFT JOIN identity_merges a ON a.source=m.person '
                    'WHERE p.scan IS NULL OR o.rejected IS NOT NULL OR a.source IS NOT NULL LIMIT 1').fetchone():
                raise ValueError('Checkpoint contains future or retired members.')
            if replay.execute('SELECT 1 FROM merge_support m JOIN observations o ON o.id=m.observation '
                    'LEFT JOIN checkpoint_prefix p ON p.scan=o.scan WHERE p.scan IS NULL '
                    'OR o.rejected IS NOT NULL OR m.policy<>? LIMIT 1', (_merge_support_policy(self.config),)).fetchone():
                raise ValueError('Checkpoint support is outside the committed prefix.')
        finally:
            cleanup([('drop checkpoint validation prefix', lambda: replay.execute('DROP TABLE checkpoint_prefix'))])

    def _validate_phase(self, replay, header):
        if header['cursor'] != len(self.selected) or not isinstance(header.get('phase_payload'),dict):
            raise ValueError('Incomplete late checkpoint.')
        if replay.execute('PRAGMA foreign_key_check').fetchone():
            raise ValueError('Invalid late checkpoint relationships.')
        policy=matching_policy(self.config)
        for row in replay.execute('SELECT c.*,o.scan AS actual_scan,o.rejected,m.person AS member '
                "FROM candidates c LEFT JOIN observations o ON o.id=json_extract(c.payload,'$.observation') "
                "LEFT JOIN members m ON m.observation=o.id"):
            if (row['scan']!=row['actual_scan'] or row['rejected'] is not None or row['policy']!=policy
                    or row['status'] not in ('accepted','withheld')
                    or row['status']=='accepted' and (row['person'] is None or row['member']!=row['person'])
                    or row['status']=='withheld' and row['member'] is not None):
                raise ValueError('Invalid late checkpoint candidate.')
        if replay.execute('SELECT 1 FROM members m LEFT JOIN observations o ON o.id=m.observation '
                          'WHERE o.id IS NULL OR o.rejected IS NOT NULL LIMIT 1').fetchone():
            raise ValueError('Invalid late checkpoint membership.')
        from .late_checkpoint import decode
        decode(replay,header['phase_payload'])

    def save(self, replay, cursor, next_clock, *, final=False):
        if (not self.enabled or cursor == self.saved_cursor or
                (not final and cursor % self.interval)):
            return
        self._optional(self._save, replay, cursor, next_clock)

    @measured('checkpoint_save')
    def _save(self, replay, cursor, next_clock):
        if replay.in_transaction:
            raise ValueError('Only a committed strict prefix can be checkpointed.')
        self._owned()
        if self.input_hash is None:
            self.input_hash = rebuild_cache.input_context(self.database.connection, self.selected,self.scope,self.namespace,self.config)
        generation = uuid.uuid4().hex
        header = dict(format=FORMAT,engine=ENGINE,owner=self.owner,generation=generation,
            scope=self.scope,namespace=self.namespace,selected=self.selected,policy=matching_policy(self.config),
            inputs=self.input_hash,limits=rebuild_cache.resource_limits(self.config),cursor=cursor,next_clock=next_clock)
        if self.phase:
            header.update(phase=self.phase,phase_context=self.phase_context,phase_payload=self.phase_payload)
        if len(encoded(header).encode()) > _MAX_META:
            raise ValueError('Oversized checkpoint input catalogue.')
        path = self.directory/f'replay-{generation}.sqlite3'
        stage = self.directory/f'.manifest-{generation}.json'
        published = stage_created = False
        with path.open('xb'):
            pass
        try:
            with closing(sqlite3.connect(path)) as target:
                target.execute('PRAGMA journal_mode=OFF')
                target.execute('PRAGMA synchronous=OFF')
                pages = self.limit//target.execute('PRAGMA page_size').fetchone()[0]
                target.execute(f'PRAGMA max_page_count={pages}')
                target.executescript(SCHEMA)
                target.execute(f'PRAGMA application_id={APPLICATION_ID}')
                target.execute(f'PRAGMA user_version={FORMAT}')
                target.execute('INSERT INTO metadata VALUES (?,?)', ('state',encoded(header)))
                target.commit()  # Establish ownership for crash orphan cleanup.
                for table, order in zip(TABLES, ORDER):
                    for batch in _rows(replay, table, order):
                        placeholders = ','.join('?' for _ in batch[0])
                        target.executemany(f'INSERT INTO {table} VALUES ({placeholders})', batch)
                target.commit()
                checksum = _checksum(target, header)
            with path.open('r+b') as finished:
                os.fsync(finished.fileno())
            _sync_directory(self.directory)
            manifest = dict(owner=self.owner,generation=path.name,checksum=checksum)
            _exclusive_json(stage, manifest)
            stage_created = True
            _check_cancel()
            self._owned()
            if self.manifest is None:
                os.link(stage, self.current)  # Exclusive first publication.
                stage.unlink()
            else:
                os.replace(stage, self.current)
            published = True
            self.manifest, self.manifest_lease = manifest, _lease(self.current)
            _sync_directory(self.directory)
            self.saved_cursor = cursor
            count('checkpoint_saved_scenes', cursor)
            count('checkpoint_bytes', path.stat().st_size)
            self._prune()
        finally:
            # A failed staged generation cannot invalidate the previous pointer.
            cleanup([('remove checkpoint manifest stage', lambda: stage.unlink(missing_ok=True) if stage_created else None),
                     ('remove failed checkpoint generation', lambda: path.unlink(missing_ok=True) if not published else None)])
