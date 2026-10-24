"""One independently owned SQLite for all face evidence, history and recovery."""
from contextlib import AbstractContextManager
from stash_ai.runtime.cleanup import closing, cleanup
import json
import os
import re
import tempfile
import threading
from pathlib import Path
import sqlite3
import time
import uuid

APPLICATION_ID = 0x46524347  # FRCG; never accept a tagger or arbitrary database.
SCHEMA_VERSION = 5
SCHEMA = '''
CREATE TABLE metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL);
CREATE TABLE runs (id TEXT PRIMARY KEY, scope TEXT NOT NULL, selection TEXT NOT NULL,
 config TEXT NOT NULL, started REAL NOT NULL, finished REAL, status TEXT NOT NULL,
 summary TEXT NOT NULL DEFAULT '{}', error TEXT);
CREATE TABLE models (namespace TEXT PRIMARY KEY, provenance TEXT NOT NULL);
CREATE TABLE scans (id TEXT PRIMARY KEY, scope TEXT NOT NULL, scene TEXT NOT NULL,
 namespace TEXT NOT NULL, input_sha TEXT NOT NULL, policy_sha TEXT NOT NULL,
 manifest TEXT NOT NULL, coverage TEXT NOT NULL, total INTEGER NOT NULL,
 cursor INTEGER NOT NULL DEFAULT 0, status TEXT NOT NULL DEFAULT 'partial',
 UNIQUE(scope,scene,namespace,input_sha,policy_sha));
CREATE TABLE observations (id TEXT PRIMARY KEY, scan TEXT NOT NULL REFERENCES scans(id),
 cue INTEGER NOT NULL, seconds REAL NOT NULL, source TEXT NOT NULL, rect TEXT NOT NULL,
 geometry TEXT NOT NULL, quality TEXT NOT NULL, rejected TEXT,
 dimensions INTEGER NOT NULL, embedding BLOB, crop BLOB);
CREATE INDEX observations_scan ON observations(scan,cue,id);
CREATE TABLE people (id TEXT PRIMARY KEY, scope TEXT NOT NULL, namespace TEXT NOT NULL,
 created REAL NOT NULL, performer TEXT, token TEXT NOT NULL UNIQUE);
CREATE TABLE members (observation TEXT PRIMARY KEY REFERENCES observations(id),
 person TEXT NOT NULL REFERENCES people(id));
CREATE INDEX members_person ON members(person);
CREATE INDEX people_scope ON people(scope,namespace,id);
CREATE TABLE candidates (id TEXT PRIMARY KEY, scan TEXT NOT NULL REFERENCES scans(id),
 policy TEXT NOT NULL, person TEXT, status TEXT NOT NULL, payload TEXT NOT NULL);
CREATE INDEX candidates_scan_policy ON candidates(scan,policy,person,status);
CREATE INDEX candidates_person ON candidates(person);
CREATE TABLE publications (id TEXT PRIMARY KEY, scope TEXT NOT NULL, scene TEXT NOT NULL,
 person TEXT NOT NULL, kind TEXT NOT NULL, target TEXT, payload TEXT NOT NULL,
 precondition TEXT NOT NULL, status TEXT NOT NULL, error TEXT);
CREATE INDEX publications_scope ON publications(scope,scene,status);
CREATE TABLE portraits (id TEXT PRIMARY KEY, scope TEXT NOT NULL, person TEXT NOT NULL,
 scene TEXT NOT NULL, metadata TEXT NOT NULL, image BLOB, digest TEXT, status TEXT NOT NULL);
CREATE TABLE progress (run TEXT NOT NULL REFERENCES runs(id), phase TEXT NOT NULL,
 data TEXT NOT NULL, PRIMARY KEY(run,phase));
'''
TABLES = ('metadata', 'runs', 'models', 'scans', 'observations', 'people', 'members',
          'candidates', 'publications', 'portraits', 'progress')
SCHEMA_V1, TABLES_V1 = SCHEMA, TABLES
MERGE_SCHEMA = '''
CREATE TABLE identity_merges (source TEXT PRIMARY KEY REFERENCES people(id),
 target TEXT NOT NULL REFERENCES people(id), created REAL NOT NULL,
 evidence TEXT NOT NULL, status TEXT NOT NULL, CHECK(source<>target),
 CHECK(status IN ('pending','confirmed','observed','local')));
CREATE INDEX identity_merges_target ON identity_merges(target);
'''
SCHEMA += MERGE_SCHEMA
TABLES += ('identity_merges',)
SCHEMA_V2, TABLES_V2 = SCHEMA, TABLES
SUPPORT_SCHEMA = '''
CREATE TABLE merge_support (
 source TEXT NOT NULL REFERENCES people(id),
 target TEXT NOT NULL REFERENCES people(id),
 observation TEXT NOT NULL REFERENCES observations(id),
 policy TEXT NOT NULL,
 PRIMARY KEY(source,target,observation,policy),
 CHECK(source<>target));
CREATE INDEX merge_support_policy_pair ON merge_support(policy,source,target);
'''
SCHEMA += SUPPORT_SCHEMA
TABLES += ('merge_support',)
SCHEMA_V3, TABLES_V3 = SCHEMA, TABLES
REBUILD_SCHEMA = '''
CREATE TABLE identity_origins (
 sample_key TEXT PRIMARY KEY,
 person TEXT NOT NULL REFERENCES people(id),
 observation TEXT NOT NULL REFERENCES observations(id));
CREATE INDEX identity_origins_person ON identity_origins(person);
'''
SCHEMA += REBUILD_SCHEMA
TABLES += ('identity_origins',)

SCHEMA_V4, TABLES_V4 = SCHEMA, TABLES
from .dependency_schema import SCHEMA as DEPENDENCY_SCHEMA
SCHEMA += DEPENDENCY_SCHEMA
TABLES += ('evidence_versions','evidence_fingerprints')


def encoded(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False)


def _reject_executable_schema(connection):
    from .dependency_schema import validate_triggers
    validate_triggers(connection)


def _checks(connection, table):
    """Compare CHECK expressions as SQL tokens, including nested parentheses.

    Never execute schema SQL read from the database. Comments, spacing and
    identifier quoting may vary; string literals retain their exact meaning.
    """
    sql = connection.execute("SELECT sql FROM sqlite_master WHERE type='table' AND name=?", (table,)).fetchone()[0]
    tokens = []
    for token in re.findall(r"--[^\n]*|/\*[\s\S]*?\*/|'(?:''|[^'])*'|\"(?:\"\"|[^\"])*\"|"
                            r"`(?:``|[^`])*`|\[[^\]]*\]|[A-Za-z_][A-Za-z_0-9]*|<>|!=|<=|>=|==|[^\s]", sql):
        if token.startswith(('--', '/*')):
            continue
        if token.startswith(('"', '`', '[')):
            quote = token[0]
            token = token[1:-1].replace(quote * 2, quote)
        tokens.append(token if token.startswith("'") else token.casefold())
    checks = []
    for index, token in enumerate(tokens[:-1]):
        if token != 'check' or tokens[index + 1] != '(':
            continue
        depth = 0
        for end in range(index + 1, len(tokens)):
            depth += (tokens[end] == '(') - (tokens[end] == ')')
            if depth == 0:
                checks.append(tuple(tokens[index + 2:end]))
                break
    return sorted(checks)


class FaceDatabase(AbstractContextManager):
    def __init__(self, path):
        self.path = Path(path).expanduser().resolve()
        self.connection = self.lock = None
        self._sqlite_connection = None
        self._owner_thread = threading.get_ident()
        try:
            if self.path.exists():
                if self.path.stat().st_nlink > 1:
                    raise ValueError('Recognition SQLite must not be a hard-linked file.')
                with closing(sqlite3.connect(self.path.as_uri() + '?mode=ro', uri=True)) as check:
                    _reject_executable_schema(check)
                    application = check.execute('PRAGMA application_id').fetchone()[0]
                    version = check.execute('PRAGMA user_version').fetchone()[0]
                    tables = check.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
                    if (tables and application != APPLICATION_ID or
                            not tables and (application != 0 or version != 0)):
                        raise ValueError('Not a facial recognition database; tag/history/foreign SQLite is refused.')
            self.path.parent.mkdir(parents=True, exist_ok=True)
            from stash_ai.storage.writer_lock import acquire_writer_lock
            try:
                self.lock = acquire_writer_lock(self.path)
            except RuntimeError as error:
                raise RuntimeError('Another recognition writer owns this SQLite database.') from error
            self.connection = sqlite3.connect(self.path, timeout=30)
            self._sqlite_connection = self.connection
            self.connection.row_factory = sqlite3.Row
            self.connection.execute('PRAGMA foreign_keys=ON')
            self.connection.execute('PRAGMA cache_size=-32768')
            self._initialize()
            # Validate ownership identity before orphan recovery changes history.
            self.installation
            from .reference_roles import index
            index(self.connection)
            # Exclusive file ownership proves that every prior writer ended.
            # Runs created later by this owner are never classified as orphans.
            with self.connection:
                self.connection.execute("UPDATE runs SET status='interrupted',finished=COALESCE(finished,?) "
                                        "WHERE status='running'", (time.time(),))
        except BaseException:
            cleanup([('close failed recognition database', self.close)])
            raise

    def _initialize(self):
        db = self.connection
        _reject_executable_schema(db)
        version = db.execute('PRAGMA user_version').fetchone()[0]
        tables = {r[0] for r in db.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        if not tables:
            try:
                db.executescript('BEGIN IMMEDIATE;\n' + SCHEMA)
                db.execute('INSERT INTO metadata VALUES (?,?)', ('installation', str(uuid.uuid4())))
                db.execute(f'PRAGMA application_id={APPLICATION_ID}')
                db.execute(f'PRAGMA user_version={SCHEMA_VERSION}')
                db.commit()
            except BaseException:
                cleanup([('roll back recognition schema', db.rollback)])
                raise
        elif version in (1, 2, 3, 4) and db.execute('PRAGMA application_id').fetchone()[0] == APPLICATION_ID:
            previous_schema = {1: (SCHEMA_V1, TABLES_V1), 2: (SCHEMA_V2, TABLES_V2),
                               3: (SCHEMA_V3, TABLES_V3),4: (SCHEMA_V4,TABLES_V4)}[version]
            self._validate(*previous_schema, tables)
            self.installation
            try:
                db.executescript('BEGIN IMMEDIATE;\n' + (MERGE_SCHEMA if version == 1 else '')
                                 + (SUPPORT_SCHEMA if version <= 2 else '') + (REBUILD_SCHEMA if version<=3 else '')
                                 + DEPENDENCY_SCHEMA)
                db.execute(f'PRAGMA user_version={SCHEMA_VERSION}')
                db.commit()
            except BaseException:
                cleanup([('roll back recognition schema', db.rollback)])
                raise
        elif db.execute('PRAGMA application_id').fetchone()[0] != APPLICATION_ID or version != SCHEMA_VERSION:
            raise ValueError('Unsupported or incomplete facial recognition schema; restore a consistent backup.')
        tables = {r[0] for r in db.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        self._validate(SCHEMA, TABLES, tables)
        from .dependency_schema import validate_triggers
        validate_triggers(db,required=True)
        self.installation
        # Additive access paths: old valid state needs no evidence migration.
        with db:
            db.execute('CREATE INDEX IF NOT EXISTS candidates_scan_policy ON candidates(scan,policy,person,status)')
            db.execute('CREATE INDEX IF NOT EXISTS candidates_person ON candidates(person)')
            # Scoped model-link isolation resolves performer IDs for each selected
            # scene; avoid rescanning the entire identity catalog every time.
            db.execute('CREATE INDEX IF NOT EXISTS people_scope_performer ON people(scope,performer)')
            db.execute("CREATE INDEX IF NOT EXISTS portraits_observation ON portraits(scope,person,"
                       "json_extract(metadata,'$.observation')) WHERE json_valid(metadata)")

    def _validate(self, schema, tables, actual_tables):
        db = self.connection
        if actual_tables != set(tables):
            raise ValueError('Unsupported or incomplete facial recognition schema; restore a consistent backup.')
        # Validate full column shape, not only the existence of tables.
        with closing(sqlite3.connect(':memory:')) as reference:
            reference.executescript(schema)
            for table in tables:
                expected = [tuple(r[1:]) for r in reference.execute(f'PRAGMA table_info({table})')]
                actual = [tuple(r[1:]) for r in db.execute(f'PRAGMA table_info({table})')]
                if expected != actual:
                    raise ValueError(f'Invalid facial recognition schema shape: {table}.')
                expected_keys = {tuple(row[2:]) for row in reference.execute(f'PRAGMA foreign_key_list({table})')}
                actual_keys = {tuple(row[2:]) for row in db.execute(f'PRAGMA foreign_key_list({table})')}
                if expected_keys != actual_keys:
                    raise ValueError(f'Invalid facial recognition foreign-key constraints: {table}.')
                def unique_keys(connection):
                    return {tuple((row[2], row[4]) for row in connection.execute(
                                'SELECT * FROM pragma_index_xinfo(?) WHERE key=1 ORDER BY seqno', (index[1],)))
                            for index in connection.execute(f'PRAGMA index_list({table})')
                            if index[2] and not index[4]}
                if not unique_keys(reference) <= unique_keys(db):
                    raise ValueError(f'Invalid facial recognition unique constraints: {table}.')
                if _checks(reference, table) != _checks(db, table):
                    raise ValueError(f'Invalid facial recognition CHECK constraints: {table}.')
        if db.execute('PRAGMA foreign_key_check').fetchone():
            raise ValueError('Facial recognition state has invalid foreign-key references.')

    @property
    def installation(self):
        row = self.connection.execute("SELECT value FROM metadata WHERE key='installation'").fetchone()
        if row is None:
            raise ValueError('Recognition installation identity is missing.')
        try:
            return str(uuid.UUID(row[0]))
        except (ValueError, TypeError, AttributeError) as error:
            raise ValueError('Recognition installation identity is invalid.') from error

    def start_run(self, scope, selection, configuration):
        identifier = str(uuid.uuid4())
        with self.connection:
            self.connection.execute('INSERT INTO runs(id,scope,selection,config,started,status) VALUES (?,?,?,?,?,?)',
                                    (identifier, scope, encoded(selection), encoded(configuration), time.time(), 'running'))
        return identifier

    def progress(self, run, phase, values):
        with self.connection:
            self.connection.execute('INSERT OR REPLACE INTO progress VALUES (?,?,?)', (run, phase, encoded(values)))

    def finish_run(self, run, status, summary, error=None):
        with self.connection:
            self.connection.execute('UPDATE runs SET finished=?,status=?,summary=?,error=? WHERE id=?',
                                    (time.time(), status, encoded(summary), error, run))

    def backup(self, destination):
        destination = Path(destination)
        if self.connection.in_transaction:
            raise ValueError('Commit or roll back the active recognition transaction before backup.')
        # A crash can leave a staging file, never an empty/partial public backup.
        # link() publishes the completed inode exclusively, including on Windows.
        descriptor, staged = tempfile.mkstemp(prefix='.recognition-backup-', suffix='.tmp', dir=destination.parent)
        staged = Path(staged)
        try:
            os.close(descriptor)
            with closing(sqlite3.connect(staged), name='close recognition backup') as target:
                self.connection.backup(target)
            with staged.open('r+b') as saved:
                os.fsync(saved.fileno())
            os.link(staged, destination)
        finally:
            cleanup([('remove staged recognition backup', lambda: staged.unlink(missing_ok=True))])

    def close(self):
        if (self.connection is not None or self.lock is not None) and threading.get_ident() != self._owner_thread:
            raise sqlite3.ProgrammingError('Recognition SQLite must be closed by its owning thread.')
        def close_connection():
            if self.connection is None:
                return
            try:
                self.connection.close()
            except BaseException:
                # A failed close may still own a live SQLite handle. Release the
                # writer only after the underlying connection is confirmed shut.
                try:
                    self._sqlite_connection.in_transaction
                except sqlite3.ProgrammingError:
                    self.connection = self._sqlite_connection = None
                raise
            self.connection = self._sqlite_connection = None
        def release_writer():
            if self.connection is None:
                self._release_lock()
        cleanup([('close recognition SQLite connection', close_connection),
                 ('release recognition writer lock', release_writer)])

    def _release_lock(self):
        lock, self.lock = self.lock, None
        if lock is not None:
            lock.close()

    def __exit__(self, exc_type, exc, traceback):
        cleanup([('close recognition database', self.close)], primary=exc)
