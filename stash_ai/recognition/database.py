"""One independently owned SQLite for all face evidence, history and recovery."""
from contextlib import AbstractContextManager, closing
import json
from pathlib import Path
import sqlite3
import time
import uuid

APPLICATION_ID = 0x46524347  # FRCG; never accept a tagger or arbitrary database.
SCHEMA_VERSION = 2
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


def encoded(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False)


class FaceDatabase(AbstractContextManager):
    def __init__(self, path):
        self.path = Path(path).expanduser().resolve()
        self.connection = self.lock = None
        try:
            if self.path.exists():
                if self.path.stat().st_nlink > 1:
                    raise ValueError('Recognition SQLite must not be a hard-linked file.')
                with closing(sqlite3.connect(self.path.as_uri() + '?mode=ro', uri=True)) as check:
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
            self.connection.row_factory = sqlite3.Row
            self.connection.execute('PRAGMA foreign_keys=ON')
            self.connection.execute('PRAGMA cache_size=-32768')
            self._initialize()
        except BaseException:
            self.close()
            raise

    def _initialize(self):
        db = self.connection
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
                db.rollback()
                raise
        elif version == 1 and db.execute('PRAGMA application_id').fetchone()[0] == APPLICATION_ID:
            self._validate(SCHEMA_V1, TABLES_V1, tables)
            try:
                db.executescript('BEGIN IMMEDIATE;\n' + MERGE_SCHEMA)
                db.execute(f'PRAGMA user_version={SCHEMA_VERSION}')
                db.commit()
            except BaseException:
                db.rollback()
                raise
        elif db.execute('PRAGMA application_id').fetchone()[0] != APPLICATION_ID or version != SCHEMA_VERSION:
            raise ValueError('Unsupported or incomplete facial recognition schema; restore a consistent backup.')
        tables = {r[0] for r in db.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        self._validate(SCHEMA, TABLES, tables)

    def _validate(self, schema, tables, actual_tables):
        db = self.connection
        if actual_tables != set(tables):
            raise ValueError('Unsupported or incomplete facial recognition schema; restore a consistent backup.')
        # Validate full column shape, not only the existence of tables.
        with closing(sqlite3.connect(':memory:')) as reference:
            reference.executescript(schema)
            for table in tables:
                expected = [(r[1], r[2], r[3], r[5]) for r in reference.execute(f'PRAGMA table_info({table})')]
                actual = [(r[1], r[2], r[3], r[5]) for r in db.execute(f'PRAGMA table_info({table})')]
                if expected != actual:
                    raise ValueError(f'Invalid facial recognition schema shape: {table}.')
        if db.execute('PRAGMA foreign_key_check').fetchone():
            raise ValueError('Facial recognition state has invalid foreign-key references.')

    @property
    def installation(self):
        row = self.connection.execute("SELECT value FROM metadata WHERE key='installation'").fetchone()
        if row is None:
            raise ValueError('Recognition installation identity is missing.')
        return str(uuid.UUID(row[0]))

    def start_run(self, scope, selection, configuration):
        identifier = str(uuid.uuid4())
        with self.connection:
            self.connection.execute("UPDATE runs SET status='interrupted',finished=? "
                                    "WHERE status='running' AND scope=? AND selection=?",
                                    (time.time(), scope, encoded(selection)))
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
        with destination.open('xb'):
            pass
        try:
            with closing(sqlite3.connect(destination)) as target:
                self.connection.backup(target)
        except BaseException:
            destination.unlink(missing_ok=True)
            raise

    def close(self):
        connection, self.connection = self.connection, None
        lock, self.lock = self.lock, None
        try:
            if connection is not None:
                connection.close()
        finally:
            if lock is not None:
                lock.close()

    def __exit__(self, *args):
        self.close()
