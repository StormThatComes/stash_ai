"""SQLite connection ownership, writer locking and base state schema validation."""
from __future__ import annotations

from contextlib import AbstractContextManager
import os
from pathlib import Path
import sqlite3


from stash_ai.storage.schema import APPLICATION_ID, LATEST_SCHEMA_VERSION


class StateDatabase(AbstractContextManager):
    """Own a locked writer or a temporary preview with a read-only source.

    StateStore initializes feature-specific schemas after this resource is open.
    Closing it releases the connection and lock even if store setup failed.
    """

    def __init__(self, path, *, dry_run=False):
        self.path = Path(path).expanduser().resolve()
        self.lock_file, self.connection = None, None
        self.preview_source = False
        try:
            if dry_run:
                self._open_preview()
            else:
                self._open_writer()
            self._initialize_schema()
        except BaseException:
            self.close()
            raise

    def _open_preview(self):
        # Empty filename = SQLite-managed temporary DB, deleted at close.
        # Its bounded page cache prevents a long preview consuming RAM
        # proportional to the entire inference database.
        self.connection = sqlite3.connect("", uri=True)
        if not self.path.exists():
            return
        source = sqlite3.connect(self.path.as_uri() + "?mode=ro", uri=True)
        try:
            source_version = source.execute('PRAGMA user_version').fetchone()[0]
            if (5 <= source_version <= LATEST_SCHEMA_VERSION and
                    source.execute('PRAGMA application_id').fetchone()[0] == APPLICATION_ID):
                # Modern schemas can be read directly; only preview
                # changes occupy the temporary overlay.
                self.connection.execute('ATTACH DATABASE ? AS cleanup_source',
                                        (self.path.as_uri() + '?mode=ro',))
                self.connection.execute(f'PRAGMA user_version={source_version}')
                self.preview_source = True
                self.preview_data_version = self.connection.execute(
                    'PRAGMA cleanup_source.data_version').fetchone()[0]
            else:
                source.backup(self.connection)
        finally:
            source.close()

    def _open_writer(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.lock_file = open(str(self.path) + ".lock", "a+b")
        if self.lock_file.seek(0, os.SEEK_END) == 0:
            self.lock_file.write(b"0")
            self.lock_file.flush()
        self.lock_file.seek(0)
        try:
            if os.name == "nt":
                import msvcrt
                msvcrt.locking(self.lock_file.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(self.lock_file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            raise RuntimeError("Another merged-output process is using this state DB.") from exc
        self.connection = sqlite3.connect(self.path, timeout=30)

    def _initialize_schema(self):
        db = self.connection
        application = db.execute("PRAGMA application_id").fetchone()[0]
        self.version = db.execute("PRAGMA user_version").fetchone()[0]
        tables = db.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
        if (application not in (0, APPLICATION_ID) or self.version not in range(LATEST_SCHEMA_VERSION + 1)
                or (tables and application != APPLICATION_ID)):
            raise ValueError("Not a supported stash-taggers state DB; choose another --state-db.")
        db.executescript(f"""
            PRAGMA application_id={APPLICATION_ID};
            CREATE TABLE IF NOT EXISTS results (
                scope TEXT NOT NULL, scene TEXT NOT NULL, model TEXT NOT NULL,
                kind TEXT NOT NULL, data BLOB NOT NULL,
                PRIMARY KEY(scope, scene, model, kind));
            CREATE TABLE IF NOT EXISTS result_thresholds (
                scope TEXT NOT NULL, scene TEXT NOT NULL, model TEXT NOT NULL,
                kind TEXT NOT NULL, threshold REAL NOT NULL
                    CHECK(typeof(threshold) IN ('integer','real') AND threshold BETWEEN 0 AND 1),
                PRIMARY KEY(scope, scene, model, kind));
            CREATE TABLE IF NOT EXISTS result_metadata (
                scope TEXT NOT NULL, scene TEXT NOT NULL, model TEXT NOT NULL,
                kind TEXT NOT NULL, format INTEGER NOT NULL, data TEXT NOT NULL,
                PRIMARY KEY(scope, scene, model, kind));
            CREATE TRIGGER IF NOT EXISTS results_delete_threshold
                AFTER DELETE ON results BEGIN
                    DELETE FROM result_thresholds WHERE scope=OLD.scope AND scene=OLD.scene
                        AND model=OLD.model AND kind=OLD.kind;
                END;
            CREATE TABLE IF NOT EXISTS snapshots (
                scope TEXT NOT NULL, scene TEXT NOT NULL, kind TEXT NOT NULL,
                published BLOB, pending BLOB,
                PRIMARY KEY(scope, scene, kind));
            CREATE TABLE IF NOT EXISTS legacy_imports (
                scope TEXT NOT NULL, scene TEXT NOT NULL, kind TEXT NOT NULL,
                data BLOB NOT NULL, PRIMARY KEY(scope, scene, kind));
        """)
        # Derived metadata stays safe when older executables write results.
        # INSERT also covers SQLite REPLACE, whose DELETE triggers may be disabled.
        for event, rows in (('INSERT', ('NEW',)), ('UPDATE', ('OLD', 'NEW')),
                            ('DELETE', ('OLD',))):
            deletes = ''.join(
                f'DELETE FROM result_metadata WHERE scope={row}.scope AND scene={row}.scene '
                f'AND model={row}.model AND kind={row}.kind;' for row in rows)
            db.execute(f'''CREATE TRIGGER IF NOT EXISTS results_metadata_{event.lower()}
                AFTER {event} ON results BEGIN {deletes} END''')

    def close(self):
        connection, self.connection = self.connection, None
        try:
            if connection is not None:
                connection.close()
        finally:
            lock_file, self.lock_file = self.lock_file, None
            if lock_file is not None:
                # Closing the descriptor releases the advisory lock on both platforms.
                lock_file.close()

    def __exit__(self, *args):
        self.close()

