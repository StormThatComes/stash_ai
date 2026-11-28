"""Publish a privately prepared replay against its original SQLite generation."""
from contextlib import contextmanager
from collections import OrderedDict
import time
import uuid


class SourceChanged(RuntimeError):
    """Another connection committed after preparation started."""


class Connection:
    """Materialize nested savepoints when the first write starts publication."""
    def __init__(self, raw, borrowed):
        self.raw = raw
        self.started = borrowed
        self.version = self._version()
        self.contexts = []
        self.fingerprints = OrderedDict()
        self.derived_records = OrderedDict()
        self.derived_bytes = 0
        self.write_start = None
        self.cache_write_seconds = 0.

    @property
    def in_transaction(self):
        # Nested stages join this logical generation, including its read phase.
        return True

    def _version(self):
        return self.raw.execute('PRAGMA main.data_version').fetchone()[0]

    def _check(self):
        if self._version() != self.version:
            raise SourceChanged('Recognition inputs changed during rebuild preparation.')

    def _start(self):
        if not self.started:
            self.raw.execute('BEGIN IMMEDIATE')
            try:
                self._check()
            except BaseException:
                self.raw.rollback()
                raise
            self.started = True
            self.write_start = time.monotonic()
            for name in self.contexts:
                self.raw.execute('SAVEPOINT ' + name)
            self.flush_cache()

    def prepare_pixel_catalogue(self, scope, namespace, disk_bytes):
        """Refreshing the derived pixel index does not publish identity decisions."""
        from .pixel_catalogue import prepare
        if self.started:
            return prepare(self, scope, namespace, disk_bytes=disk_bytes)
        self.raw.execute('BEGIN IMMEDIATE')
        start = time.monotonic()
        try:
            self._check()
            result = prepare(self.raw, scope, namespace, disk_bytes=disk_bytes)
            self.raw.commit()
            return result
        except BaseException:
            self.raw.rollback()
            raise
        finally:
            self.cache_write_seconds += time.monotonic() - start

    def stage_fingerprint(self, kind, key, policy, generation, value):
        if self.started:
            return False
        name = kind, key, policy
        self.fingerprints[name] = generation, value
        self.fingerprints.move_to_end(name)
        if len(self.fingerprints) > 2048:
            self.fingerprints.popitem(last=False)
        return True

    def stage_derived(self, store, key, record, charge):
        if self.started:
            return False
        previous = self.derived_records.pop(key, None)
        if previous:
            self.derived_bytes -= previous[2]
        self.derived_records[key] = store, record, charge
        self.derived_bytes += charge
        while self.derived_bytes > min(store.budget, 8 * 1024**2):
            _, evicted = self.derived_records.popitem(last=False)
            self.derived_bytes -= evicted[2]
        return True

    def flush_cache(self):
        if not (self.fingerprints or self.derived_records):
            return
        if not self.started:
            self._start()
            return
        while self.derived_records:
            key, (store, record, charge) = self.derived_records.popitem(last=False)
            self.derived_bytes -= charge
            store._put(key, record)
        if self.fingerprints:
            from .dependencies import store
            self._start()
            while self.fingerprints:
                name, (generation, value) = self.fingerprints.popitem(last=False)
                store(self, *name, generation, value)

    def __getattr__(self, name):
        return getattr(self.raw, name)

    def __enter__(self):
        name = 'rebuild_' + uuid.uuid4().hex
        if self.started:
            self.raw.execute('SAVEPOINT ' + name)
        self.contexts.append(name)
        return self

    def __exit__(self, kind, error, traceback):
        name = self.contexts.pop()
        if self.started:
            if kind is not None:
                self.raw.execute('ROLLBACK TO ' + name)
            self.raw.execute('RELEASE ' + name)

    def execute(self, sql, parameters=()):
        if not self.started:
            words = sql.lstrip().split(None, 2)
            operation = words[0].upper() if words else ''
            if operation == 'SAVEPOINT':
                self.contexts.append(words[1])
                return self.raw.execute('SELECT 1 WHERE 0')
            if operation == 'RELEASE' and self.contexts and words[-1] == self.contexts[-1]:
                self.contexts.pop()
                return self.raw.execute('SELECT 1 WHERE 0')
            if operation == 'ROLLBACK' and len(words) == 3 and words[2] in self.contexts:
                return self.raw.execute('SELECT 1 WHERE 0')
            # Only audited read statements may run outside the transaction.
            pragma = sql.strip().lower()
            read_pragma = (operation == 'PRAGMA' and '=' not in pragma and len(words) == 2 and
                           words[1].lower().split('.')[-1].split('(')[0] in {
                               'data_version','schema_version','user_version','application_id',
                               'query_only','journal_mode','function_list','database_list','table_info','index_info',
                               'page_size','page_count','foreign_keys','cache_size'})
            if operation != 'SELECT' and not read_pragma:
                self._start()
        cursor = self.raw.execute(sql, parameters)
        # Explicit dependency savepoints can span the first durable write.
        if self.started and sql.lstrip().upper().startswith('RELEASE ') and self.contexts:
            if sql.split()[-1] == self.contexts[-1]:
                self.contexts.pop()
        return cursor

    def executemany(self, sql, rows):
        self._start()
        return self.raw.executemany(sql, rows)

    def cursor(self, *args, **kwargs):
        self._start()
        return self.raw.cursor(*args, **kwargs)

    def commit(self):
        # Stages may finish, but the complete rebuild owns durability.
        pass

    def rollback(self):
        raise RuntimeError('Rebuild stages must roll back their savepoint context.')

    def executescript(self, sql):
        raise RuntimeError('Implicit-commit scripts are forbidden during an identity rebuild.')


@contextmanager
def atomic(database):
    raw = database.connection
    borrowed = raw.in_transaction
    wrapper = Connection(raw, borrowed)
    database.connection = wrapper
    try:
        with wrapper:
            yield
            if not wrapper.started:
                wrapper._check()
                wrapper.flush_cache()
        if not borrowed and wrapper.started:
            raw.commit()
    except BaseException:
        if not borrowed and raw.in_transaction:
            raw.rollback()
        raise
    finally:
        database.connection = raw
        database.rebuild_writer_seconds = wrapper.cache_write_seconds + (time.monotonic() - wrapper.write_start
                                           if wrapper.write_start is not None else 0.)
