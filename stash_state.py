"""Persistent SQLite state store for merged-output evidence and publications."""

from __future__ import annotations

from contextlib import AbstractContextManager
from copy import deepcopy
import hashlib
import json
from urllib.parse import urlparse

from stash_runtime import measured
from stash_state_codec import encode, decode
from stash_state_db import APPLICATION_ID, StateDatabase
from stash_taggers import SOURCES


KINDS = ("scene", "markers")


def scope_for(url):
    parsed = urlparse(url)
    port = parsed.port or (443 if parsed.scheme == "https" else 80)
    canonical = f"{parsed.scheme.lower()}://{parsed.hostname.lower()}:{port}{parsed.path.rstrip('/')}"
    return hashlib.sha256(canonical.encode()).hexdigest()


class StateStore(AbstractContextManager):
    """Results commit after each successful inference; API writes have an intent log.

    A dry run uses a read-only source plus a sparse temporary overlay, no source lock.
    A writer holds a process lock for its whole run, including Stash requests.
    """
    def __init__(self, path, scope, dry_run=False, cache_bytes=0, *, allow_cleanup=False, cleanup_preview=False,
                 memory_budget_bytes=0):
        from stash_runtime import RamCache
        self.cache = RamCache(cache_bytes, rss_budget=memory_budget_bytes)
        self.scope = scope
        self._database = None
        self.connection = None
        try:
            database = StateDatabase(path, dry_run=dry_run)
            self._database = database
            self.path = database.path
            self.connection = database.connection
            self.preview_source = database.preview_source
            if self.preview_source:
                self.preview_data_version = database.preview_data_version
            version = database.version
            from stash_progress import initialize_progress
            initialize_progress(self, version)
            from stash_cleanup import initialize
            initialize(self, version)
            if self.preview_source:
                self.connection.execute('INSERT INTO cleanup_state SELECT * FROM cleanup_source.cleanup_state WHERE scope=?', (self.scope,))
                self.connection.commit()
                initialize(self, version)
            from stash_performance import initialize as initialize_performance
            initialize_performance(self)
            from stash_recovery import initialize as initialize_recovery
            initialize_recovery(self)
            from stash_preview import initialize as initialize_preview
            initialize_preview(self)
            if self.cleanup_pending and not allow_cleanup:
                raise RuntimeError('Global cleanup is interrupted; rerun the same --models all cleanup mode '
                                   'with the same exclusion policies, or use --abandon-pending with current policies.')
        except BaseException:
            self.close()
            raise

    def close(self):
        self.cache.clear()
        if self._database is not None:
            try:
                self._database.close()
            finally:
                self.connection = None

    def __exit__(self, *args):
        self.close()

    def cached(self, key):
        # A transaction may still roll back: never cache its provisional view.
        from stash_preview import check_source
        check_source(self)
        return None if self.connection.in_transaction else self.cache.get(key)

    def table(self, name):
        from stash_preview import table
        return table(self, name)

    @measured('state_cache')
    def retain(self, key, value):
        if not self.connection.in_transaction:
            self.cache.put(key, value)

    @measured('state_sql_read')
    def read_rows(self, sql, params):
        return self.connection.execute(sql, params).fetchall()

    def pack(self, value):
        return encode(value, getattr(self, 'report', None))

    def unpack(self, value):
        return decode(value, getattr(self, 'report', None))

    @measured('state_read')
    def records(self, scene, kind, model=None, *, borrow=False):
        """Public reads are detached; internal read-only consumers may borrow."""
        key = ('records', str(scene), kind)
        records = self.cached(key)
        if records is None:
            records = {model: self.unpack(data) for model, data in self.read_rows(
                "SELECT model,data FROM " + self.table('results') + " WHERE scope=? AND scene=? AND kind=?",
                (self.scope, str(scene), kind))}
            # Retired video samples are never inputs. Remove them from the decoded
            # view without scanning/recompressing every scene during DB migration.
            for segments in records.values():
                from stash_performance import StoredSegment
                for index, segment in enumerate(segments):
                    segment.pop("refinement", None)
                    segments[index] = StoredSegment(segment)
            self.retain(key, records)
        result = {model: records[model]} if model in records else {} if model else records
        return result if borrow else deepcopy(result)

    def sources(self, scene, kind):
        key = ('sources', str(scene), kind)
        value = self.cached(key)
        if value is None:
            value = {row[0] for row in self.connection.execute(
                "SELECT model FROM " + self.table('results') + " WHERE scope=? AND scene=? AND kind=?", (self.scope, str(scene), kind))}
            self.retain(key, value)
        return set(value)

    def preload_sources(self):
        if not self.cache.limit:
            return
        last, values = None, set()
        for scene, kind, model in self.connection.execute(
                'SELECT scene,kind,model FROM ' + self.table('results') + ' WHERE scope=? ORDER BY scene,kind', (self.scope,)):
            key = ('sources', scene, kind)
            if key != last:
                if last is not None:
                    self.cache.put(last, values)
                last, values = key, set()
            values.add(model)
        if last is not None:
            self.cache.put(last, values)

    @measured('state_write')
    def put(self, scene, model, outputs, replace=None):
        with self.connection:
            self._put(scene, model, outputs, replace)

    def _put(self, scene, model, outputs, replace=None):
        if model not in SOURCES or set(outputs) - set(KINDS):
            raise ValueError("Unknown result source or output kind.")
        for kind, segment in outputs.items():
            segment = {key: value for key, value in segment.items() if key != "refinement"}
            old = self.records(scene, kind, model, borrow=True).get(model, []) if not (replace or {}).get(kind, True) else []
            # Repeating an identical run with replacement disabled need not
            # accumulate identical frame streams merely due to its timestamp.
            keep = []
            if old:
                # Only append mode needs comparison. A shallow metadata copy
                # avoids duplicating all retained frame observations.
                def semantic(value):
                    data = dict(value)
                    if 'meta' in data:
                        data['meta'] = {key: item for key, item in data['meta'].items() if key != 'captured_at'}
                    # JSON normalizes tuples in new results to stored lists.
                    return json.dumps(data, sort_keys=True, allow_nan=False)
                current = semantic(segment)
                keep = [previous for previous in old if semantic(previous) != current]
            self.replace_records(scene, model, kind, keep + [segment])

    def replace_records(self, scene, model, kind, segments):
        """Replace a segment list inside the caller's transaction, including previews."""
        from stash_progress import index_result
        self.cache.discard(('records', str(scene), kind))
        self.cache.discard(('sources', str(scene), kind))
        self.cache.discard(('status', str(scene), model))
        self.connection.execute("INSERT INTO results VALUES (?,?,?,?,?) "
            "ON CONFLICT(scope,scene,model,kind) DO UPDATE SET data=excluded.data",
            (self.scope, str(scene), model, kind, self.pack(segments)))
        index_result(self, scene, model, kind, segments)

    @measured('state_read')
    def imported_legacy(self, scene, kind, *, borrow=False):
        key = ('legacy', str(scene), kind)
        cached = self.cached(key)
        if cached is not None:
            return cached if borrow else deepcopy(cached)
        rows = self.read_rows("SELECT data FROM " + self.table('legacy_imports') + " WHERE scope=? AND scene=? AND kind=?",
                              (self.scope, str(scene), kind))
        values = self.unpack(rows[0][0]) if rows else []
        self.retain(key, values)
        return values if borrow else deepcopy(values)

    @measured('state_write')
    def remember_legacy(self, scene, kind, data):
        from stash_performance import freeze
        previous = self.imported_legacy(scene, kind, borrow=True)
        known = {freeze(item) for item in previous}
        fresh = [item for item in data if freeze(item) not in known]
        if not fresh:
            return
        previous = previous + fresh
        self.cache.discard(('legacy', str(scene), kind))
        with self.connection:
            self.connection.execute("INSERT INTO legacy_imports VALUES (?,?,?,?) "
                "ON CONFLICT(scope,scene,kind) DO UPDATE SET data=excluded.data",
                (self.scope, str(scene), kind, self.pack(previous)))

    @measured('state_read')
    def snapshot(self, scene, kind, *, borrow=False):
        key = ('snapshot', str(scene), kind)
        cached = self.cached(key)
        if cached is not None:
            return cached if borrow else deepcopy(cached)
        rows = self.read_rows("SELECT published,pending FROM " + self.table('snapshots') + " WHERE scope=? AND scene=? AND kind=?",
                              (self.scope, str(scene), kind))
        from stash_cleanup import mask_snapshot
        values = tuple(mask_snapshot(self, self.unpack(item), kind) if item is not None else None
                       for item in rows[0]) if rows else (None, None)
        self.retain(key, values)
        return values if borrow else deepcopy(values)

    def has_pending_publications(self):
        return self.connection.execute('SELECT 1 FROM ' + self.table('snapshots') +
            ' WHERE scope=? AND pending IS NOT NULL LIMIT 1', (self.scope,)).fetchone() is not None

    @measured('state_write')
    def prepare(self, scene, snapshots):
        from stash_cleanup import stamp_snapshot
        with self.connection:
            for kind, data in snapshots.items():
                self.cache.discard(('snapshot', str(scene), kind))
                if self.preview_source:
                    self.connection.execute('INSERT OR IGNORE INTO snapshots SELECT * FROM '
                        'cleanup_source.snapshots WHERE scope=? AND scene=? AND kind=?',
                        (self.scope, str(scene), kind))
                self.connection.execute("INSERT INTO snapshots VALUES (?,?,?,NULL,?) "
                    "ON CONFLICT(scope,scene,kind) DO UPDATE SET pending=excluded.pending",
                    (self.scope, str(scene), kind, self.pack(stamp_snapshot(self, data))))

    @measured('state_write')
    def publish(self, scene, snapshots):
        from stash_cleanup import stamp_snapshot
        from stash_recovery import clear_publications
        with self.connection:
            for kind, data in snapshots.items():
                self.cache.discard(('snapshot', str(scene), kind))
                self.connection.execute("INSERT INTO snapshots VALUES (?,?,?,?,NULL) "
                    "ON CONFLICT(scope,scene,kind) DO UPDATE SET published=excluded.published,pending=NULL",
                    (self.scope, str(scene), kind, self.pack(stamp_snapshot(self, data))))
                clear_publications(self, scene, kind)


    @measured('state_write')
    def commit_prepared(self, scene, kinds, *, publication=None):
        """Promote durable intent without encoding large evidence a second time."""
        from stash_recovery import clear_publications
        with self.connection:
            for kind in kinds:
                self.cache.discard(('snapshot', str(scene), kind))
                changed = self.connection.execute('UPDATE snapshots SET published=pending,pending=NULL '
                    'WHERE scope=? AND scene=? AND kind=? AND pending IS NOT NULL',
                    (self.scope, str(scene), kind)).rowcount
                if changed != 1:
                    raise RuntimeError('Publication intent disappeared before commit.')
                clear_publications(self, scene, kind)
            if publication is not None:
                from stash_performance import record_check
                record_check(self, scene, *publication)
