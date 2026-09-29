"""Persistent SQLite state store for merged-output evidence and publications."""

from __future__ import annotations

from contextlib import AbstractContextManager
import hashlib
from urllib.parse import urlparse

from stash_ai.runtime import measured
from stash_ai.state_codec import encode, decode
from stash_ai.state_db import APPLICATION_ID, StateDatabase


from stash_ai.state_results import KINDS, ResultStore
from stash_ai.state_publications import PublicationStore


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
                 memory_budget_bytes=0, allow_parent_publication_updates=False):
        from stash_ai.runtime import RamCache
        if allow_parent_publication_updates and not dry_run:
            raise ValueError("Parent publication updates are only valid for read-only preview stores.")
        self.cache = RamCache(cache_bytes, rss_budget=memory_budget_bytes)
        self.scope = scope
        self.allow_parent_publication_updates = bool(allow_parent_publication_updates)
        self._database = None
        self.connection = None
        try:
            database = StateDatabase(path, dry_run=dry_run)
            self._database = database
            self.path = database.path
            self.connection = database.connection
            self.results = ResultStore(self)
            self.publications = PublicationStore(self)
            self.preview_source = database.preview_source
            if self.preview_source:
                self.preview_data_version = database.preview_data_version
            version = database.version
            from stash_ai.progress import initialize_progress
            initialize_progress(self, version)
            from stash_ai.cleanup import initialize
            initialize(self, version)
            if self.preview_source:
                self.connection.execute('INSERT INTO cleanup_state SELECT * FROM cleanup_source.cleanup_state WHERE scope=?', (self.scope,))
                self.connection.commit()
                initialize(self, version)
            from stash_ai.performance import initialize as initialize_performance
            initialize_performance(self)
            from stash_ai.recovery import initialize as initialize_recovery
            initialize_recovery(self)
            from stash_ai.preview import initialize as initialize_preview
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
        from stash_ai.preview import check_source
        check_source(self)
        return None if self.connection.in_transaction else self.cache.get(key)

    def table(self, name):
        from stash_ai.preview import table
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

    def records(self, scene, kind, model=None, *, borrow=False):
        return self.results.records(scene, kind, model, borrow=borrow)

    def sources(self, scene, kind):
        return self.results.sources(scene, kind)

    def preload_sources(self):
        return self.results.preload_sources()

    def put(self, scene, model, outputs, replace=None):
        return self.results.put(scene, model, outputs, replace)

    def _put(self, scene, model, outputs, replace=None):
        return self.results._put(scene, model, outputs, replace)

    def replace_records(self, scene, model, kind, segments):
        return self.results.replace_records(scene, model, kind, segments)

    def imported_legacy(self, scene, kind, *, borrow=False):
        return self.results.imported_legacy(scene, kind, borrow=borrow)

    def remember_legacy(self, scene, kind, data):
        return self.results.remember_legacy(scene, kind, data)

    def snapshot(self, scene, kind, *, borrow=False):
        return self.publications.snapshot(scene, kind, borrow=borrow)

    def has_pending_publications(self):
        return self.publications.has_pending_publications()

    def prepare(self, scene, snapshots):
        return self.publications.prepare(scene, snapshots)

    def publish(self, scene, snapshots):
        return self.publications.publish(scene, snapshots)

    def commit_prepared(self, scene, kinds, *, publication=None):
        return self.publications.commit_prepared(scene, kinds, publication=publication)
