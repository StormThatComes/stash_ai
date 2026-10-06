"""Persistent SQLite state store for merged-output evidence and publications."""

from __future__ import annotations

from contextlib import AbstractContextManager
import hashlib
from urllib.parse import urlparse

from stash_ai.runtime.timing import measured, stage
from stash_ai.storage.codec import encode, decode
from stash_ai.storage.database import StateDatabase


from stash_ai.storage.results import ResultStore
from stash_ai.storage.publications import PublicationStore
from stash_ai.storage.metadata_store import MetadataStore


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
        from stash_ai.runtime.cache import RamCache
        if allow_parent_publication_updates and not dry_run:
            raise ValueError("Parent publication updates are only valid for read-only preview stores.")
        self.cache = RamCache(cache_bytes, rss_budget=memory_budget_bytes)
        self.scope = scope
        self.report = None
        self.preview_data_version = None
        self.abandoned_cleanups = {}
        # Parallel merge workers replay immutable inference evidence after the
        # inference phase has finished. The owning parent may legitimately
        # commit snapshots/merge_checks while those readers are still alive.
        self.allow_parent_publication_updates = bool(allow_parent_publication_updates)
        self._database = None
        self.connection = None
        try:
            database = StateDatabase(path, dry_run=dry_run)
            self._database = database
            self.path = database.path
            self.connection = database.connection
            self.metadata = MetadataStore(self)
            self.results = ResultStore(self)
            self.publications = PublicationStore(self)
            self.preview_source = database.preview_source
            if self.preview_source:
                self.preview_data_version = database.preview_data_version
            from stash_ai.storage.schema import initialize_features, load_features
            if database.fresh:
                initialize_features(self)
            else:
                load_features(self)
            if database.fresh:
                from stash_ai.storage.schema import LATEST_SCHEMA_VERSION
                self.connection.execute(f'PRAGMA user_version={LATEST_SCHEMA_VERSION}')
                self.connection.commit()
            if self.cleanup_pending and not allow_cleanup:
                raise RuntimeError('Global cleanup is interrupted; rerun the same --models all cleanup mode '
                                   'with the same exclusion policies, or use --abandon-pending with current policies.')
            from stash_ai.storage.initialization import InitializationStore
            self.initialization = InitializationStore(self)
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
        from stash_ai.storage.preview import check_source
        check_source(self)
        return None if self.connection.in_transaction else self.cache.get(key)

    def table(self, name):
        from stash_ai.storage.preview import table
        return table(self, name)

    @measured('state_cache')
    def retain(self, key, value):
        if not self.connection.in_transaction:
            self.cache.put(key, value)

    @measured('state_sql_read')
    def read_rows(self, sql, params):
        return self.connection.execute(sql, params).fetchall()

    def pack(self, value):
        return encode(value, self.report)

    def pack_snapshot(self, value):
        """Encode shared proofs using the current snapshot format."""
        from stash_ai.storage.snapshots import compact
        with stage(self.report, 'state_snapshot_compact'):
            packed = compact(value)
        return self.pack(packed)

    def unpack(self, value):
        return decode(value, self.report)

    def unpack_snapshot(self, value):
        from stash_ai.storage.snapshots import decode
        return decode(value, self.report)

    def records(self, scene, kind, model=None, *, borrow=False):
        return self.results.records(scene, kind, model, borrow=borrow)

    def model_record(self, scene, kind, model, *, borrow=False):
        return self.results.model_record(scene, kind, model, borrow=borrow)

    def result_metadata(self, scene, kind, model):
        return self.metadata.get(scene, kind, model)

    def result_presence(self, models):
        return self.results.result_presence(models)

    def result_index_presence(self, models):
        return self.results.result_index_presence(models)

    def sources(self, scene, kind):
        return self.results.sources(scene, kind)

    def preload_sources(self):
        return self.results.preload_sources()

    def put(self, scene, model, outputs):
        return self.results.put(scene, model, outputs)

    def replace_record(self, scene, model, kind, record):
        return self.results.replace_record(scene, model, kind, record)

    def set_result_threshold(self, scene, model, kind, threshold, record, *, metadata_only=False):
        return self.results.set_threshold(scene, model, kind, threshold, record,
                                          metadata_only=metadata_only)


    def snapshot(self, scene, kind, *, borrow=False):
        return self.publications.snapshot(scene, kind, borrow=borrow)

    def has_pending_publications(self):
        return self.publications.has_pending_publications()

    def has_pending_publication(self, scene):
        return self.publications.has_pending(scene)

    def pending_publication_scenes(self):
        return self.publications.pending_scenes()

    def prepare(self, scene, snapshots):
        return self.publications.prepare(scene, snapshots)

    def publish(self, scene, snapshots):
        return self.publications.publish(scene, snapshots)

    def commit_prepared(self, scene, kinds, *, publication=None):
        return self.publications.commit_prepared(scene, kinds, publication=publication)
