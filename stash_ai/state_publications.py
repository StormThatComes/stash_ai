"""Published/pending snapshot persistence behind StateStore."""
from __future__ import annotations

from copy import deepcopy

from stash_ai.runtime import measured


class PublicationStore:
    """Manage durable publication intent while StateStore owns DB/cache lifecycle."""

    def __init__(self, store):
        self.store = store

    @property
    def report(self):
        return getattr(self.store, 'report', None)

    @measured('state_read')
    def snapshot(self, scene, kind, *, borrow=False):
        store = self.store
        key = ('snapshot', str(scene), kind)
        cached = store.cached(key)
        if cached is not None:
            return cached if borrow else deepcopy(cached)
        rows = store.read_rows(
            "SELECT published,pending FROM " + store.table('snapshots') + " WHERE scope=? AND scene=? AND kind=?",
            (store.scope, str(scene), kind))
        from stash_ai.cleanup import mask_snapshot
        values = tuple(mask_snapshot(store, store.unpack(item), kind) if item is not None else None
                       for item in rows[0]) if rows else (None, None)
        store.retain(key, values)
        return values if borrow else deepcopy(values)

    def has_pending_publications(self):
        store = self.store
        return store.connection.execute(
            'SELECT 1 FROM ' + store.table('snapshots') +
            ' WHERE scope=? AND pending IS NOT NULL LIMIT 1', (store.scope,)).fetchone() is not None

    @measured('state_write')
    def prepare(self, scene, snapshots):
        store = self.store
        from stash_ai.cleanup import stamp_snapshot
        with store.connection:
            for kind, data in snapshots.items():
                store.cache.discard(('snapshot', str(scene), kind))
                if store.preview_source:
                    store.connection.execute(
                        'INSERT OR IGNORE INTO snapshots SELECT * FROM '
                        'cleanup_source.snapshots WHERE scope=? AND scene=? AND kind=?',
                        (store.scope, str(scene), kind))
                store.connection.execute("INSERT INTO snapshots VALUES (?,?,?,NULL,?) "
                    "ON CONFLICT(scope,scene,kind) DO UPDATE SET pending=excluded.pending",
                    (store.scope, str(scene), kind, store.pack(stamp_snapshot(store, data))))

    @measured('state_write')
    def publish(self, scene, snapshots):
        store = self.store
        from stash_ai.cleanup import stamp_snapshot
        from stash_ai.recovery import clear_publications
        with store.connection:
            for kind, data in snapshots.items():
                store.cache.discard(('snapshot', str(scene), kind))
                store.connection.execute("INSERT INTO snapshots VALUES (?,?,?,?,NULL) "
                    "ON CONFLICT(scope,scene,kind) DO UPDATE SET published=excluded.published,pending=NULL",
                    (store.scope, str(scene), kind, store.pack(stamp_snapshot(store, data))))
                clear_publications(store, scene, kind)

    @measured('state_write')
    def commit_prepared(self, scene, kinds, *, publication=None):
        """Promote durable intent without encoding large evidence a second time."""
        store = self.store
        from stash_ai.recovery import clear_publications
        with store.connection:
            for kind in kinds:
                store.cache.discard(('snapshot', str(scene), kind))
                changed = store.connection.execute(
                    'UPDATE snapshots SET published=pending,pending=NULL '
                    'WHERE scope=? AND scene=? AND kind=? AND pending IS NOT NULL',
                    (store.scope, str(scene), kind)).rowcount
                if changed != 1:
                    raise RuntimeError('Publication intent disappeared before commit.')
                clear_publications(store, scene, kind)
            if publication is not None:
                from stash_ai.performance import record_check
                record_check(store, scene, *publication)
