"""Anti-duplication legacy ledger, independent of inference result persistence."""
from copy import deepcopy
from stash_ai.runtime.timing import measured


class LegacyStore:
    def __init__(self, store):
        self.store = store

    @property
    def report(self):
        return self.store.report

    @measured('state_read')
    def records(self, scene, kind, *, borrow=False):
        store = self.store
        key = ('legacy', str(scene), kind)
        cached = store.cached(key)
        if cached is not None:
            return cached if borrow else deepcopy(cached)
        rows = store.read_rows(
            "SELECT data FROM " + store.table('legacy_imports') + " WHERE scope=? AND scene=? AND kind=?",
            (store.scope, str(scene), kind))
        values = store.unpack(rows[0][0]) if rows else []
        store.retain(key, values)
        return values if borrow else deepcopy(values)

    @measured('state_write')
    def remember(self, scene, kind, data):
        with self.store.connection:
            self.remember_in_transaction(scene, kind, data)

    def remember_in_transaction(self, scene, kind, data):
        """Store ledger entries under the caller's evidence transaction."""
        store = self.store
        from stash_ai.signatures import freeze
        previous = store.imported_legacy(scene, kind, borrow=True)
        known = {freeze(item) for item in previous}
        fresh = [item for item in data if freeze(item) not in known]
        if not fresh:
            return
        previous = previous + fresh
        store.cache.discard(('legacy', str(scene), kind))
        store.connection.execute("INSERT INTO legacy_imports VALUES (?,?,?,?) "
            "ON CONFLICT(scope,scene,kind) DO UPDATE SET data=excluded.data",
            (store.scope, str(scene), kind, store.pack(previous)))
