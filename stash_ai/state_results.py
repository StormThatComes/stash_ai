"""Inference result and legacy-evidence persistence behind StateStore."""
from __future__ import annotations

from copy import deepcopy
import json

from stash_ai.runtime import measured
from stash_ai.taggers import SOURCES


KINDS = ("scene", "markers")


class ResultStore:
    """Persist per-model inference results while StateStore owns DB/cache lifecycle."""

    def __init__(self, store):
        self.store = store

    @property
    def report(self):
        return getattr(self.store, 'report', None)

    @staticmethod
    def _normalize_segments(segments):
        """Apply the lazy compatibility view used by all decoded result reads."""
        from stash_ai.performance import StoredSegment
        for index, segment in enumerate(segments):
            segment.pop("refinement", None)
            segments[index] = StoredSegment(segment)
        return segments

    @measured('state_read')
    def records(self, scene, kind, model=None, *, borrow=False):
        """Public reads are detached; internal read-only consumers may borrow."""
        store = self.store
        key = ('records', str(scene), kind)
        records = store.cached(key)
        if records is None:
            records = {model: self._normalize_segments(store.unpack(data)) for model, data in store.read_rows(
                "SELECT model,data FROM " + store.table('results') + " WHERE scope=? AND scene=? AND kind=?",
                (store.scope, str(scene), kind))}
            store.retain(key, records)
        result = {model: records[model]} if model in records else {} if model else records
        return result if borrow else deepcopy(result)

    @measured('state_read')
    def model_records(self, scene, kind, model, *, borrow=False):
        """Read one model payload without decoding or caching sibling model blobs."""
        store = self.store
        cached = store.cached(('records', str(scene), kind))
        if cached is not None:
            segments = cached.get(model, [])
            return segments if borrow else deepcopy(segments)
        rows = store.read_rows(
            "SELECT data FROM " + store.table('results') +
            " WHERE scope=? AND scene=? AND model=? AND kind=?",
            (store.scope, str(scene), model, kind))
        segments = self._normalize_segments(store.unpack(rows[0][0])) if rows else []
        return segments if borrow else deepcopy(segments)

    def result_presence(self, models):
        """Return lightweight saved-result keys without decoding payload BLOBs."""
        store = self.store
        from stash_ai.preview import check_source
        check_source(store)
        selected = tuple(dict.fromkeys(models))
        if not selected:
            return set()
        placeholders = ','.join('?' for _ in selected)
        rows = store.read_rows(
            "SELECT scene,model,kind FROM " + store.table('results') +
            f" WHERE scope=? AND model IN ({placeholders})",
            (store.scope, *selected))
        return {(str(scene), model, kind) for scene, model, kind in rows}

    def result_index_presence(self, models):
        """Return indexed result keys without decoding result payloads."""
        store = self.store
        from stash_ai.preview import check_source
        check_source(store)
        selected = tuple(dict.fromkeys(models))
        if not selected:
            return set()
        placeholders = ','.join('?' for _ in selected)
        rows = store.read_rows(
            "SELECT scene,model,kind FROM " + store.table('result_index') +
            f" WHERE scope=? AND model IN ({placeholders})",
            (store.scope, *selected))
        return {(str(scene), model, kind) for scene, model, kind in rows}

    def sources(self, scene, kind):
        store = self.store
        key = ('sources', str(scene), kind)
        value = store.cached(key)
        if value is None:
            value = {row[0] for row in store.connection.execute(
                "SELECT model FROM " + store.table('results') + " WHERE scope=? AND scene=? AND kind=?",
                (store.scope, str(scene), kind))}
            store.retain(key, value)
        return set(value)

    def preload_sources(self):
        store = self.store
        if not store.cache.limit:
            return
        last, values = None, set()
        for scene, kind, model in store.connection.execute(
                'SELECT scene,kind,model FROM ' + store.table('results') + ' WHERE scope=? ORDER BY scene,kind',
                (store.scope,)):
            key = ('sources', scene, kind)
            if key != last:
                if last is not None:
                    store.cache.put(last, values)
                last, values = key, set()
            values.add(model)
        if last is not None:
            store.cache.put(last, values)

    @measured('state_write')
    def put(self, scene, model, outputs, replace=None):
        with self.store.connection:
            # Route through the façade so existing StateStore interception remains valid.
            self.store._put(scene, model, outputs, replace)

    def _put(self, scene, model, outputs, replace=None):
        if model not in SOURCES or set(outputs) - set(KINDS):
            raise ValueError("Unknown result source or output kind.")
        for kind, segment in outputs.items():
            segment = {key: value for key, value in segment.items() if key != "refinement"}
            old = (self.store.records(scene, kind, model, borrow=True).get(model, [])
                   if not (replace or {}).get(kind, True) else [])
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
            self.store.replace_records(scene, model, kind, keep + [segment])

    def replace_records(self, scene, model, kind, segments):
        """Replace a segment list inside the caller's transaction, including previews."""
        store = self.store
        from stash_ai.progress import index_result
        store.cache.discard(('records', str(scene), kind))
        store.cache.discard(('sources', str(scene), kind))
        store.cache.discard(('status', str(scene), model))
        store.connection.execute("INSERT INTO results VALUES (?,?,?,?,?) "
            "ON CONFLICT(scope,scene,model,kind) DO UPDATE SET data=excluded.data",
            (store.scope, str(scene), model, kind, store.pack(segments)))
        index_result(store, scene, model, kind, segments)

    @measured('state_read')
    def imported_legacy(self, scene, kind, *, borrow=False):
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
    def remember_legacy(self, scene, kind, data):
        store = self.store
        from stash_ai.performance import freeze
        previous = store.imported_legacy(scene, kind, borrow=True)
        known = {freeze(item) for item in previous}
        fresh = [item for item in data if freeze(item) not in known]
        if not fresh:
            return
        previous = previous + fresh
        store.cache.discard(('legacy', str(scene), kind))
        with store.connection:
            store.connection.execute("INSERT INTO legacy_imports VALUES (?,?,?,?) "
                "ON CONFLICT(scope,scene,kind) DO UPDATE SET data=excluded.data",
                (store.scope, str(scene), kind, store.pack(previous)))
