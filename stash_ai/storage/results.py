"""Inference result persistence behind StateStore."""
from __future__ import annotations

from copy import deepcopy

from stash_ai.runtime.timing import measured
from stash_ai.storage.thresholds import overlay
from stash_ai.models import SOURCES
from .selection import selected_rows


KINDS = ("scene", "markers")


class ResultStore:
    """Persist per-model inference results while StateStore owns DB/cache lifecycle."""

    def __init__(self, store):
        self.store = store

    @property
    def report(self):
        return self.store.report

    @staticmethod
    def _normalize_record(record, kind, threshold):
        from stash_ai.storage.segments import StoredSegment
        from stash_ai.evidence.formats import validate_current
        validate_current(record)
        if record.get('mode') != ('scene' if kind == 'scene' else 'frames') or not isinstance(record.get('items'), list):
            raise ValueError('Invalid active result record; run --db-check.')
        return StoredSegment(overlay(record, kind, threshold))

    @measured('state_read')
    def records(self, scene, kind, model=None, *, borrow=False):
        """One record per source; public reads are detached."""
        store = self.store
        key = ('records', str(scene), kind)
        records = store.cached(key)
        if records is None:
            records = {model: self._normalize_record(store.unpack(data), kind, threshold)
                for model, data, threshold in store.read_rows(
                "SELECT r.model,r.data,t.threshold FROM " + store.table('results') + " AS r LEFT JOIN " +
                store.table('result_thresholds') + " AS t USING(scope,scene,model,kind) "
                "WHERE r.scope=? AND r.scene=? AND r.kind=?", (store.scope, str(scene), kind))}
            store.retain(key, records)
        result = {model: records[model]} if model in records else {} if model else records
        return result if borrow else deepcopy(result)

    @measured('state_read')
    def model_record(self, scene, kind, model, *, borrow=False):
        store = self.store
        cached = store.cached(('records', str(scene), kind))
        if cached is not None:
            record = cached.get(model)
        else:
            rows = store.read_rows(
                "SELECT r.data,t.threshold FROM " + store.table('results') + " AS r LEFT JOIN " +
                store.table('result_thresholds') + " AS t USING(scope,scene,model,kind) "
                "WHERE r.scope=? AND r.scene=? AND r.model=? AND r.kind=?", (store.scope, str(scene), model, kind))
            record = self._normalize_record(store.unpack(rows[0][0]), kind, rows[0][1]) if rows else None
        return record if borrow else deepcopy(record)

    def result_presence(self, models, scenes=None):
        """Return lightweight saved-result keys without decoding payload BLOBs."""
        store = self.store
        from stash_ai.storage.preview import check_source
        check_source(store)
        selected = tuple(dict.fromkeys(models))
        if not selected:
            return set()
        placeholders = ','.join('?' for _ in selected)
        rows = selected_rows(store,
            "SELECT scene,model,kind FROM " + store.table('results') +
            f" WHERE scope=? AND model IN ({placeholders})",
            (store.scope, *selected), scenes)
        return {(str(scene), model, kind) for scene, model, kind in rows}

    def result_index_presence(self, models, scenes=None):
        """Return indexed result keys without decoding result payloads."""
        store = self.store
        from stash_ai.storage.preview import check_source
        check_source(store)
        selected = tuple(dict.fromkeys(models))
        if not selected:
            return set()
        placeholders = ','.join('?' for _ in selected)
        rows = selected_rows(store,
            "SELECT scene,model,kind FROM " + store.table('result_index') +
            f" WHERE scope=? AND model IN ({placeholders})",
            (store.scope, *selected), scenes)
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

    def preload_sources(self, scenes=None):
        store = self.store
        if not store.cache.limit:
            return
        last, values = None, set()
        for scene, kind, model in selected_rows(store,
                'SELECT scene,kind,model FROM ' + store.table('results') + ' WHERE scope=?',
                (store.scope,), scenes, suffix=' ORDER BY scene,kind'):
            key = ('sources', scene, kind)
            if key != last:
                if last is not None:
                    store.cache.put(last, values)
                last, values = key, set()
            values.add(model)
        if last is not None:
            store.cache.put(last, values)

    @measured('state_write')
    def put(self, scene, model, outputs):
        with self.store.connection:
            self.put_in_transaction(scene, model, outputs)

    def put_in_transaction(self, scene, model, outputs):
        """Replace inference outputs inside the caller's transaction."""
        if model not in SOURCES or set(outputs) - set(KINDS):
            raise ValueError("Unknown result source or output kind.")
        for kind, segment in outputs.items():
            self.replace_record(scene, model, kind, segment)

    def replace_record(self, scene, model, kind, record):
        """Replace the active record while retaining exactly pinned prior proofs."""
        if (model not in SOURCES or kind not in KINDS or not isinstance(record, dict)
                or record.get('mode') != ('scene' if kind == 'scene' else 'frames')
                or not isinstance(record.get('meta'), dict) or not isinstance(record.get('items'), list)):
            raise ValueError('Incomplete result record: mode, metadata and items are required.')
        from stash_ai.evidence.formats import validate_current
        validate_current(record)
        if 'refinement' in record:
            raise ValueError('Retired refinement evidence is not supported; run --db-update for saved results.')
        store = self.store
        from stash_ai.storage.archive import retain
        previous = self.model_record(scene, kind, model, borrow=True)
        retain(store, scene, model, kind, previous, record)
        from stash_ai.storage.preview import delete
        from stash_ai.storage.progress import index_result
        store.cache.discard(('records', str(scene), kind))
        store.cache.discard(('sources', str(scene), kind))
        store.cache.discard(('status', str(scene), model))
        from stash_ai.storage.thresholds import separate
        raw, threshold = separate(record, kind)
        from stash_ai.storage.metadata import summarize
        metadata = summarize(raw)
        # New inference resets the effective publication policy.
        delete(store, 'result_thresholds', scene, model, kind)
        delete(store, 'result_metadata', scene, model, kind)
        store.connection.execute('DELETE FROM result_thresholds WHERE scope=? AND scene=? AND model=? AND kind=?',
            (store.scope, str(scene), model, kind))
        store.connection.execute("INSERT INTO results VALUES (?,?,?,?,?) "
            "ON CONFLICT(scope,scene,model,kind) DO UPDATE SET data=excluded.data",
            (store.scope, str(scene), model, kind, store.pack(raw)))
        if threshold is not None:
            store.connection.execute('INSERT INTO result_thresholds VALUES (?,?,?,?,?)',
                                     (store.scope, str(scene), model, kind, threshold))
        store.metadata.save(scene, model, kind, metadata)
        index_result(store, scene, model, kind, record)

    @measured('state_write')
    def set_threshold(self, scene, model, kind, threshold, record, *, metadata_only=False):
        """Update policy and its completion index inside the caller's transaction.

        The record is a validated effective view or its small metadata when
        metadata_only is true.
        The evidence BLOB is never encoded or written by this path.
        """
        from stash_ai.storage.progress import index_result
        from stash_ai.evidence.thresholds import _score
        if model not in SOURCES or kind not in KINDS or not record or not _score(threshold):
            raise ValueError('Invalid result threshold.')
        store = self.store
        store.cache.discard(('records', str(scene), kind))
        store.cache.discard(('status', str(scene), model))
        if not store.connection.in_transaction:
            store.connection.execute('BEGIN')
        store.connection.execute('INSERT INTO result_thresholds VALUES (?,?,?,?,?) '
            'ON CONFLICT(scope,scene,model,kind) DO UPDATE SET threshold=excluded.threshold',
            (store.scope, str(scene), model, kind, threshold))
        index_result(store, scene, model, kind, record, metadata_only=metadata_only)
