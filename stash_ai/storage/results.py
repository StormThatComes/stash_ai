"""Inference result and legacy-evidence persistence behind StateStore."""
from __future__ import annotations

from copy import deepcopy
import json

from stash_ai.runtime.timing import measured
from stash_ai.storage.thresholds import overlay
from stash_ai.models import SOURCES


KINDS = ("scene", "markers")


class ResultStore:
    """Persist per-model inference results while StateStore owns DB/cache lifecycle."""

    def __init__(self, store):
        self.store = store

    @property
    def report(self):
        return getattr(self.store, 'report', None)

    @staticmethod
    def _normalize_segments(segments, kind, threshold):
        """Apply the lazy compatibility view used by all decoded result reads."""
        from stash_ai.performance import StoredSegment
        overlay(segments, kind, threshold)
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
            records = {model: self._normalize_segments(store.unpack(data), kind, threshold)
                for model, data, threshold in store.read_rows(
                "SELECT r.model,r.data,t.threshold FROM " + store.table('results') + " AS r LEFT JOIN " +
                store.table('result_thresholds') + " AS t USING(scope,scene,model,kind) "
                "WHERE r.scope=? AND r.scene=? AND r.kind=?",
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
            "SELECT r.data,t.threshold FROM " + store.table('results') + " AS r LEFT JOIN " +
            store.table('result_thresholds') + " AS t USING(scope,scene,model,kind) "
            "WHERE r.scope=? AND r.scene=? AND r.model=? AND r.kind=?",
            (store.scope, str(scene), model, kind))
        segments = self._normalize_segments(store.unpack(rows[0][0]), kind, rows[0][1]) if rows else []
        return segments if borrow else deepcopy(segments)

    @measured('state_read')
    def metadata(self, scene, kind, model):
        """Validate from small SQL records; lazily backfill older result payloads."""
        from stash_ai.storage.preview import check_source
        from stash_ai.storage.metadata import FORMAT, backfill_batch, decode, summarize
        store = self.store
        check_source(store)
        params = (store.scope, str(scene), model, kind)
        rows = store.read_rows(
            'SELECT m.format,m.data,t.threshold FROM ' + store.table('results') + ' AS r LEFT JOIN ' +
            store.table('result_metadata') + ' AS m USING(scope,scene,model,kind) LEFT JOIN ' +
            store.table('result_thresholds') + ' AS t USING(scope,scene,model,kind) '
            'WHERE r.scope=? AND r.scene=? AND r.model=? AND r.kind=?', params)
        if not rows:
            return []
        version, data, threshold = rows[0]
        segments = None
        if version == FORMAT:
            try:
                segments = decode(data)
            except (TypeError, ValueError):
                pass  # A damaged derived row is rebuilt from the authoritative evidence.
        if segments is None:
            payload = store.read_rows('SELECT data FROM ' + store.table('results') +
                ' WHERE scope=? AND scene=? AND model=? AND kind=?', params)[0][0]
            segments = summarize(store.unpack(payload))
            with backfill_batch(store):
                self._save_metadata(scene, model, kind, segments)
        return overlay(segments, kind, threshold)

    def _save_metadata(self, scene, model, kind, segments):
        from stash_ai.storage.metadata import FORMAT, encode
        store = self.store
        store.connection.execute('INSERT INTO result_metadata VALUES (?,?,?,?,?,?) '
            'ON CONFLICT(scope,scene,model,kind) DO UPDATE SET format=excluded.format,data=excluded.data',
            (store.scope, str(scene), model, kind, FORMAT, encode(segments)))

    def result_presence(self, models):
        """Return lightweight saved-result keys without decoding payload BLOBs."""
        store = self.store
        from stash_ai.storage.preview import check_source
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
        from stash_ai.storage.preview import check_source
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
            self.put_in_transaction(scene, model, outputs, replace)

    def put_in_transaction(self, scene, model, outputs, replace=None):
        """Write outputs while the caller owns the commit or rollback boundary."""
        if model not in SOURCES or set(outputs) - set(KINDS):
            raise ValueError("Unknown result source or output kind.")
        for kind, segment in outputs.items():
            segment = {key: value for key, value in segment.items() if key != "refinement"}
            old = (self.records(scene, kind, model, borrow=True).get(model, [])
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
            self.replace_records(scene, model, kind, keep + [segment])

    def replace_records(self, scene, model, kind, segments):
        """Replace a segment list inside the caller's transaction, including previews."""
        store = self.store
        from stash_ai.storage.preview import delete
        from stash_ai.storage.progress import index_result
        store.cache.discard(('records', str(scene), kind))
        store.cache.discard(('sources', str(scene), kind))
        store.cache.discard(('status', str(scene), model))
        # Appended history already carries its effective inline view. A new
        # inference must not inherit the previous result's table-wide threshold.
        delete(store, 'result_thresholds', scene, model, kind)
        delete(store, 'result_metadata', scene, model, kind)
        store.connection.execute('DELETE FROM result_thresholds WHERE scope=? AND scene=? AND model=? AND kind=?',
            (store.scope, str(scene), model, kind))
        store.connection.execute("INSERT INTO results VALUES (?,?,?,?,?) "
            "ON CONFLICT(scope,scene,model,kind) DO UPDATE SET data=excluded.data",
            (store.scope, str(scene), model, kind, store.pack(segments)))
        from stash_ai.storage.metadata import summarize
        try:
            metadata = summarize(segments)
        except (KeyError, TypeError):
            # Low-level replacement also accepts incomplete state records.
            # Leave those uncached rather than adding a new write-time constraint.
            metadata = None
        if metadata is not None:
            self._save_metadata(scene, model, kind, metadata)
        index_result(store, scene, model, kind, segments)

    @measured('state_write')
    def set_threshold(self, scene, model, kind, threshold, segments, *, metadata_only=False):
        """Update policy and its completion index inside the caller's transaction.

        Segments are the validated effective view from the threshold planner,
        or its small metadata records when metadata_only is true.
        The evidence BLOB is never encoded or written by this path.
        """
        from stash_ai.storage.progress import index_result
        from stash_ai.evidence.thresholds import _score
        if model not in SOURCES or kind not in KINDS or not segments or not _score(threshold):
            raise ValueError('Invalid result threshold.')
        store = self.store
        store.cache.discard(('records', str(scene), kind))
        store.cache.discard(('status', str(scene), model))
        if not store.connection.in_transaction:
            store.connection.execute('BEGIN')
        # Older executables must reject databases whose effective policy lives
        # outside results.data. Merely opening an old DB need not upgrade it.
        if store.connection.execute('PRAGMA user_version').fetchone()[0] < 8:
            store.connection.execute('PRAGMA user_version=8')
        store.connection.execute('INSERT INTO result_thresholds VALUES (?,?,?,?,?) '
            'ON CONFLICT(scope,scene,model,kind) DO UPDATE SET threshold=excluded.threshold',
            (store.scope, str(scene), model, kind, threshold))
        index_result(store, scene, model, kind, segments, metadata_only=metadata_only)

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
