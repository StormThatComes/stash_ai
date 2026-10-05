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
        return self.store.report

    @staticmethod
    def _normalize_segments(segments, kind, threshold):
        """Apply separate publication policy to canonical evidence records."""
        from stash_ai.storage.segments import StoredSegment
        overlay(segments, kind, threshold)
        from stash_ai.evidence.formats import validate_current
        for segment in segments:
            validate_current(segment)
        for index, segment in enumerate(segments):
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
                "SELECT r.model,r.data,t.thresholds FROM " + store.table('results') + " AS r LEFT JOIN " +
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
            "SELECT r.data,t.thresholds FROM " + store.table('results') + " AS r LEFT JOIN " +
            store.table('result_thresholds') + " AS t USING(scope,scene,model,kind) "
            "WHERE r.scope=? AND r.scene=? AND r.model=? AND r.kind=?",
            (store.scope, str(scene), model, kind))
        segments = self._normalize_segments(store.unpack(rows[0][0]), kind, rows[0][1]) if rows else []
        return segments if borrow else deepcopy(segments)

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
    def put(self, scene, model, outputs):
        with self.store.connection:
            self.put_in_transaction(scene, model, outputs)

    def put_in_transaction(self, scene, model, outputs):
        """Replace inference outputs inside the caller's transaction."""
        if model not in SOURCES or set(outputs) - set(KINDS):
            raise ValueError("Unknown result source or output kind.")
        for kind, segment in outputs.items():
            self.replace_records(scene, model, kind, [segment])

    def append_legacy_in_transaction(self, scene, model, kind, segment):
        """Append an explicit legacy import, deduplicating capture timestamps."""
        if segment.get('meta', {}).get('origin') != 'legacy':
            raise ValueError('Only imported legacy evidence may be appended.')
        def semantic(value):
            data = dict(value)
            data['meta'] = {key: item for key, item in data['meta'].items()
                            if key != 'captured_at'}
            return json.dumps(data, sort_keys=True, allow_nan=False)
        current = semantic(segment)
        old = self.model_records(scene, kind, model, borrow=True)
        keep = [previous for previous in old if semantic(previous) != current]
        self.replace_records(scene, model, kind, keep + [segment])

    def replace_records(self, scene, model, kind, segments):
        """Replace a segment list inside the caller's transaction, including previews."""
        modes = ('scene',) if kind == 'scene' else ('frames', 'spans')
        if (model not in SOURCES or kind not in KINDS or not isinstance(segments, list)
                or any(not isinstance(segment, dict) or segment.get('mode') not in modes
                       or not isinstance(segment.get('meta'), dict)
                       or not isinstance(segment.get('items'), list) for segment in segments)):
            raise ValueError('Incomplete result record: mode, metadata and items are required.')
        from stash_ai.evidence.formats import validate_current
        for segment in segments:
            validate_current(segment)
        if any('refinement' in segment for segment in segments):
            raise ValueError('Retired refinement evidence is not supported; run --db-update for saved results.')
        store = self.store
        from stash_ai.storage.preview import delete
        from stash_ai.storage.progress import index_result
        store.cache.discard(('records', str(scene), kind))
        store.cache.discard(('sources', str(scene), kind))
        store.cache.discard(('status', str(scene), model))
        from stash_ai.storage.thresholds import separate
        raw, thresholds = separate(segments, kind)
        from stash_ai.storage.metadata import summarize
        metadata = summarize(raw)
        # History keeps its own effective thresholds. A new segment must not
        # inherit the previous result's policy.
        delete(store, 'result_thresholds', scene, model, kind)
        delete(store, 'result_metadata', scene, model, kind)
        store.connection.execute('DELETE FROM result_thresholds WHERE scope=? AND scene=? AND model=? AND kind=?',
            (store.scope, str(scene), model, kind))
        store.connection.execute("INSERT INTO results VALUES (?,?,?,?,?) "
            "ON CONFLICT(scope,scene,model,kind) DO UPDATE SET data=excluded.data",
            (store.scope, str(scene), model, kind, store.pack(raw)))
        if thresholds is not None:
            store.connection.execute('INSERT INTO result_thresholds VALUES (?,?,?,?,?)',
                                     (store.scope, str(scene), model, kind, thresholds))
        store.metadata.save(scene, model, kind, metadata)
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
        from stash_ai.storage.thresholds import encode
        store.connection.execute('INSERT INTO result_thresholds VALUES (?,?,?,?,?) '
            'ON CONFLICT(scope,scene,model,kind) DO UPDATE SET thresholds=excluded.thresholds',
            (store.scope, str(scene), model, kind, encode([threshold] * len(segments))))
        index_result(store, scene, model, kind, segments, metadata_only=metadata_only)
