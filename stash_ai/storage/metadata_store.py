"""Derived metadata persistence and repair after invalidation or corruption."""
from stash_ai.runtime.timing import measured
from stash_ai.storage.thresholds import overlay


class MetadataStore:
    def __init__(self, store):
        self.store = store

    @property
    def report(self):
        return self.store.report

    @measured('state_read')
    def get(self, scene, kind, model):
        """Validate small SQL records and repair invalidated derived rows."""
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
                self.save(scene, model, kind, segments)
        return overlay(segments, kind, threshold)

    def save(self, scene, model, kind, segments):
        from stash_ai.storage.metadata import FORMAT, encode
        store = self.store
        store.connection.execute('INSERT INTO result_metadata VALUES (?,?,?,?,?,?) '
            'ON CONFLICT(scope,scene,model,kind) DO UPDATE SET format=excluded.format,data=excluded.data',
            (store.scope, str(scene), model, kind, FORMAT, encode(segments)))
