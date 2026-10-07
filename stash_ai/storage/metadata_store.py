"""Derived metadata persistence and repair after invalidation or corruption."""
from stash_ai.runtime.timing import measured
from stash_ai.storage.thresholds import overlay
from contextlib import contextmanager


class MetadataStore:
    def __init__(self, store):
        self.store = store
        self._batch = None

    @contextmanager
    def batch(self, pairs):
        """Fetch only one bounded group; all required output kinds share this read."""
        from stash_ai.storage.preview import check_source
        store = self.store
        check_source(store)
        pairs = tuple(dict.fromkeys((str(scene), model) for scene, model in pairs))
        if len(pairs) > 128:
            raise ValueError('Metadata batches are limited to 128 scene/model pairs.')
        previous = self._batch
        values = {}
        if pairs:
            condition = ' OR '.join('(r.scene=? AND r.model=?)' for _ in pairs)
            rows = store.read_rows(
                'SELECT r.scene,r.model,r.kind,m.format,m.data,t.threshold FROM ' + store.table('results') +
                ' AS r LEFT JOIN ' + store.table('result_metadata') + ' AS m USING(scope,scene,model,kind) '
                'LEFT JOIN ' + store.table('result_thresholds') + ' AS t USING(scope,scene,model,kind) '
                'WHERE r.scope=? AND (' + condition + ')',
                (store.scope, *(value for pair in pairs for value in pair)))
            values = {(scene, model, kind): (version, data, threshold)
                      for scene, model, kind, version, data, threshold in rows}
        self._batch = values
        try:
            yield
        finally:
            self._batch = previous

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
        if self._batch is not None:
            key = str(scene), model, kind
            rows = [self._batch[key]] if key in self._batch else []
        else:
            rows = store.read_rows(
                'SELECT m.format,m.data,t.threshold FROM ' + store.table('results') + ' AS r LEFT JOIN ' +
                store.table('result_metadata') + ' AS m USING(scope,scene,model,kind) LEFT JOIN ' +
                store.table('result_thresholds') + ' AS t USING(scope,scene,model,kind) '
                'WHERE r.scope=? AND r.scene=? AND r.model=? AND r.kind=?', params)
        if not rows:
            return None
        version, data, threshold = rows[0]
        record = None
        if version == FORMAT:
            try:
                record = decode(data)
            except (TypeError, ValueError):
                pass  # A damaged derived row is rebuilt from the authoritative evidence.
        if record is None:
            payload = store.read_rows('SELECT data FROM ' + store.table('results') +
                ' WHERE scope=? AND scene=? AND model=? AND kind=?', params)[0][0]
            record = summarize(store.unpack(payload))
            with backfill_batch(store):
                self.save(scene, model, kind, record)
        return overlay(record, kind, threshold)

    def save(self, scene, model, kind, record):
        from stash_ai.storage.metadata import FORMAT, encode
        store = self.store
        store.connection.execute('INSERT INTO result_metadata VALUES (?,?,?,?,?,?) '
            'ON CONFLICT(scope,scene,model,kind) DO UPDATE SET format=excluded.format,data=excluded.data',
            (store.scope, str(scene), model, kind, FORMAT, encode(record)))
