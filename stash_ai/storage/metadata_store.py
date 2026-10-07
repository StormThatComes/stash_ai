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
            wanted = ','.join('(?,?)' for _ in pairs)
            metadata, thresholds = store.table('result_metadata'), store.table('result_thresholds')
            ctes, scopes = '', ()
            if store.preview_source:
                # SQLite otherwise materializes the entire UNION preview views
                # before the LEFT JOIN. Bound both derived sides to these keys.
                ctes = (', metadata AS (SELECT m.* FROM wanted AS w CROSS JOIN ' + metadata +
                        ' AS m WHERE m.scope=? AND m.scene=w.scene AND m.model=w.model)'
                        ', thresholds AS (SELECT t.* FROM wanted AS w CROSS JOIN ' + thresholds +
                        ' AS t WHERE t.scope=? AND t.scene=w.scene AND t.model=w.model)')
                metadata, thresholds, scopes = 'metadata', 'thresholds', (store.scope, store.scope)
            rows = store.read_rows(
                'WITH wanted(scene,model) AS (VALUES ' + wanted + ')' + ctes + ' '
                'SELECT r.scene,r.model,r.kind,m.format,m.data,t.threshold FROM wanted AS w CROSS JOIN ' +
                store.table('results') + ' AS r LEFT JOIN ' + metadata +
                ' AS m ON m.scope=r.scope AND m.scene=r.scene AND m.model=r.model AND m.kind=r.kind '
                'LEFT JOIN ' + thresholds +
                ' AS t ON t.scope=r.scope AND t.scene=r.scene AND t.model=r.model AND t.kind=r.kind '
                'WHERE r.scope=? AND r.scene=w.scene AND r.model=w.model',
                (*(value for pair in pairs for value in pair), *scopes, store.scope))
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
