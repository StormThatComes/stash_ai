"""Offline migrations. Ordinary state readers never import this module."""
from types import SimpleNamespace

from stash_ai.storage.codec import decode
from stash_ai.storage.metadata import FORMAT, decode as decode_metadata, encode, summarize
from stash_ai.storage.progress import index_result
from stash_ai.storage.schema import LATEST_SCHEMA_VERSION, initialize_features


def migrate(database):
    db = database.connection
    state = SimpleNamespace(connection=db, scope='', preview_source=False)
    initialize_features(state)
    with db:
        if database.version < 2:
            for scope, scene, model, kind, payload in db.execute(
                    'SELECT scope,scene,model,kind,data FROM results'):
                index_result(state, scene, model, kind, decode(payload), scope=scope)
        db.execute('DROP TABLE IF EXISTS refinement_progress')
        for event in ('insert', 'update', 'delete'):
            db.execute(f'DROP TRIGGER IF EXISTS perf_legacy_imports_{event}')
        # Historical metadata is derived once here. Runtime repair remains
        # necessary for corruption and invalidation by direct SQLite writers.
        for scope, scene, model, kind, payload, version, data in db.execute(
                'SELECT r.scope,r.scene,r.model,r.kind,r.data,m.format,m.data '
                'FROM results AS r LEFT JOIN result_metadata AS m USING(scope,scene,model,kind)'):
            if version == FORMAT:
                try:
                    decode_metadata(data)
                    continue
                except (TypeError, ValueError):
                    pass
            db.execute('INSERT OR REPLACE INTO result_metadata VALUES (?,?,?,?,?,?)',
                       (scope, scene, model, kind, FORMAT, encode(summarize(decode(payload)))))
        db.execute(f'PRAGMA user_version={LATEST_SCHEMA_VERSION}')
