"""Offline upgrades from schema v18 and repair of current derived metadata."""
from types import SimpleNamespace

from stash_ai.storage.codec import decode
from stash_ai.storage.schema import LATEST_SCHEMA_VERSION, initialize_features


def migrate(database):
    db = database.connection
    state = SimpleNamespace(connection=db, scope='', preview_source=False)
    initialize_features(state)
    with db:
        db.execute('BEGIN')
        if 'inference' in {row[1] for row in db.execute('PRAGMA table_info(result_index)')}:
            db.execute('ALTER TABLE result_index DROP COLUMN inference')
        retired = {}
        if database.version < 23:
            from .current_evidence_migration import migrate as retire_obsolete
            retired = retire_obsolete(db)
        if database.version < 25:
            from .scalar_evidence_migration import migrate as migrate_publications
            migrate_publications(db, retired)
        from .duplicate_aggregates import migrate as deduplicate_aggregates
        deduplicate_aggregates(db)
        from stash_ai.storage.metadata import FORMAT, decode as decode_metadata, encode, summarize
        for scope, scene, model, kind, payload, version, data in db.execute(
                'SELECT r.scope,r.scene,r.model,r.kind,r.data,m.format,m.data '
                'FROM results AS r LEFT JOIN result_metadata AS m USING(scope,scene,model,kind)'):
            derived = encode(summarize(decode(payload)))
            if version == FORMAT:
                try:
                    if encode(decode_metadata(data)) == derived:
                        continue
                except (TypeError, ValueError):
                    pass
            db.execute('INSERT OR REPLACE INTO result_metadata VALUES (?,?,?,?,?,?)',
                       (scope, scene, model, kind, FORMAT, derived))
        db.execute(f'PRAGMA user_version={LATEST_SCHEMA_VERSION}')
