"""Offline upgrades from schema v18 and repair of current derived metadata."""
from types import SimpleNamespace

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
        from .revisions import completed, metadata_complete, mark_completed
        if database.version < 25 or not completed(db):
            from .duplicate_aggregates import migrate as repair_evidence
            repair_evidence(db, metadata=True)
            mark_completed(db)
        elif not metadata_complete(db):
            from .repair import repair_missing_metadata
            repair_missing_metadata(db)
        db.execute(f'PRAGMA user_version={LATEST_SCHEMA_VERSION}')
