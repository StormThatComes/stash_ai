"""Offline upgrades from schema v18 and repair of current derived metadata."""
from types import SimpleNamespace

from stash_ai.storage.schema import LATEST_SCHEMA_VERSION, initialize_features


def migrate(database, *, progress=None):
    def phase(name, completed):
        if progress is not None:
            progress(name, completed, 1)
    result = dict(required_repair=False, historical_conversions=False, metadata_repaired=0)
    db = database.connection
    state = SimpleNamespace(connection=db, scope='', preview_source=False)
    phase('schema features', 0)
    initialize_features(state)
    phase('schema features', 1)
    with db:
        db.execute('BEGIN')
        if 'inference' in {row[1] for row in db.execute('PRAGMA table_info(result_index)')}:
            db.execute('ALTER TABLE result_index DROP COLUMN inference')
        retired = {}
        if database.version < 23:
            phase('historical evidence', 0)
            from .current_evidence_migration import migrate as retire_obsolete
            retired = retire_obsolete(db)
            phase('historical evidence', 1)
        if database.version < 25:
            result['historical_conversions'] = True
            phase('historical publications', 0)
            from .scalar_evidence_migration import migrate as migrate_publications
            migrate_publications(db, retired)
            phase('historical publications', 1)
        from .revisions import completed, metadata_complete, mark_completed
        if database.version < 25 or not completed(db):
            from .duplicate_aggregates import migrate as repair_evidence
            result['required_repair'] = True
            result.update(repair_evidence(db, metadata=True, progress=progress))
            mark_completed(db)
        elif not metadata_complete(db):
            from .repair import repair_missing_metadata
            result['metadata_repaired'] = repair_missing_metadata(db, progress=progress)
        db.execute(f'PRAGMA user_version={LATEST_SCHEMA_VERSION}')
    phase('committed', 1)
    return result
