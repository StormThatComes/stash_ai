"""Explicit full evidence/derived-metadata repair without Stash or models."""
from pathlib import Path
from contextlib import closing
import sqlite3
import zlib

from stash_ai.storage.database import StateDatabase
from stash_ai.storage.schema import APPLICATION_ID, LATEST_SCHEMA_VERSION, shape_issues
from .common import connect_readonly
from .revisions import mark_completed


def repair_missing_metadata(db):
    from stash_ai.storage.codec import decode
    from stash_ai.storage.metadata import FORMAT, encode, summarize
    for scope, scene, model, kind, data in db.execute('''
            SELECT r.scope,r.scene,r.model,r.kind,r.data FROM results AS r
            LEFT JOIN result_metadata AS m USING(scope,scene,model,kind)
            WHERE m.format IS NULL OR m.format!=? OR NOT json_valid(m.data)''', (FORMAT,)):
        db.execute('INSERT OR REPLACE INTO result_metadata VALUES (?,?,?,?,?,?)',
                   (scope, scene, model, kind, FORMAT, encode(summarize(decode(data)))))


def _repair(db):
    if db.execute('PRAGMA quick_check(1)').fetchone()[0] != 'ok':
        raise RuntimeError('SQLite quick_check failed; repair was not started.')
    from .duplicate_aggregates import migrate
    with db:
        db.execute('BEGIN IMMEDIATE')
        try:
            result = migrate(db, metadata=True)
        except (KeyError, TypeError, zlib.error) as exc:
            raise ValueError('Saved evidence could not be repaired; inspect with --db-check.') from exc
        mark_completed(db)
    return result


def repair_database(path, *, dry_run=False):
    path = Path(path).expanduser().resolve()
    with closing(connect_readonly(path)) as source:
        source.execute('BEGIN')
        if source.execute('PRAGMA application_id').fetchone()[0] != APPLICATION_ID:
            raise ValueError('Not a supported stash-taggers state DB.')
        if source.execute('PRAGMA user_version').fetchone()[0] != LATEST_SCHEMA_VERSION or shape_issues(source):
            raise ValueError('Repair requires the current complete schema; run --db-update first.')
        if dry_run:
            # SQLite-managed disk-backed temporary copy; source stays read-only.
            with closing(sqlite3.connect('')) as preview:
                source.backup(preview)
                result = _repair(preview)
            return dict(result, dry_run=True)
    with StateDatabase(path) as database:
        result = _repair(database.connection)
    return dict(result, dry_run=False)
