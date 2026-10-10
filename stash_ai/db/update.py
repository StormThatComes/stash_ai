"""Explicit offline upgrade through the application's existing state migrations."""

from contextlib import closing
from pathlib import Path
import zlib

from stash_ai.storage.database import StateDatabase
from stash_ai.storage.codec import StateDecodeError
from stash_ai.storage.schema import APPLICATION_ID, LATEST_SCHEMA_VERSION, require_supported_version, shape_issues

from .common import connect_readonly, table_names


def update_database(path, *, dry_run=False, show_progress=False, progress=None, details=None):
    """Upgrade all scopes, preserving evidence and unfinished publication plans.

    Initializers are idempotent and may commit creation of missing feature tables.
    Only a fully initialized database receives the latest version marker, so a
    failed upgrade can be retried through the same migration path.
    Dry-run reports the target without running migrations or taking a writer lock.
    """
    from .progress import maintenance_progress
    with maintenance_progress('DB update', show=show_progress, callback=progress) as notify:
        return _update_database(path, dry_run=dry_run, progress=notify, details=details)


def _update_database(path, *, dry_run, progress, details):
    if details is None:
        details = {}
    details.clear()
    details['noop'] = False
    path = Path(path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"State database does not exist: {path}")
    with closing(connect_readonly(path)) as db:
        db.execute('BEGIN')
        application = db.execute('PRAGMA application_id').fetchone()[0]
        version = db.execute('PRAGMA user_version').fetchone()[0]
        if application != APPLICATION_ID or 'results' not in table_names(db):
            raise ValueError('Not a supported stash-taggers state DB.')
        require_supported_version(version)
        from .revisions import completed, metadata_complete
        if (version == LATEST_SCHEMA_VERSION and not shape_issues(db)
                and completed(db) and metadata_complete(db)):
            details['noop'] = True
            return version, LATEST_SCHEMA_VERSION
        if not dry_run:
            progress('SQLite quick_check', 0, 1)
            if db.execute('PRAGMA quick_check(1)').fetchone()[0] != 'ok':
                raise RuntimeError('SQLite quick_check failed; database update was not started.')
            progress('SQLite quick_check', 1, 1)
    if dry_run:
        return version, LATEST_SCHEMA_VERSION

    try:
        from .migrations import migrate
        with StateDatabase(path, migration=True) as database:
            version = database.version
            details.update(migrate(database, progress=progress))
    except (KeyError, TypeError, zlib.error, StateDecodeError) as exc:
        raise ValueError('Saved state could not be migrated; inspect the database with --db-check.') from exc
    return version, LATEST_SCHEMA_VERSION
