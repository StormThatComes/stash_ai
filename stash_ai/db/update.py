"""Explicit offline upgrade through the application's existing state migrations."""

from contextlib import closing
from pathlib import Path
import zlib

from stash_ai.storage.store import StateStore
from stash_ai.storage.database import APPLICATION_ID, LATEST_SCHEMA_VERSION

from .common import connect_readonly, table_names


def update_database(path, *, dry_run=False):
    """Upgrade all scopes, preserving evidence and unfinished publication plans.

    Initializers are idempotent and may commit intermediate schema migrations.
    Only a fully initialized database receives the latest version marker, so a
    failed upgrade can be retried through the same migration path.
    Dry-run reports the target without running migrations or taking a writer lock.
    """
    path = Path(path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"State database does not exist: {path}")
    with closing(connect_readonly(path)) as db:
        application = db.execute('PRAGMA application_id').fetchone()[0]
        version = db.execute('PRAGMA user_version').fetchone()[0]
        if (application != APPLICATION_ID or version not in range(LATEST_SCHEMA_VERSION + 1)
                or 'results' not in table_names(db)):
            raise ValueError('Not a supported stash-taggers state DB.')
        if db.execute('PRAGMA quick_check(1)').fetchone()[0] != 'ok':
            raise RuntimeError('SQLite quick_check failed; database update was not started.')
    if dry_run:
        return version, LATEST_SCHEMA_VERSION

    # StateStore owns the same writer lock as tagging/cleanup and initializes
    # every feature schema. Its old result-index migration visits all scopes;
    # no current Stash URL or scope-specific policy is needed for maintenance.
    try:
        with StateStore(path, '', allow_cleanup=True) as state:
            version = state._database.version
            with state.connection:
                if state.connection.execute('PRAGMA user_version').fetchone()[0] < LATEST_SCHEMA_VERSION:
                    state.connection.execute(f'PRAGMA user_version={LATEST_SCHEMA_VERSION}')
    except (KeyError, TypeError, zlib.error) as exc:
        raise ValueError('Saved state could not be migrated; inspect the database with --db-check.') from exc
    return version, LATEST_SCHEMA_VERSION
