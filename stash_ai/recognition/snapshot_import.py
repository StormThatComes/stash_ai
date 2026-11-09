"""Bulk import into a freshly created, private snapshot, without row triggers."""
from contextlib import contextmanager


@contextmanager
def bulk(target):
    # Only the trusted destination schema is used. Source DDL is never executed.
    triggers = list(target.execute("SELECT name,sql FROM main.sqlite_master WHERE type='trigger'"))
    target.execute('SAVEPOINT snapshot_import')
    try:
        for name, _ in triggers:
            target.execute('DROP TRIGGER main."'+name.replace('"', '""')+'"')
        yield
        if target.execute("SELECT 1 FROM sqlite_master WHERE name='scene_owners'").fetchone():
            from .scene_owner_index import populate
            populate(target)
        if target.execute("SELECT 1 FROM sqlite_master WHERE name='body_alias_scenes'").fetchone():
            from .alias_scene_index import populate as alias_scenes
            alias_scenes(target)
        for name, sql in triggers:
            # A replay may shadow observations with a TEMP vector-source view.
            # Bind restored triggers to the trusted main schema explicitly.
            target.execute(sql.replace('CREATE TRIGGER '+name,
                'CREATE TRIGGER main.'+name, 1))
        # Fresh private generations have no inherited fingerprint consumers.
        target.execute('DELETE FROM evidence_fingerprints')
        target.execute('RELEASE snapshot_import')
    except BaseException:
        if target.in_transaction:
            target.execute('ROLLBACK TO snapshot_import')
            target.execute('RELEASE snapshot_import')
        raise
