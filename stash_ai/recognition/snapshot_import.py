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
        for _, sql in triggers:
            target.execute(sql)
        # Fresh private generations have no inherited fingerprint consumers.
        target.execute('DELETE FROM evidence_fingerprints')
        target.execute('RELEASE snapshot_import')
    except BaseException:
        if target.in_transaction:
            target.execute('ROLLBACK TO snapshot_import')
            target.execute('RELEASE snapshot_import')
        raise
