"""Private derived state over one immutable, indexed disk evidence snapshot."""
from pathlib import Path

# These tables never change during a trial. Keep ID/scan skeletons for SQLite
# foreign keys and dependency triggers; queries see the original indexed rows.
SHARED = ('observations', 'person_evidence', 'body_observations', 'evidence_thumbnails')


def prepare(target, seed):
    path = seed.execute('PRAGMA database_list').fetchone()[2]
    if not path:
        # Standalone callers may supply a RAM seed, which cannot be attached.
        seed.backup(target)
        return
    from .database import TABLES
    from .profiling import count
    uri = Path(path).resolve().as_uri() + '?mode=ro&immutable=1'
    target.execute('ATTACH DATABASE ? AS retained', (uri,))
    target.execute('PRAGMA retained.cache_size=-2048')
    target.execute('PRAGMA retained.mmap_size=0')
    target.execute('PRAGMA foreign_keys=OFF')
    triggers = list(target.execute("SELECT name,sql FROM main.sqlite_master WHERE type='trigger'"))
    target.execute('SAVEPOINT trial_prepare')
    try:
        for name, _ in triggers:
            target.execute('DROP TRIGGER main."'+name.replace('"','""')+'"')
        # Bulk copies retain row order and all mutable history, but no vectors
        # or images. A trial cannot write through the read-only evidence views.
        for table in TABLES:
            if table in ('person_evidence', 'evidence_thumbnails', 'evidence_versions', 'evidence_fingerprints'):
                continue
            columns = [r[1] for r in target.execute(f'PRAGMA table_info({table})')]
            minimal = dict(embedding='NULL', crop='NULL', geometry="'[]'", quality="'{}'", source="''", rect="'[]'")
            fields = ','.join(minimal.get(c, c) if table in SHARED else c for c in columns)
            target.execute(f'INSERT INTO main.{table}(rowid,{",".join(columns)}) '
                           f'SELECT rowid,{fields} FROM retained.{table} ORDER BY rowid')
        for table in ('evidence_versions', 'evidence_fingerprints'):
            target.execute(f'DELETE FROM main.{table}')
            target.execute(f'INSERT INTO main.{table} SELECT * FROM retained.{table}')
        for _, sql in triggers:
            target.execute(sql)
        target.execute('RELEASE trial_prepare')
        target.commit()
    except BaseException:
        target.execute('ROLLBACK TO trial_prepare')
        target.execute('RELEASE trial_prepare')
        raise
    finally:
        target.execute('PRAGMA foreign_keys=ON')
    for table in SHARED:
        target.execute(f'CREATE TEMP VIEW {table} AS SELECT rowid AS rowid,* FROM retained.{table}')
    # Statistics belong to the immutable seed; never rescan vectors per trial.
    row = seed.execute("SELECT value FROM metadata WHERE key='replay-input-statistics'").fetchone()
    if row:
        import json
        count('trial_shared_input_bytes', json.loads(row[0])['embedding_bytes'])
    count('trial_private_pages', target.execute('PRAGMA main.page_count').fetchone()[0])
