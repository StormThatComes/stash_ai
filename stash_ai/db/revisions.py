"""Completed data conversions are independent of DDL schema versions."""
DATA_REVISION = 1
REVISION_NAME = 'canonical-evidence-and-metadata'


def completed(db):
    if not db.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='data_revisions'").fetchone():
        return False
    columns = [row[1] for row in db.execute('PRAGMA table_info(data_revisions)')]
    if columns != ['name', 'revision']:
        raise ValueError('Invalid maintenance revision ledger; restore the database.')
    return db.execute('SELECT revision FROM data_revisions WHERE name=?', (REVISION_NAME,)).fetchone() == (DATA_REVISION,)


def mark_completed(db):
    db.execute('CREATE TABLE IF NOT EXISTS data_revisions (name TEXT PRIMARY KEY, revision INTEGER NOT NULL)')
    db.execute('INSERT OR REPLACE INTO data_revisions VALUES (?,?)', (REVISION_NAME, DATA_REVISION))


def metadata_complete(db):
    from stash_ai.storage.metadata import FORMAT
    # Small metadata/key scan; never read/decompress the evidence BLOB column.
    return not db.execute('''SELECT 1 FROM results AS r LEFT JOIN result_metadata AS m
        USING(scope,scene,model,kind) WHERE m.format IS NULL OR m.format!=?
        OR NOT json_valid(m.data) LIMIT 1''', (FORMAT,)).fetchone()
