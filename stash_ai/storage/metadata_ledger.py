"""Optional derived dirty-key ledger; old v18+ writers are tracked by SQL triggers."""
from stash_ai.storage.metadata import FORMAT

TABLE = 'metadata_dirty'
REVISION = 'metadata-dirty-ledger'
KEYS = ('scope', 'scene', 'model', 'kind')
DDL = 'CREATE TABLE metadata_dirty (scope TEXT NOT NULL, scene TEXT NOT NULL, model TEXT NOT NULL, kind TEXT NOT NULL, PRIMARY KEY(scope,scene,model,kind)) WITHOUT ROWID'


def _key(row):
    return ' AND '.join(f'{key}={row}.{key}' for key in KEYS)


def _mark(row):
    values = ','.join(f'{row}.{key}' for key in KEYS)
    return f'INSERT OR IGNORE INTO {TABLE} SELECT {values} WHERE EXISTS (SELECT 1 FROM results WHERE {_key(row)});'


def definitions():
    result = {}
    for event in ('INSERT', 'UPDATE', 'DELETE'):
        body = ''
        if event != 'INSERT':
            body += f'DELETE FROM {TABLE} WHERE {_key("OLD")};'
            if event == 'UPDATE':
                body += _mark('OLD')
        if event != 'DELETE':
            body += _mark('NEW')
        name = f'metadata_ledger_results_{event.lower()}'
        result[name] = f'CREATE TRIGGER {name} AFTER {event} ON results BEGIN {body} END'
        body = _mark('OLD') if event != 'INSERT' else ''
        if event != 'DELETE':
            body += _mark('NEW')
            body += (f'DELETE FROM {TABLE} WHERE {_key("NEW")} '
                     f'AND NEW.format={FORMAT} AND json_valid(NEW.data);')
        name = f'metadata_ledger_metadata_{event.lower()}'
        result[name] = f'CREATE TRIGGER {name} AFTER {event} ON result_metadata BEGIN {body} END'
    return result


def _normalized(sql):
    return ''.join(sql.lower().split()).rstrip(';')


def trusted(db):
    objects = dict(db.execute("SELECT name,sql FROM sqlite_master WHERE name=? OR name LIKE 'metadata_ledger_%'", (TABLE,)))
    if _normalized(objects.get(TABLE, '')) != _normalized(DDL):
        return False
    if any(_normalized(objects.get(name, '')) != _normalized(sql) for name, sql in definitions().items()):
        return False
    if not db.execute("SELECT 1 FROM sqlite_master WHERE name='data_revisions' AND type='table'").fetchone():
        return False
    return db.execute('SELECT revision FROM data_revisions WHERE name=?', (REVISION,)).fetchone() == (FORMAT,)


def initialize(db):
    """Seed once under a writer lock, then maintain completeness incrementally."""
    if trusted(db):
        return
    for name in definitions():
        db.execute(f'DROP TRIGGER IF EXISTS main.{name}')
    db.execute(f'DROP TABLE IF EXISTS main.{TABLE}')
    db.execute(DDL)
    db.execute('''INSERT INTO metadata_dirty SELECT r.scope,r.scene,r.model,r.kind
        FROM results AS r LEFT JOIN result_metadata AS m USING(scope,scene,model,kind)
        WHERE m.format IS NULL OR m.format!=? OR NOT json_valid(m.data)''', (FORMAT,))
    for sql in definitions().values():
        db.execute(sql)
    db.execute('CREATE TABLE IF NOT EXISTS data_revisions (name TEXT PRIMARY KEY, revision INTEGER NOT NULL)')
    db.execute('INSERT OR REPLACE INTO data_revisions VALUES (?,?)', (REVISION, FORMAT))
