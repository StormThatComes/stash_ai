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
    # An outer UPSERT can override a trigger's OR IGNORE policy. Avoid the
    # conflicting insert explicitly; SQLite's writer transaction serializes it.
    return (f'INSERT INTO {TABLE} SELECT {values} WHERE EXISTS (SELECT 1 FROM results WHERE {_key(row)}) '
            f'AND NOT EXISTS (SELECT 1 FROM {TABLE} WHERE {_key(row)});')



def _valid_metadata_sql(row):
    """The small metadata contract, including required nested provenance."""
    data = f'{row}.data'
    from stash_ai.storage.metadata import META_KEYS
    allowed = ','.join("'" + key + "'" for key in sorted(META_KEYS))
    from stash_ai.evidence.formats import ANALYSIS_FORMAT, REQUIRED_SCORE_FIELDS
    def score(key):
        path = '$.meta.' + key
        return (f"json_type({data},'{path}') IN ('integer','real') "
                f"AND json_extract({data},'{path}') BETWEEN 0 AND 1")
    scores = ' AND '.join(score(key) for key in REQUIRED_SCORE_FIELDS)
    signatures = ' AND '.join(f"json_type({data},'$.meta.{key}')='text' "
                             f"AND length(json_extract({data},'$.meta.{key}'))>0"
                             for key in ('inference_signature', 'frame_input_signature'))
    return (f"COALESCE((CASE WHEN json_valid({data}) THEN "
            f"json_type({data})='object' "
            f"AND json_type({data},'$.mode')='text' "
            f"AND json_extract({data},'$.mode') IN ('scene','frames') "
            f"AND json_type({data},'$.meta')='object' "
            f"AND json_type({data},'$.has_frames') IN ('true','false') "
            f"AND json_type({data},'$.populated') IN ('true','false') "
            f"AND (SELECT COUNT(*) FROM json_each({data}))=4 "
            f"AND NOT EXISTS (SELECT 1 FROM json_each({data},'$.meta') "
            f"WHERE key NOT IN ({allowed})) "
            f"AND (SELECT COUNT(*)=COUNT(DISTINCT key) FROM json_each({data},'$.meta')) "
            f"AND json_extract({data},'$.meta.origin')='inference' "
            f"AND json_type({data},'$.meta.analysis_format')='integer' "
            f"AND json_extract({data},'$.meta.analysis_format')={ANALYSIS_FORMAT} "
            f"AND {signatures} AND {scores} "
            f"AND (json_extract({data},'$.mode')!='scene' OR "
            f"(json_type({data},'$.has_frames')='true' AND {score('scene_score_floor')})) "
            f"ELSE 0 END),0)")


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
                     f'AND NEW.format={FORMAT} AND {_valid_metadata_sql("NEW")};')
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
    db.execute(f'''INSERT INTO metadata_dirty SELECT r.scope,r.scene,r.model,r.kind
        FROM results AS r LEFT JOIN result_metadata AS m USING(scope,scene,model,kind)
        WHERE m.format IS NULL OR m.format!=? OR NOT {_valid_metadata_sql('m')}''', (FORMAT,))
    for sql in definitions().values():
        db.execute(sql)
    db.execute('CREATE TABLE IF NOT EXISTS data_revisions (name TEXT PRIMARY KEY, revision INTEGER NOT NULL)')
    db.execute('INSERT OR REPLACE INTO data_revisions VALUES (?,?)', (REVISION, FORMAT))
