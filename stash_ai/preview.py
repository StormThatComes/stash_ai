"""Sparse copy-on-write SQLite previews. Source files are attached read-only."""

TABLES = ('results', 'result_thresholds', 'snapshots', 'legacy_imports', 'result_index', 'refresh_requests',
          'checkpoints', 'checkpoint_frames', 'checkpoint_analysis', 'recovery_publications')


def initialize(store):
    if not store.preview_source:
        return
    db = store.connection
    db.execute('''CREATE TABLE IF NOT EXISTS preview_deletions (
        table_name TEXT, scope TEXT, scene TEXT, model TEXT, kind TEXT,
        PRIMARY KEY(table_name,scope,scene,model,kind))''')
    available = {row[0] for row in db.execute("SELECT name FROM cleanup_source.sqlite_master WHERE type='table'")}
    for table in TABLES:
        info = list(db.execute(f'PRAGMA main.table_info({table})'))
        columns = {row[1] for row in info}
        primary = [row[1] for row in sorted(info, key=lambda row: row[5]) if row[5]]
        if table not in available:
            db.execute(f'CREATE TEMP VIEW read_{table} AS SELECT * FROM main.{table}')
            continue
        key_match = ' AND '.join(f'm.{key}=s.{key}' for key in primary)
        deleted = "d.table_name='" + table + "' AND d.scope=s.scope AND d.scene=s.scene"
        if 'model' in columns:
            deleted += " AND (d.model='' OR d.model=s.model)"
        if 'kind' in columns:
            deleted += " AND (d.kind='' OR d.kind=s.kind)"
        db.execute(f'''CREATE TEMP VIEW read_{table} AS SELECT * FROM main.{table}
            UNION ALL SELECT * FROM cleanup_source.{table} AS s
            WHERE NOT EXISTS (SELECT 1 FROM main.{table} AS m WHERE {key_match})
            AND NOT EXISTS (SELECT 1 FROM preview_deletions AS d WHERE {deleted})''')
    if 'merge_checks' in available:
        db.execute('INSERT INTO merge_checks SELECT * FROM cleanup_source.merge_checks WHERE scope=?', (store.scope,))
    db.commit()
    check_source(store)


def table(store, name):
    if name not in TABLES:
        raise ValueError('Unknown preview table.')
    return 'read_' + name if store.preview_source else name


def delete(store, name, scene, model='', kind=''):
    if not store.preview_source:
        return
    if name not in TABLES:
        raise ValueError('Unknown preview table.')
    store.connection.execute('INSERT OR IGNORE INTO preview_deletions VALUES (?,?,?,?,?)',
        (name, store.scope, str(scene), model, kind))


def check_source(store):
    if (store.preview_source and hasattr(store, 'preview_data_version')
            and not getattr(store, 'allow_parent_publication_updates', False)):
        current = store.connection.execute('PRAGMA cleanup_source.data_version').fetchone()[0]
        if current != store.preview_data_version:
            raise RuntimeError('The source state DB changed during dry-run; repeat the preview.')
