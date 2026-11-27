"""Lossless vector sharing across revisions of a scene in a private seed."""
TABLES = ('observations','body_observations')


def revision_heads(source, target, scope, namespace):
    from .scan_heads import predicate
    rows = source.execute('SELECT s.scene,s.id FROM scans s WHERE s.scope=? AND s.namespace=? '
                          'AND s.scene IN (SELECT scene FROM scans WHERE scope=? AND namespace=? '
                          'GROUP BY scene HAVING COUNT(*)>1) AND '+predicate(),(scope,namespace,scope,namespace))
    heads = dict(rows)
    if heads:
        target.execute('CREATE TABLE snapshot_vector_links(kind TEXT,id TEXT,donor TEXT,'
                       'PRIMARY KEY(kind,id)) WITHOUT ROWID')
    return heads


def revision_rows(source, target, table, scan, head):
    # Retain each historical row at its original position. Only link exact
    # payloads to the admitted head, whose own rows always remain inline.
    fields = [r[1] for r in source.execute(f'PRAGMA main.table_info({table})')]
    projection = ','.join('NULL' if k=='crop' else 'old.'+k for k in fields)
    feature = fields.index('embedding')
    policy = 'AND d.contract=old.contract ' if table=='body_observations' else 'AND d.dimensions=old.dimensions '
    rows = source.execute(f'SELECT {projection},(SELECT d.id FROM {table} d WHERE d.scan=? AND d.cue=old.cue '
                          +policy+f'AND d.embedding=old.embedding ORDER BY d.id LIMIT 1) FROM {table} old '
                          'WHERE old.scan=? ORDER BY old.rowid',(head,scan))
    for row in rows:
        values = list(row[:-1])
        if row[-1] is not None:
            target.execute('INSERT INTO snapshot_vector_links VALUES (?,?,?)',(table,values[0],row[-1]))
            values[feature] = None
        yield values


def enabled(db,schema):
    return db.execute(f"SELECT 1 FROM {schema}.sqlite_master WHERE type='table' AND name='snapshot_vector_links'").fetchone() is not None


def projection(db,schema,table):
    if table not in TABLES or not enabled(db,schema):
        return f'SELECT rowid AS rowid,* FROM {schema}.{table}'
    columns = [r[1] for r in db.execute(f'PRAGMA {schema}.table_info({table})')]
    fields = ','.join('coalesce(o.embedding,d.embedding) AS embedding' if k=='embedding' else 'o.'+k for k in columns)
    return (f'SELECT o.rowid AS rowid,{fields} FROM {schema}.{table} o '
            f"LEFT JOIN {schema}.snapshot_vector_links l ON l.kind='{table}' AND l.id=o.id "
            f'LEFT JOIN {schema}.{table} d ON d.id=l.donor')


def install(db):
    if enabled(db,'main'):
        for table in TABLES:
            db.execute(f'CREATE TEMP VIEW IF NOT EXISTS {table} AS '+projection(db,'main',table))
