"""Share immutable solver vectors while retaining private replay metadata."""
from pathlib import Path

KEY = 'private-replay-vector-source'


def prepare(source, target):
    # Solver seeds are immutable. A live writer owns BEGIN IMMEDIATE; admit
    # its committed source only when the raw clocks match its current snapshot.
    path = next((r[2] for r in source.execute('PRAGMA database_list') if r[1] == 'retained'), None)
    immutable=bool(path)
    if not path:
        import sqlite3
        from contextlib import closing
        from .dependencies import supported
        if not source.in_transaction or not supported(source):return False
        path=next((r[2] for r in source.execute('PRAGMA database_list') if r[1]=='main'),None)
        if not path:return False
        with closing(sqlite3.connect(Path(path).as_uri()+'?mode=ro',uri=True)) as committed:
            query="SELECT key,version FROM evidence_versions WHERE kind='raw-observations' ORDER BY key"
            from itertools import zip_longest
            if any(tuple(a or ())!=tuple(b or ()) for a,b in zip_longest(source.execute(query),committed.execute(query))):
                return False
    target.execute('INSERT INTO main.metadata VALUES (?,?)',(KEY+'-mode','immutable' if immutable else 'live'))
    target.execute('INSERT INTO main.metadata VALUES (?,?)', (KEY, path))
    install(target)
    return True


def install(db):
    if not db.execute("SELECT 1 FROM main.sqlite_master WHERE type='table' AND name='metadata'").fetchone():
        return
    row = db.execute('SELECT value FROM main.metadata WHERE key=?', (KEY,)).fetchone()
    if row is None:
        return
    mode=db.execute('SELECT value FROM main.metadata WHERE key=?',(KEY+'-mode',)).fetchone()
    uri = Path(row[0]).resolve().as_uri()+'?mode=ro'+('' if mode and mode[0]=='live' else '&immutable=1')
    db.execute('ATTACH DATABASE ? AS replay_vectors', (uri,))
    db.execute('PRAGMA replay_vectors.cache_size=-2048')
    db.execute('PRAGMA replay_vectors.mmap_size=0')
    fields = [r[1] for r in db.execute('PRAGMA main.table_info(observations)')]
    columns = ','.join(('v.' if name == 'embedding' else 'o.')+name for name in fields)
    db.execute('CREATE TEMP VIEW observations AS SELECT o.rowid AS rowid,'+columns+
        ' FROM main.observations o LEFT JOIN replay_vectors.observations v ON v.id=o.id')
