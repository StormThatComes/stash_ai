"""Share immutable solver vectors while retaining private replay metadata."""
from pathlib import Path

KEY = 'private-replay-vector-source'


def prepare(source, target):
    # Only solver trials carry this immutable attachment. A live run always
    # copies its vectors, preserving its original snapshot isolation.
    path = next((r[2] for r in source.execute('PRAGMA database_list') if r[1] == 'retained'), None)
    if not path:
        return False
    target.execute('INSERT INTO main.metadata VALUES (?,?)', (KEY, path))
    install(target)
    return True


def install(db):
    if not db.execute("SELECT 1 FROM main.sqlite_master WHERE type='table' AND name='metadata'").fetchone():
        return
    row = db.execute('SELECT value FROM main.metadata WHERE key=?', (KEY,)).fetchone()
    if row is None:
        return
    uri = Path(row[0]).resolve().as_uri()+'?mode=ro&immutable=1'
    db.execute('ATTACH DATABASE ? AS replay_vectors', (uri,))
    db.execute('PRAGMA replay_vectors.cache_size=-2048')
    db.execute('PRAGMA replay_vectors.mmap_size=0')
    fields = [r[1] for r in db.execute('PRAGMA main.table_info(observations)')]
    columns = ','.join(('v.' if name == 'embedding' else 'o.')+name for name in fields)
    db.execute('CREATE TEMP VIEW observations AS SELECT o.rowid AS rowid,'+columns+
        ' FROM main.observations o LEFT JOIN replay_vectors.observations v ON v.id=o.id')
