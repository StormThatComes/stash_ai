"""Read-only observation and baseline maps shared by spawned solver processes."""
from collections.abc import Mapping
from contextlib import closing
import os
import sqlite3


def build(images, observations, baseline):
    """Ordinal is the coordinator's insertion order, including unassigned observations."""
    path = images.directory / 'observations.sqlite'
    partial = path.with_suffix('.partial')
    room = images.limit - sum(p.stat().st_size for p in images.directory.iterdir())
    if room < 65536:
        return None
    from .disk_budget import reserve_file
    try:
        partial.touch(exist_ok=False)
        with reserve_file(partial, room) as allowed, closing(sqlite3.connect(partial)) as db:
            if allowed < 65536:
                return None
            db.execute(f'PRAGMA max_page_count={allowed // 4096}')
            db.execute('CREATE TABLE slots(ordinal INTEGER PRIMARY KEY, id TEXT UNIQUE, scene TEXT, baseline TEXT)')
            db.executemany('INSERT INTO slots(id,scene,baseline) VALUES (?,?,?)',
                           ((key, scene, baseline[key]) for key, scene in observations.items()))
            db.commit()
        os.replace(partial, path)
        return path
    except (sqlite3.Error, OSError):
        return None
    finally:
        partial.unlink(missing_ok=True)


class Slots(Mapping):
    def __init__(self, db, column, size):
        self.db, self.column, self.size = db, column, size

    def __len__(self):
        return self.size

    def __iter__(self):
        return (key for key, in self.db.execute('SELECT id FROM slots ORDER BY ordinal'))

    def __getitem__(self, key):
        row = self.db.execute(f'SELECT {self.column} FROM slots WHERE id=?', (key,)).fetchone()
        if row is None:
            raise KeyError(key)
        return row[0]


def open_maps(path):
    db = sqlite3.connect(path.as_uri() + '?mode=ro&immutable=1', uri=True)
    try:
        db.execute('PRAGMA mmap_size=1073741824')
        size, = db.execute('SELECT COUNT(*) FROM slots').fetchone()
        return db, Slots(db, 'scene', size), Slots(db, 'baseline', size)
    except BaseException:
        db.close()
        raise
