"""Read-only observation and baseline maps shared by spawned solver processes."""
from collections.abc import Mapping
from contextlib import closing
import os
import sqlite3

from contextlib import contextmanager
from contextvars import ContextVar
from pathlib import Path

_owner = ContextVar('coordinator_slots', default=None)


@contextmanager
def coordinator(images, source, config, scans):
    """Stream scoped retained inputs into the same file opened by workers."""
    from .body_identity import enabled, available
    from .body_observations import contract
    from .database import encoded
    from .disk_budget import reserve_file
    from .resource_errors import ResourceLimit
    path = images.directory / 'observations.sqlite'
    selected = encoded([scan['id'] for scan in scans])
    path.touch(exist_ok=False)
    room = images.limit - sum(p.stat().st_size for p in images.directory.iterdir())
    with reserve_file(path, room) as capacity, closing(sqlite3.connect(path)) as db:
        if capacity < 65536:
            raise ResourceLimit('Solver observation index exceeds the configured snapshot budget.')
        db.execute(f'PRAGMA max_page_count={capacity // 4096}')
        db.execute('CREATE TABLE slots(ordinal INTEGER PRIMARY KEY,id TEXT UNIQUE,scene TEXT,retained TEXT,baseline TEXT)')
        body = enabled(config) and available(source)
        join = " LEFT JOIN body_members b ON b.observation=o.id AND b.status='accepted'" if body else ''
        person = 'CASE WHEN b.observation IS NULL THEN f.person ELSE b.person END' if body else 'f.person'
        for table in ('observations', 'body_observations') if body else ('observations',):
            sql = f'SELECT o.id,s.scene,{person} FROM {table} o JOIN scans s ON s.id=o.scan ' + (
                'LEFT JOIN members f ON f.observation=o.id' + join +
                ' WHERE o.scan IN (SELECT value FROM json_each(?))')
            args = (selected,)
            if table == 'body_observations':
                sql += ' AND o.contract=?'
                args += (contract(config),)
            db.executemany('INSERT INTO slots(id,scene,retained,baseline) VALUES (?,?,?,?) '
                           'ON CONFLICT(id) DO UPDATE SET scene=excluded.scene,retained=excluded.retained,baseline=excluded.baseline',
                           ((oid,scene,owner,owner) for oid,scene,owner in source.execute(sql,args)))
        db.execute('CREATE INDEX slot_baseline_owner ON slots(baseline,id)')
        db.execute('CREATE INDEX baseline_initial_changes ON slots(id) WHERE retained IS NOT baseline')
        from .body_observations import coverage as body_coverage
        from .person_signal import coverage as hybrid_coverage
        scan_ids = [scan['id'] for scan in scans]
        db.execute('CREATE TABLE input_counts(body_enabled INTEGER, body TEXT, hybrid TEXT)')
        db.execute('INSERT INTO input_counts VALUES (?,?,?)',
                   (body, encoded(body_coverage(source, config, scan_ids)),
                    encoded(hybrid_coverage(source, config, scan_ids))))
        db.commit()
        size = db.execute('SELECT count(*) FROM slots').fetchone()[0]
        observations = Slots(db, 'scene', size)
        token = _owner.set(observations)
        try:
            yield observations, Slots(db, 'retained', size)
        finally:
            _owner.reset(token)


def apply_baseline(changes):
    slots = _owner.get()
    if slots is None:
        raise ValueError('Missing solver coordinator index.')
    with slots.db:
        slots.db.executemany('UPDATE slots SET baseline=? WHERE id=?', ((person,oid) for oid,person in changes.items()))
    return Slots(slots.db, 'baseline', len(slots))


def track(target):
    """Membership writes are transactional, including deletes and old addresses."""
    target.execute('CREATE TEMP TABLE IF NOT EXISTS trial_changed(id TEXT PRIMARY KEY) WITHOUT ROWID')
    for table in ('members', 'body_members'):
        for event, sides in [('INSERT', ('new',)), ('DELETE', ('old',)), ('UPDATE', ('old','new'))]:
            statements = ' '.join(f'INSERT INTO trial_changed VALUES ({side}.observation) ON CONFLICT DO NOTHING;'
                                  for side in sides)
            target.execute(f'CREATE TEMP TRIGGER IF NOT EXISTS trial_{table}_{event} AFTER {event} ON main.{table} BEGIN {statements} END')


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

    def items(self):
        return self.db.execute(f'SELECT id,{self.column} FROM slots ORDER BY ordinal')

    def input_counts(self, settings):
        import json
        from .config import plain
        from .body_observations import contract, enabled
        from .person_signal import contract as person_contract
        if not self.db.execute("SELECT 1 FROM sqlite_master WHERE name='input_counts'").fetchone():
            return None, None
        data = self.db.execute('SELECT body,hybrid FROM input_counts').fetchone()
        body, hybrid = map(json.loads, data)
        if body.get('contract') != contract(settings):
            body = None
        else:
            body['enabled'] = enabled(settings)
        if hybrid['contract'] != person_contract(settings):
            hybrid = None
        else:
            hybrid.update(enabled=settings.data['sapiensid']['enabled'], matching=plain(settings.data['sapiensid']['matching']))
        return body, hybrid

    def for_owners(self, owners):
        import json
        return self.db.execute(f'SELECT id,{self.column} FROM slots WHERE {self.column} IN '
                               '(SELECT value FROM json_each(?))', (json.dumps(sorted(owners)),))

    def __getitem__(self, key):
        row = self.db.execute(f'SELECT {self.column} FROM slots WHERE id=?', (key,)).fetchone()
        if row is None:
            raise KeyError(key)
        return row[0]

    def attach(self, target):
        from pathlib import Path
        path = self.db.execute('PRAGMA database_list').fetchone()[2]
        target.execute('ATTACH DATABASE ? AS solver_slots', (Path(path).as_uri() + '?mode=ro&immutable=1',))
        target.execute('PRAGMA solver_slots.mmap_size=1073741824')

    def prediction_delta(self, target, settings):
        from .body_identity import available, enabled
        attached = any(row[1] == 'solver_slots' for row in target.execute('PRAGMA database_list'))
        if not attached:
            self.attach(target)
        try:
            owner = 'f.person'
            joins = 'LEFT JOIN members f ON f.observation=s.id'
            if enabled(settings) and available(target):
                joins += " LEFT JOIN body_members b ON b.observation=s.id AND b.status='accepted'"
                owner = 'CASE WHEN b.observation IS NULL THEN f.person ELSE b.person END'
            tracked = target.execute("SELECT 1 FROM sqlite_temp_master WHERE name='trial_changed'").fetchone()
            indexed = any(row[1] == 'retained' for row in target.execute('PRAGMA solver_slots.table_info(slots)'))
            population = 'solver_slots.slots s'
            same_projection = indexed and self.db.execute('SELECT body_enabled FROM input_counts').fetchone()[0] == enabled(settings)
            if tracked and same_projection:
                population = ('(SELECT id FROM trial_changed UNION SELECT id FROM solver_slots.slots '
                              'WHERE retained IS NOT baseline) d JOIN solver_slots.slots s ON s.id=d.id')
            rows = target.execute(f'SELECT s.id,{owner} FROM {population} {joins} '
                                  f'WHERE s.baseline IS NOT ({owner})')
            return dict(rows)
        finally:
            if not attached:
                target.execute('DETACH DATABASE solver_slots')


def open_maps(path):
    db = sqlite3.connect(path.as_uri() + '?mode=ro&immutable=1', uri=True)
    try:
        db.execute('PRAGMA mmap_size=1073741824')
        size, = db.execute('SELECT COUNT(*) FROM slots').fetchone()
        return db, Slots(db, 'scene', size), Slots(db, 'baseline', size)
    except BaseException:
        db.close()
        raise
