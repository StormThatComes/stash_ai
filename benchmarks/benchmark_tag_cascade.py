"""Compare native tag cascade with prior bounded detachment in disposable SQLite.

Models indexed marker/scene/image associations from Stash's schema. The detached
variant omits API/network, hooks and object loading, giving it a lower-bound cost.
No connection to Stash, application state or inference is made.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import sqlite3
import statistics
import tempfile
from time import perf_counter


TABLES = ('tags', 'tags_image', 'scenes', 'scene_markers', 'images',
          'scene_markers_tags', 'scenes_tags', 'images_tags')


def fixture(markers):
    db = sqlite3.connect(':memory:')
    db.executescript('''PRAGMA foreign_keys=ON;
        CREATE TABLE tags(id INTEGER PRIMARY KEY, name TEXT UNIQUE);
        CREATE TABLE tags_image(tag_id INTEGER PRIMARY KEY REFERENCES tags(id) ON DELETE CASCADE, image BLOB);
        CREATE TABLE scenes(id INTEGER PRIMARY KEY, title TEXT);
        CREATE TABLE images(id INTEGER PRIMARY KEY, title TEXT);
        CREATE TABLE scene_markers(id INTEGER PRIMARY KEY,
            scene_id INTEGER REFERENCES scenes(id), title TEXT,
            primary_tag_id INTEGER NOT NULL REFERENCES tags(id));
        CREATE INDEX index_scene_markers_on_primary_tag_id ON scene_markers(primary_tag_id);
        INSERT INTO tags VALUES(1,'ai:blocked'),(2,'manual:keep'),(3,'ai:owner');
        INSERT INTO tags_image VALUES(1,X'0102'),(2,X'0304');''')
    for table, column, parent in [('scene_markers_tags', 'scene_marker_id', 'scene_markers'),
                                   ('scenes_tags', 'scene_id', 'scenes'),
                                   ('images_tags', 'image_id', 'images')]:
        db.executescript(f'''CREATE TABLE {table}(
            {column} INTEGER REFERENCES {parent}(id) ON DELETE CASCADE,
            tag_id INTEGER REFERENCES tags(id) ON DELETE CASCADE,
            PRIMARY KEY({column},tag_id));
            CREATE INDEX index_{table}_on_tag_id ON {table}(tag_id);''')
    scenes = max(1, math.ceil(markers / 100))
    db.executemany('INSERT INTO scenes VALUES(?,?)', ((i, f'scene {i}') for i in range(1, scenes + 1)))
    db.executemany('INSERT INTO scene_markers VALUES(?,?,?,2)',
                   ((i, 1 + (i - 1) // 100, f'marker {i}') for i in range(1, markers + 1)))
    db.executemany('INSERT INTO images VALUES(?,?)', ((i, f'image {i}') for i in range(1, 21)))
    for table, parent, keep in [('scene_markers_tags', 'scene_markers', 3),
                               ('scenes_tags', 'scenes', 2), ('images_tags', 'images', 2)]:
        db.execute(f'INSERT INTO {table} SELECT id,1 FROM {parent}')
        db.execute(f'INSERT INTO {table} SELECT id,? FROM {parent}', (keep,))
    db.commit()
    return db


def fingerprint(db):
    digest, counts = hashlib.sha256(), {}
    for table in TABLES:
        digest.update(table.encode())
        counts[table] = 0
        for row in db.execute(f'SELECT * FROM {table} ORDER BY 1,2'):
            # BLOBs have a stable JSON representation, independent of DB layout.
            row = [value.hex() if isinstance(value, bytes) else value for value in row]
            digest.update(json.dumps(row, separators=(',', ':')).encode() + b'\n')
            counts[table] += 1
    assert not list(db.execute('PRAGMA foreign_key_check'))
    assert db.execute('SELECT COUNT(*) FROM scene_markers WHERE primary_tag_id != 2').fetchone()[0] == 0
    return digest.hexdigest(), counts


def trial(seed, path, *, detach, batch_size):
    db = sqlite3.connect(path, isolation_level='IMMEDIATE')
    try:
        seed.backup(db)  # Fixture construction/copy are outside measured work.
        # Stash v0.31.1: WAL, NORMAL, foreign keys and immediate writes.
        db.executescript('''PRAGMA foreign_keys=ON; PRAGMA journal_mode=WAL;
            PRAGMA synchronous=NORMAL; PRAGMA busy_timeout=50;''')
        durations = []
        started = perf_counter()
        if detach:
            for table, column in [('scene_markers_tags', 'scene_marker_id'), ('scenes_tags', 'scene_id')]:
                after = 0
                while True:
                    rows = list(db.execute(
                        f'SELECT rowid,{column} FROM {table} WHERE tag_id=1 AND rowid>? '
                        'ORDER BY rowid LIMIT ?', (after, batch_size)))
                    if not rows:
                        break
                    transaction = perf_counter()
                    with db:
                        # Native bulk REMOVE modifies each object's join rows.
                        # Omit object reads/updates and hooks: optimistic lower bound.
                        db.executemany(f'DELETE FROM {table} WHERE {column}=? AND tag_id=1',
                                       ((identifier,) for _, identifier in rows))
                    durations.append(perf_counter() - transaction)
                    after = rows[-1][0]
        transaction = perf_counter()
        with db:
            # Tag.Destroy clears its image, rejects primary references, then
            # deletes the existing tag. Foreign keys remove remaining joins.
            db.execute('DELETE FROM tags_image WHERE tag_id=1')
            assert db.execute('SELECT COUNT(*) FROM scene_markers WHERE primary_tag_id=1').fetchone()[0] == 0
            assert db.execute('SELECT id FROM tags WHERE id=1').fetchone() == (1,)
            db.execute('DELETE FROM tags WHERE id=1')
        durations.append(perf_counter() - transaction)
        elapsed = perf_counter() - started
        state_hash, counts = fingerprint(db)  # Validation is outside measured time.
        return {'seconds': elapsed, 'max_transaction_seconds': max(durations),
                'transactions': len(durations), 'native_mutations': len(durations),
                'state_sha256': state_hash, 'remaining_rows': counts}
    finally:
        db.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--markers', type=int, default=100_000)
    parser.add_argument('--batch-size', type=int, default=200)
    parser.add_argument('--repeats', type=int, default=3)
    args = parser.parse_args()
    if min(args.markers, args.batch_size, args.repeats) < 1:
        parser.error('All counts must be positive.')
    seed, trials = fixture(args.markers), []
    try:
        with tempfile.TemporaryDirectory(prefix='stash-tag-cascade-') as directory:
            for repeat in range(args.repeats):
                result = {}
                # Alternate order to reduce systematic filesystem/cache bias.
                for name in (('cascade', 'detach') if repeat % 2 == 0 else ('detach', 'cascade')):
                    result[name] = trial(seed, Path(directory) / f'{repeat}-{name}.sqlite3',
                                         detach=name == 'detach', batch_size=args.batch_size)
                assert result['cascade']['state_sha256'] == result['detach']['state_sha256']
                trials.append(result)
    finally:
        seed.close()
    summary = {}
    for name in ('cascade', 'detach'):
        summary[name] = {
            'median_ms': statistics.median(t[name]['seconds'] for t in trials) * 1000,
            'median_max_transaction_ms': statistics.median(t[name]['max_transaction_seconds'] for t in trials) * 1000,
            'native_mutations': trials[0][name]['native_mutations']}
    print(json.dumps({'scope': 'Disposable indexed SQLite, WAL/NORMAL/immediate writes; no API, network, hooks or object loading.',
                      'sqlite_version': sqlite3.sqlite_version, 'markers': args.markers,
                      'scenes': max(1, math.ceil(args.markers / 100)), 'batch_size': args.batch_size,
                      'identical_state': True, 'trials': trials, 'summary': summary}, indent=2))


if __name__ == '__main__':
    main()
