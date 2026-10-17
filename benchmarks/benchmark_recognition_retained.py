"""Read-only retained-evidence input; cold/cached benchmark copies stay in RAM."""
import argparse
from contextlib import closing, contextmanager
from dataclasses import replace
import hashlib
from pathlib import Path
import platform
import sqlite3
import sys
from threading import Event, Thread
import time
from types import SimpleNamespace

import numpy as np
from threadpoolctl import threadpool_info

from benchmarks.benchmark_recognition_parallel import state_hash
from stash_ai.recognition.config import load, plain, validate, digest
from stash_ai.recognition.database import SCHEMA, TABLES, FaceDatabase, encoded
from stash_ai.recognition.fragmentation_report import _open_readonly
from stash_ai.recognition.models import namespace as model_namespace
from stash_ai.recognition.rebuild import rebuild_selected
from stash_ai.recognition.rebuild_cache import _hash_row
from stash_ai.runtime.memory import process_rss


@contextmanager
def sampled_rss():
    """A sampled process high-water mark, including the private RAM copies."""
    stopped = Event()
    result = dict(method='sampled-process-rss', interval_seconds=.05,
                  sampled_peak_bytes=process_rss(), samples=1)
    def sample():
        value = process_rss()
        if value is not None:
            result['sampled_peak_bytes'] = max(result['sampled_peak_bytes'] or 0, value)
        result['samples'] += 1
    def monitor():
        while not stopped.wait(.05):
            sample()
    thread = Thread(target=monitor, name='recognition-benchmark-rss', daemon=True)
    thread.start()
    try:
        yield result
    finally:
        stopped.set()
        thread.join()
        sample()
        result['rss_after_bytes'] = process_rss()


def _ram(config):
    db = sqlite3.connect(':memory:')
    db.row_factory = sqlite3.Row
    db.execute('PRAGMA temp_store=MEMORY')
    db.execute('PRAGMA journal_mode=MEMORY')
    page = db.execute('PRAGMA page_size').fetchone()[0]
    db.execute(f"PRAGMA max_page_count={max(1,config.data['execution']['rebuild_memory_bytes']//page)}")
    return db


def _selection(source, scenes, all_scenes, scope, namespace):
    if bool(scenes) == bool(all_scenes):
        raise ValueError('Choose scene IDs or all scenes explicitly.')
    if scenes and (any(not isinstance(s, str) or not s.isdecimal() or int(s) < 1 for s in scenes)
                   or len(set(scenes)) != len(scenes)):
        raise ValueError('Scene IDs must be distinct positive decimal strings.')
    scopes = [r[0] for r in source.execute('SELECT DISTINCT scope FROM scans WHERE namespace=?', (namespace,))]
    if scope is None:
        if len(scopes) != 1:
            raise ValueError('Select one explicit scope for this recognizer namespace.')
        scope = scopes[0]
    selected = []
    rows = source.execute('SELECT s.id,s.scene,s.status FROM scans s WHERE s.scope=? AND s.namespace=? '
        'AND NOT EXISTS (SELECT 1 FROM scans n WHERE n.scope=s.scope AND n.namespace=s.namespace '
        'AND n.scene=s.scene AND n.rowid>s.rowid) ORDER BY CAST(s.scene AS INTEGER),s.scene', (scope, namespace))
    wanted = set(scenes or ())
    skipped = 0
    for row in rows:
        if scenes and row['scene'] not in wanted:
            continue
        if row['status'] != 'complete':
            skipped += 1
            continue
        selected.append(dict(scene=row['scene'], scan=row['id']))
    if not selected or scenes and {s['scene'] for s in selected} != wanted:
        raise ValueError('Requested scenes require latest complete scans in one scope/namespace.')
    return scope, selected, skipped


def _snapshot(source, seed, scope, namespace):
    """Known schema/columns only. Do not read quality, crops, portraits or runs."""
    seed.executescript(SCHEMA)
    owner = SimpleNamespace(connection=source)
    installation = FaceDatabase.installation.fget(owner)
    seed.execute('INSERT INTO metadata VALUES (?,?)', ('installation', installation))
    statements = {
        'scans': ('SELECT * FROM scans WHERE scope=? AND namespace=? ORDER BY rowid', (scope, namespace)),
        'observations': ("SELECT o.id,o.scan,o.cue,o.seconds,o.source,o.rect,o.geometry,'{}',o.rejected,"
            'o.dimensions,o.embedding,NULL FROM observations o JOIN scans s ON s.id=o.scan '
            'WHERE s.scope=? AND s.namespace=? ORDER BY s.rowid,o.cue,o.id', (scope, namespace)),
        'people': ('SELECT * FROM people WHERE scope=? AND namespace=? ORDER BY created,id', (scope, namespace)),
        'members': ('SELECT m.* FROM members m JOIN people p ON p.id=m.person '
            'WHERE p.scope=? AND p.namespace=? ORDER BY m.observation', (scope, namespace)),
        'identity_merges': ('SELECT m.* FROM identity_merges m JOIN people p ON p.id=m.source '
            'WHERE p.scope=? AND p.namespace=? ORDER BY m.source', (scope, namespace)),
        'identity_origins': ('SELECT o.* FROM identity_origins o JOIN people p ON p.id=o.person '
            'WHERE p.scope=? AND p.namespace=? ORDER BY o.sample_key', (scope, namespace)),
    }
    for table, (sql, args) in statements.items():
        rows = source.execute(sql, args)
        columns = len(rows.description)
        seed.executemany(f'INSERT INTO {table} VALUES ({",".join("?" for _ in range(columns))})', rows)
    seed.commit()
    return installation


def run(config, scenes=None, *, all_scenes=False, scope=None, namespace=None,
        workers=(1, 8, 16, 20), repeats=1, database=None):
    if repeats < 1 or not workers or len(workers) != len(set(workers)):
        raise ValueError('Require positive repeats and distinct worker budgets.')
    namespace = namespace or model_namespace(config)
    started = time.perf_counter()
    with closing(_ram(config)) as seed:
        with closing(_open_readonly(database or config.database)) as source:
            # Validate via the existing read-only schema checker; never construct
            # FaceDatabase, acquire its writer lock or run orphan recovery.
            tables = {r[0] for r in source.execute("SELECT name FROM sqlite_master WHERE type='table'")}
            FaceDatabase._validate(SimpleNamespace(connection=source), SCHEMA, TABLES, tables)
            scope, selected, skipped = _selection(source, scenes, all_scenes, scope, namespace)
            installation = _snapshot(source, seed, scope, namespace)
        input_hash = hashlib.sha256()
        for table in ('scans', 'observations', 'people', 'members', 'identity_merges', 'identity_origins'):
            for row in seed.execute(f'SELECT * FROM {table} ORDER BY 1'):
                _hash_row(input_hash, row)
        dimensions = [dict(dimensions=r[0], observations=r[1]) for r in seed.execute(
            'SELECT dimensions,COUNT(*) FROM observations WHERE rejected IS NULL GROUP BY dimensions ORDER BY dimensions')]
        selected_ids = encoded([item['scan'] for item in selected])
        selected_counts = seed.execute('SELECT COUNT(*),SUM(rejected IS NULL) FROM observations '
            'WHERE scan IN (SELECT value FROM json_each(?))', (selected_ids,)).fetchone()
        manifest = dict(format=1, retained_sha256=input_hash.hexdigest(), selection_sha256=digest(selected),
                        scenes=len(selected), skipped_latest_incomplete=skipped,
                        selected_observations=selected_counts[0], selected_accepted_observations=selected_counts[1] or 0,
                        retained_embedding_bytes=seed.execute('SELECT COALESCE(SUM(length(embedding)),0) FROM observations').fetchone()[0],
                        retained_scans=seed.execute('SELECT COUNT(*) FROM scans').fetchone()[0],
                        retained_observations=seed.execute('SELECT COUNT(*) FROM observations').fetchone()[0],
                        historical_people=seed.execute('SELECT COUNT(*) FROM people').fetchone()[0], dimensions=dimensions,
                        config_sha256=digest(config.data))
        snapshot_seconds = time.perf_counter()-started
        trials = []
        for requested in workers:
            for repeat in range(repeats):
                data = plain(config.data)
                data['execution'].update(rebuild_workers=requested, profile_rebuild=True, rebuild_checkpoint_scenes=0)
                validate(data)
                trial_config = replace(config, data=data)
                with closing(_ram(config)) as target:
                    seed.backup(target)
                    state = SimpleNamespace(connection=target, installation=installation)
                    for mode in ('cold', 'cached'):
                        with sampled_rss() as memory:
                            start, cpu = time.perf_counter(), time.process_time()
                            rebuild_selected(state, selected, trial_config)
                            wall, used = time.perf_counter()-start, time.process_time()-cpu
                        trials.append(dict(requested_workers=requested, repeat=repeat, mode=mode,
                            wall_seconds=wall, process_cpu_seconds=used, memory=memory,
                            semantic_sha256=state_hash(target), profile=state.last_rebuild_profile))
        return dict(format=1, scope='local-retained-evidence-rebuild-only', manifest=manifest,
            source_read_only=True, working_storage='RAM', snapshot_seconds=snapshot_seconds,
            environment=dict(python=platform.python_version(), numpy=np.__version__, platform=platform.platform(),
                native_libraries=[{k: library.get(k) for k in ('internal_api', 'prefix', 'version')}
                                  for library in threadpool_info()],
                sqlite=sqlite3.sqlite_version, gil_enabled=getattr(sys, '_is_gil_enabled', lambda: True)()),
            identical_semantics=len({t['semantic_sha256'] for t in trials}) == 1, trials=trials)


def main():
    import json
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--database')
    selection = parser.add_mutually_exclusive_group(required=True)
    selection.add_argument('--scene-id', nargs='+')
    selection.add_argument('--all-scenes', action='store_true')
    parser.add_argument('--scope')
    parser.add_argument('--namespace')
    parser.add_argument('--workers', type=int, nargs='+', default=[1, 8, 16, 20])
    parser.add_argument('--repeats', type=int, default=1)
    args = parser.parse_args()
    try:
        result = run(load(args.config), args.scene_id, all_scenes=args.all_scenes,
            scope=args.scope, namespace=args.namespace, workers=tuple(args.workers),
            repeats=args.repeats, database=args.database)
    except (ValueError, sqlite3.DatabaseError) as error:
        parser.exit(2, f'Benchmark failed: {error}\n')
    print(json.dumps(result, indent=2))
    if not result['identical_semantics']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
