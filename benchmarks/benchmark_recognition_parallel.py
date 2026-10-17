"""Synthetic RAM-only rebuild scaling; no GPU, network or production forecast."""
import argparse
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import platform
import sqlite3
import statistics
import sys
import tempfile
import time

import numpy as np

from benchmarks.benchmark_recognition_replay import fixture
from stash_ai.recognition.config import load, plain, validate
from stash_ai.recognition.database import FaceDatabase, encoded
from stash_ai.recognition.rebuild import rebuild_selected
from stash_ai.runtime.memory import process_rss


def state_hash(db):
    result = hashlib.sha256()
    for sql in ('SELECT * FROM members ORDER BY observation',
                'SELECT * FROM candidates ORDER BY id',
                'SELECT id,scope,namespace,performer,token FROM people ORDER BY id',
                'SELECT * FROM identity_origins ORDER BY sample_key',
                'SELECT source,target,evidence,status FROM identity_merges ORDER BY source'):
        result.update(sql.encode())
        for row in db.execute(sql):
            result.update(encoded(tuple(row)).encode())
    return result.hexdigest()


def run(scenes=240, people=48, dimensions=128, repeats=1, workers=(1, 8, 16, 20),
        native_threads=1, queue_bytes=67108864):
    if not 16 <= people <= dimensions or dimensions not in (128, 512) or scenes < people or repeats < 1:
        raise ValueError('Require 128/512 dimensions, 16..dimensions people, scenes >= people and positive repeats.')
    trials = []
    for requested in workers:
        for repeat in range(repeats):
            with tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                config = load(root=root)
                data = plain(config.data)
                data['execution'].update(rebuild_workers=requested, rebuild_native_threads=native_threads,
                                         rebuild_queue_bytes=queue_bytes, profile_rebuild=True)
                validate(data)
                config = replace(config, data=data)
                with FaceDatabase(config.database) as state:
                    state.connection.execute("UPDATE metadata SET value=? WHERE key='installation'",
                                             ('61fa1b28-d46c-47f5-843e-e58ae54345d6',))
                    selected = fixture(state, scenes, dimensions, people)
                    for mode in ('cold', 'cached'):
                        started, cpu = time.perf_counter(), time.process_time()
                        rebuild_selected(state, selected, config)
                        elapsed, cpu_elapsed = time.perf_counter()-started, time.process_time()-cpu
                        trials.append(dict(requested_workers=requested, repeat=repeat, mode=mode,
                            wall_seconds=elapsed, process_cpu_seconds=cpu_elapsed,
                            rss_after_bytes=process_rss(), semantic_sha256=state_hash(state.connection),
                            profile=state.last_rebuild_profile))
    return dict(format=1, scope='synthetic-retained-embedding-rebuild-only',
        environment=dict(python=platform.python_version(), numpy=np.__version__, sqlite=sqlite3.sqlite_version,
                         platform=platform.platform(), gil_enabled=getattr(sys, '_is_gil_enabled', lambda: True)()),
        scenes=scenes, observations=scenes*3, people=people, dimensions=dimensions,
        native_threads=native_threads, queue_bytes=queue_bytes,
        identical_semantics=len({t['semantic_sha256'] for t in trials}) == 1,
        medians=[dict(requested_workers=n, mode=m, wall_seconds=statistics.median(
            t['wall_seconds'] for t in trials if t['requested_workers'] == n and t['mode'] == m))
            for n in workers for m in ('cold', 'cached')], trials=trials)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--scenes', type=int, default=240)
    parser.add_argument('--people', type=int, default=48)
    parser.add_argument('--dimensions', type=int, choices=(128, 512), default=128)
    parser.add_argument('--repeats', type=int, default=1)
    parser.add_argument('--workers', type=int, nargs='+', default=[1, 8, 16, 20])
    parser.add_argument('--native-threads', type=int, default=1)
    parser.add_argument('--queue-bytes', type=int, default=67108864)
    args = parser.parse_args()
    try:
        result = run(args.scenes, args.people, args.dimensions, args.repeats,
                     args.workers, args.native_threads, args.queue_bytes)
    except ValueError as error:
        parser.error(str(error))
    print(json.dumps(result, indent=2))
    if not result['identical_semantics']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
