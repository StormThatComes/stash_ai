"""Synthetic recovery cost and skipped work; no production speedup forecast."""
import argparse
from dataclasses import replace
import json
import os
from pathlib import Path
import platform
import sqlite3
import statistics
import tempfile
from threading import Event, Thread
import time

import numpy as np

from stash_ai.recognition import rebuild
from stash_ai.recognition.config import load, plain
from stash_ai.recognition.database import FaceDatabase
from stash_ai.runtime.memory import process_rss
from .benchmark_recognition_replay import fixture


def run(scenes, repeats):
    results, outputs = {}, []
    for mode in ('full','checkpointed','resumed','safe-append'):
        trials=[]
        for _ in range(repeats):
            with tempfile.TemporaryDirectory() as directory:
                root=Path(directory);path=root/'config.yml'
                interval=max(1,scenes//4)
                path.write_text('version: 1\nclustering: {max_samples_per_person: 4}\n')
                config=load(path);data=plain(config.data)
                data['execution'].update(profile_rebuild=True,rebuild_checkpoint_scenes=0 if mode=='full' else interval)
                config=replace(config,data=data)
                with FaceDatabase(config.database) as state:
                    selected=fixture(state,scenes)
                    prefix=3*scenes//4
                    if mode=='resumed':
                        original=rebuild.cluster_scene
                        n=0
                        def stop(*args,**kwargs):
                            nonlocal n
                            if n==prefix:raise KeyboardInterrupt('synthetic interruption')
                            n+=1
                            return original(*args,**kwargs)
                        rebuild.cluster_scene=stop
                        try:
                            rebuild.rebuild_selected(state,selected,config)
                        except KeyboardInterrupt:
                            pass
                        finally:
                            rebuild.cluster_scene=original
                    elif mode=='safe-append':
                        rebuild.rebuild_selected(state,selected[:prefix],config)
                    stopped=Event();peak=[process_rss()]
                    def sample():
                        while not stopped.wait(.01):peak[0]=max(peak[0],process_rss())
                    worker=Thread(target=sample);worker.start()
                    started=time.perf_counter()
                    try:
                        rebuild.rebuild_selected(state,selected,config)
                        elapsed=time.perf_counter()-started
                    finally:
                        stopped.set();worker.join()
                        peak[0]=max(peak[0],process_rss())
                    partition=sorted(sorted(r[0] for r in state.connection.execute(
                        'SELECT observation FROM members WHERE person=?',(p[0],)))
                        for p in state.connection.execute('SELECT DISTINCT person FROM members'))
                    decisions=[(r['id'],r['status'],json.loads(r['payload'])['scores'])
                               for r in state.connection.execute('SELECT * FROM candidates ORDER BY id')]
                    outputs.append((partition,decisions))
                    profile=state.last_rebuild_profile
                    sidecar=state.path.with_name(state.path.name+'.rebuild')
                    sizes=[p.stat().st_size for p in sidecar.glob('replay-*.sqlite3')]
                    trials.append(dict(seconds=elapsed,rss_sampled_peak_bytes=peak[0],
                        retained_generation_bytes=sum(sizes),counts=profile['counts'],sql=profile['sql'],stages=profile['stages']))
        results[mode]=dict(median_seconds=statistics.median(t['seconds'] for t in trials),trials=trials)
    return dict(scope='synthetic-offline-recovery-only',python=platform.python_version(),numpy=np.__version__,
        sqlite=sqlite3.sqlite_version,platform=platform.platform(),cpu=platform.processor(),logical_cpus=os.cpu_count(),
        scenes=scenes,observations=3*scenes,identities=6,dimensions=128,representatives=4,
        checkpoint_interval=max(1,scenes//4),interrupted_at=3*scenes//4,
        identical_partition_and_scores=all(o==outputs[0] for o in outputs),results=results)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--scenes',type=int,default=240)
    parser.add_argument('--repeats',type=int,default=2)
    args=parser.parse_args()
    if args.scenes<8 or args.scenes%4 or args.repeats<1:parser.error('require scenes >=8 divisible by four and repeats >=1')
    print(json.dumps(run(args.scenes,args.repeats),indent=2))
