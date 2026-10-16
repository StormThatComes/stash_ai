"""Offline synthetic replay comparison; no models, network or speedup forecast."""
import argparse
from contextlib import contextmanager
from dataclasses import replace
import json
from pathlib import Path
import platform
import sqlite3
import statistics
import tempfile
import time
import uuid

import numpy as np

from stash_ai.recognition.config import load,plain
from stash_ai.recognition.database import FaceDatabase,encoded
from stash_ai.recognition import rebuild
from stash_ai.runtime.memory import process_rss


def fixture(state, scenes, dimensions=128, people=6):
    selected=[]
    for index in range(scenes):
        scene=str(index+1)
        identifier=str(uuid.uuid5(uuid.NAMESPACE_URL,'synthetic-scan-'+scene))
        state.connection.execute('INSERT INTO scans VALUES (?,?,?,?,?,?,?,?,?,?,?)',
            (identifier,'scope',scene,'ns',scene,'detection','{}','{}',3,3,'complete'))
        v=np.zeros(dimensions,dtype='<f4');v[index%people]=1
        for cue in range(3):
            observation=str(uuid.uuid5(uuid.UUID(identifier),str(cue)))
            state.connection.execute('INSERT INTO observations VALUES (?,?,?,?,?,?,?,?,?,?,?,?)',
                (observation,identifier,cue,float(cue),'source','[]','{}','{}',None,dimensions,v.tobytes(),None))
        selected.append(dict(scene=scene,scan=identifier))
    state.connection.commit()
    return selected


def run(scenes,repeats):
    results={}
    original=rebuild.memory_replay
    for mode in ('temporary-file','ram-serial','ram-lookahead'):
        trials=[]
        for _ in range(repeats):
            with tempfile.TemporaryDirectory() as directory:
                root=Path(directory);path=root/'config.yml'
                path.write_text('version: 1\nexecution: {profile_rebuild: true}\nclustering: {max_samples_per_person: 4}\n')
                config=load(path)
                data=plain(config.data)
                data['execution']['rebuild_prefetch_scenes']=mode=='ram-lookahead'
                config=replace(config,data=data)
                @contextmanager
                def disk_replay(config):
                    db=sqlite3.connect(root/'replay.sqlite3');db.row_factory=sqlite3.Row
                    try:
                        yield db,''
                    finally:
                        db.close()
                rebuild.memory_replay=disk_replay if mode=='temporary-file' else original
                try:
                    with FaceDatabase(config.database) as state:
                        selected=fixture(state,scenes)
                        started=time.perf_counter()
                        rebuild.rebuild_selected(state,selected,config)
                        elapsed=time.perf_counter()-started
                        partition=sorted(sorted(r[0] for r in state.connection.execute(
                            'SELECT s.scene FROM members m JOIN observations o ON o.id=m.observation '
                            'JOIN scans s ON s.id=o.scan WHERE m.person=?',(person[0],)))
                            for person in state.connection.execute('SELECT DISTINCT person FROM members'))
                        profile=state.last_rebuild_profile
                        trials.append(dict(seconds=elapsed,partition=partition,counts=profile['counts'],
                                           sql=profile['sql'],rss_after_bytes=process_rss()))
                finally:
                    rebuild.memory_replay=original
        results[mode]=dict(median_seconds=statistics.median(t['seconds'] for t in trials),trials=trials)
    partitions=[t.pop('partition') for mode in results.values() for t in mode['trials']]
    return dict(scope='synthetic-offline-rebuild-only',python=platform.python_version(),
                numpy=np.__version__,sqlite=sqlite3.sqlite_version,platform=platform.platform(),
                scenes=scenes,observations=scenes*3,identities=6,dimensions=128,representatives=4,
                identical_partition=all(p==partitions[0] for p in partitions),results=results)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--scenes',type=int,default=240)
    parser.add_argument('--repeats',type=int,default=3)
    args=parser.parse_args()
    if args.scenes<6 or args.repeats<1:
        parser.error('require at least six scenes and one repeat')
    print(json.dumps(run(args.scenes,args.repeats),indent=2))


if __name__=='__main__':
    main()
