"""Benchmark real retained recognition, incremental reconciliation and solver replay."""
import argparse
from collections import Counter, defaultdict
from contextlib import closing
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import time
import uuid

import numpy as np

from benchmarks.benchmark_recognition_retained import sampled_rss
from benchmarks.benchmark_scene_library import process_io
from stash_ai.recognition.body_observations import contract, save_records
from stash_ai.recognition.config import load, plain, validate
from stash_ai.recognition.database import FaceDatabase, encoded
from stash_ai.recognition.fragmentation_report import _open_readonly
from stash_ai.recognition.merge_goals import resolve
from stash_ai.recognition.merge_solver import solve
from stash_ai.recognition.models import namespace
from stash_ai.recognition.rebuild import rebuild_selected
from stash_ai.recognition.scanning import scan_identity
from stash_ai.recognition.visibility import CONTRACT_ID
from stash_ai.recognition.scratch_space import directory, scope


def features(actors,dimensions):
    rng = np.random.default_rng(dimensions)
    vectors = rng.normal(size=(actors,3,4,dimensions))
    vectors *= .12
    baseline = rng.normal(size=(actors,dimensions))
    baseline[1::2] += .65*baseline[::2]
    vectors += baseline[:,None,None,:]+.25*rng.normal(size=(actors,3,1,dimensions))
    return (vectors/np.linalg.norm(vectors,axis=-1,keepdims=True)).astype('<f4')


def seed_scene(db,config,number,cues,faces,bodies):
    scan = str(uuid.uuid5(uuid.NAMESPACE_URL,'fragment-benchmark-'+str(number)))
    ns = namespace(config)
    rng = np.random.default_rng(number)
    actors = rng.choice(len(faces),2,replace=False)
    clothes = rng.integers(0,3,size=2)
    if number%4==0:
        actors[1]=int(actors[0])^1
    body_contract = contract(config)
    db.execute('INSERT INTO scans VALUES (?,?,?,?,?,?,?,?,?,?,?)',(scan,'benchmark',str(number),ns,scan,
               scan_identity(config,ns),'{}','{}',cues,cues,'complete'))
    seconds = 0.
    for cue in range(cues):
        seconds += int(rng.integers(1,26))
        records = [('',{'_body_cue':(scan,body_contract,cue)},None,None)]
        if cue%10<7:
            for side,actor in enumerate(actors):
                actor = int(actor)
                view = (cue//3+side)%4
                x = 20+side*130
                face = str(uuid.uuid5(uuid.UUID(scan),f'face-{cue}-{side}'))
                oid = str(uuid.uuid5(uuid.UUID(scan),f'body-{cue}-{side}'))
                # Keep detected face rows on rear views, with no usable embedding.
                rear = view==2
                quality = encoded(dict(pixels=96,sharpness=120,pose_deviation=.05,view_shot=cue//25,
                                       visible_face_fraction=.96,visibility=dict(contract=CONTRACT_ID)))
                rect = [x,12,48,48]
                geometry = rect+[x+14,28,x+34,28,x+24,37,x+16,48,x+32,48,.99]
                db.execute('INSERT INTO observations VALUES (?,?,?,?,?,?,?,?,?,?,?,?)',
                           (face,scan,cue,seconds,'vtt',encoded(rect),encoded(geometry),quality,
                            'unusable_rear_face' if rear else None,512,
                            None if rear else faces[actor,clothes[side],view].tobytes(),None))
                points = [[x+27,25]]*17
                points = list(points)
                for joint,dx,y in ((5,10,44),(6,42,44),(11,15,93),(12,38,93)):
                    points[joint]=[x+(52-dx if rear else dx),y]
                if rear:
                    points[1:3]=[[-1,-1],[-1,-1]]
                person=dict(bounds=[x,8,x+54,125],keypoints=points,score=.99)
                q=dict(keypoints=15,width=54,height=117,sharpness=120.,overlap=0.,detected_people=2,
                       pixel_digest=oid,view_shot=cue//25,frame_signature=('11' if cue//25%2==0 else 'ee')*48,
                       orientation=('front','side','back','side')[view])
                body=dict(scan=scan,cue=cue,seconds=seconds,contract=body_contract,
                          face=None if rear else face,face_accepted=not rear,rejected=None)
                records.append((oid,dict(_body=body,person=person,quality=q,benchmark_truth=actor),
                                bodies[actor,clothes[side],view],None))
        save_records(db,config,records)
    db.commit()
    return dict(scene=str(number),scan=scan)


def partition(db):
    h = hashlib.sha256()
    owners = {}
    assigned = defaultdict(Counter)
    n = missing = conflicts = 0
    rows = db.execute("""SELECT b.id,json_extract(b.quality,'$.benchmark_truth'),m.person,f.person,m.status
        FROM body_observations b LEFT JOIN body_members m ON m.observation=b.id
        LEFT JOIN members f ON f.observation=b.face ORDER BY b.id""")
    for oid,truth,body,face,status in rows:
        decisions = [owners.setdefault(p,len(owners)) if p is not None else None for p in (body,face)]
        h.update(encoded([oid,decisions,status]).encode())
        physical = {face,body if status=='accepted' else None}-{None}
        n += 1
        missing += not physical
        conflicts += len(physical)>1
        if len(physical)==1:
            assigned[truth][physical.pop()] += 1
    return dict(sha256=h.hexdigest(),physical_observations=n,withheld=missing,projection_conflicts=conflicts,
                extra_fragments=sum(max(0,len(people)-1) for people in assigned.values()),
                false_joins=sum(len(truths)-1 for truths in (
                    [truth for truth,counts in assigned.items() if owner in counts]
                    for owner in set().union(*(set(counts) for counts in assigned.values())))))


def measure(state,items,config):
    before = process_io()
    with sampled_rss() as rss:
        start=time.perf_counter()
        rebuild_selected(state,items,config)
        seconds=time.perf_counter()-start
    profile = state.last_rebuild_profile
    return dict(seconds=seconds,stages={k:v['seconds'] for k,v in profile['stages'].items()
                                       if k in ('body_scene_planning','body_global_reconciliation')},
                peak_rss_bytes=rss['sampled_peak_bytes'],
                sqlite_selects=sum(v['calls'] for k,v in profile['sql'].items() if k.startswith('select:')),
                sqlite_rows_fetched=sum(v['rows'] for v in profile['sql_fetch'].values()),
                process_io={k:v-before[k] for k,v in process_io().items() if k in before})


def run(config,size,cues,batch,trials):
    if min(size,cues,batch,trials)<1:
        raise ValueError('Scene, cue, batch and trial counts must be positive.')
    data = plain(config.data)
    data['sapiensid']['enabled'] = data['sapiensid']['collection']['enabled'] = True
    data['sapiensid']['identities'].update(enabled=True,tracklet_matching=True,scene_bonus_mode='evidence')
    data['sapiensid']['global_matching']['enabled'] = True
    data['execution']['profile_rebuild'] = True
    validate(data)
    faces,bodies = (features(48,n) for n in (512,4096))
    result = dict(scenes=size,cues_per_scene=cues,scenarios={})
    with scope(config):
        for mode in ('cold','incremental'):
            with directory(prefix='fragment-pipeline-') as folder:
                cfg=replace(config,data=data,database=Path(folder)/'benchmark.sqlite3')
                with FaceDatabase(cfg.database) as state:
                    selected=[]
                    timings=[]
                    for n in range(1,size+1):
                        selected.append(seed_scene(state.connection,cfg,n,cues,faces,bodies))
                        if mode=='incremental' and (n%batch==0 or n==size):
                            first=((n-1)//batch)*batch
                            timings.append(measure(state,selected[first:n],cfg))
                    if mode=='cold':
                        timings.append(measure(state,selected,cfg))
                    report = result['scenarios'][mode] = dict(rebuilds=timings,partition=partition(state.connection))
                    if mode=='incremental':
                        report['complete_selection']=measure(state,selected,cfg)
                        report['reconciled']=partition(state.connection)
                    else:
                        owners = [p for p, in state.connection.execute('SELECT person FROM members GROUP BY person '
                                                                        'ORDER BY COUNT(*),person LIMIT 2')]
                if mode=='cold':
                    if len(owners)<2:
                        raise ValueError('The benchmark needs two accepted face identities for solver goals.')
                    with closing(_open_readonly(cfg.database)) as db, sampled_rss() as memory:
                        selects = Counter()
                        db.set_trace_callback(lambda sql: selects.update(['select']) if sql.lstrip().lower().startswith('select') else None)
                        before=process_io()
                        goals=resolve(db,cfg,[[p] for p in owners],by_performer=False,policy='identity-physical')
                        start=time.perf_counter()
                        solved=solve(cfg,goals,db,maximum=trials,compact_report=True)
                        report['solver']=dict(seconds=time.perf_counter()-start,peak_rss_bytes=memory['sampled_peak_bytes'],
                                              attempted=solved['search']['attempted'],selected=solved['selected_trial'],
                                              source_selects=selects['select'],process_io={k:v-before[k] for k,v in
                                                  process_io().items() if k in before})
    cold=result['scenarios']['cold']['partition']['sha256']
    staged=result['scenarios']['incremental']
    result['incremental_equals_cold']=staged['partition']['sha256']==cold
    result['reconciled_equals_cold']=staged['reconciled']['sha256']==cold
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,required=True)
    parser.add_argument('--scenes',type=int,nargs='+',default=[32,128,1000,10000])
    parser.add_argument('--cues',type=int,default=300)
    parser.add_argument('--batch',type=int,default=32)
    parser.add_argument('--solver-trials',type=int,default=4)
    args=parser.parse_args()
    config=load(args.config)
    for size in args.scenes:
        print(json.dumps(run(config,size,args.cues,args.batch,args.solver_trials)),flush=True)


if __name__=='__main__':
    main()
