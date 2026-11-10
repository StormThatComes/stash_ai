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


def seed_scene(db,config,number,cues,faces,bodies, *, occupancy='periodic', varied_crops=False):
    scan = str(uuid.uuid5(uuid.NAMESPACE_URL,'fragment-benchmark-'+str(number)))
    ns = namespace(config)
    rng = np.random.default_rng(number)
    actors = rng.choice(len(faces),2,replace=False)
    clothes = rng.integers(0,3,size=2)
    if number%4==0:
        actors[1]=int(actors[0])^1
    entrant = next(i for i in range(len(faces)) if i not in actors)
    body_contract = contract(config)
    db.execute('INSERT INTO scans VALUES (?,?,?,?,?,?,?,?,?,?,?)',(scan,'benchmark',str(number),ns,scan,
               scan_identity(config,ns),'{}','{}',cues,cues,'complete'))
    crop_rng = np.random.default_rng(number+813)
    occupied = {i for i in range(cues) if i%10<7}
    if occupancy=='spread':
        occupied = {i for i in range(cues) if i%10 not in (2,5,8)}
    elif occupancy=='burst':
        occupied = {i for i in range(cues) if i%30<21}
    elif occupancy=='random':
        occupied = set(np.random.default_rng(number+413).choice(cues,len(occupied),replace=False))
    seconds = 0.
    for cue in range(cues):
        seconds += int(rng.integers(1,26))
        records = [('',{'_body_cue':(scan,body_contract,cue)},None,None)]
        if cue in occupied:
            for side,actor in enumerate(actors):
                actor = entrant if side == 0 and 52 <= cue%100 < 71 else int(actor)
                view = (cue//3+side)%4
                position = side if cue//25 % 2 == 0 else 1-side
                x = 20 + position*(28 if cue%25 in (4,5) else 130)
                face = str(uuid.uuid5(uuid.UUID(scan),f'face-{cue}-{side}'))
                oid = str(uuid.uuid5(uuid.UUID(scan),f'body-{cue}-{side}'))
                # Keep detected face rows on rear views, with no usable embedding.
                rear = view==2
                quality = encoded(dict(pixels=96,sharpness=120,pose_deviation=.05,view_shot=cue-cue%25,
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
                    width = dx if rear else 52-dx
                    if view in (1,3):
                        width = 26 + (width-26)*.25
                    points[joint]=[x+width,y]
                if rear:
                    points[1:3]=[[-1,-1],[-1,-1]]
                person=dict(bounds=[x,8,x+54,125],keypoints=points,score=.99)
                q=dict(keypoints=15,width=54,height=117,sharpness=120.,overlap=.48 if cue%25 in (4,5) else 0.,detected_people=2,
                       pixel_digest=oid,view_shot=cue-cue%25,frame_signature=('112233' if cue//25%2==0 else '8833bb')*16,
                       orientation=('front','side','back','side')[view])
                body=dict(scan=scan,cue=cue,seconds=seconds,contract=body_contract,
                          face=None if rear else face,face_accepted=not rear,rejected=None)
                outfit = (clothes[side]+cue//100)%3 if varied_crops else clothes[side]
                feature = bodies[actor, outfit, view]
                if varied_crops:
                    # Shared garment component for the pair plus independent crop noise.
                    garment = bodies[0,outfit,0]
                    noise = crop_rng.normal(size=feature.shape).astype('<f4')
                    feature = feature + .35*garment + .14*noise/np.linalg.norm(noise)
                    feature = feature/np.linalg.norm(feature)
                    q.update(appearance_hash=np.packbits(feature[:64]>0).tobytes().hex(),
                             pixel_digest=hashlib.sha256(feature.tobytes()).hexdigest())
                if rear and side == 0:
                    # A change of pose leaves a near-threshold positive for the solver.
                    feature = .77*feature + .64*np.roll(feature, 41)
                    feature = feature/np.linalg.norm(feature)
                records.append((oid,dict(_body=body,person=person,quality=q,benchmark_truth=actor),feature,None))
        save_records(db,config,records)
    db.execute('INSERT INTO metadata(key,value) VALUES (?,?)',
               ('cue-view-shots:'+scan, encoded([[cue,cue] for cue in range(0,cues,25)])))
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
    by_owner = Counter()
    true_pairs = 0
    for counts in assigned.values():
        by_owner.update(counts)
        true_pairs += sum(v*(v-1)//2 for v in counts.values())
    return dict(sha256=h.hexdigest(),physical_observations=n,withheld=missing,projection_conflicts=conflicts,
                extra_fragments=sum(max(0,len(people)-1) for people in assigned.values()),
                false_merge_pairs=sum(v*(v-1)//2 for v in by_owner.values())-true_pairs)


def freeze_challenge(db, config, scan):
    """Freeze manually labelled front/rear fragments in this benchmark database."""
    members = ['benchmark-front', 'benchmark-rear', 'benchmark-partner']
    sample = lambda cue, side: str(uuid.uuid5(uuid.UUID(scan), f'body-{cue}-{side}'))
    truth_sql = "SELECT json_extract(quality,'$.benchmark_truth') FROM body_observations WHERE id=?"
    truth, partner = (db.execute(truth_sql, (sample(0, side),)).fetchone()[0] for side in (0, 1))
    for name, performer in zip(members, ('900001', None, '900002')):
        db.execute('INSERT INTO people VALUES (?,?,?,?,?,?)',
                   (name, 'benchmark', namespace(config), 0, performer, name))
    rows = list(db.execute("""SELECT id,face,json_extract(quality,'$.benchmark_truth'),
        json_extract(quality,'$.quality.orientation') FROM body_observations WHERE scan=?""",(scan,)))
    count = Counter()
    for oid,face,actor,orientation in rows:
        if actor not in (truth, partner):
            continue
        owner = members[2] if actor == partner else members[int(orientation == 'back')]
        count[owner] += 1
        db.execute("UPDATE body_members SET person=?,status='accepted' WHERE observation=?", (owner,oid))
        if face:
            db.execute('UPDATE members SET person=? WHERE observation=?',(owner,face))
    if not all(count[p] for p in members):
        return None
    return resolve(db, config, [members[:2], members[2:]], by_performer=False,
                   policy='identity-physical', observation_exclusions=[('900001', sample(0,1))])


def natural_goals(db, config, case):
    """Label fixed synthetic observations against their independently face-rooted owner."""
    if case == 'contact':
        selection = "s.scene IN ('1','2') AND b.cue BETWEEN 27 AND 29"
        truth = "((s.scene='1' AND actor=24) OR (s.scene='2' AND actor=39))"
    else:
        selection = "s.scene IN ('9','10')"
        truth = 'actor=0'
    rows = list(db.execute(f"""SELECT b.id,b.scan,b.cue,s.scene,
        json_extract(b.quality,'$.benchmark_truth') actor FROM body_observations b
        JOIN scans s ON s.id=b.scan WHERE {selection} AND {truth}
        AND json_extract(b.quality,'$.quality.orientation')='back' ORDER BY actor,s.scene,b.cue"""))
    if not rows:
        raise ValueError('No benchmark labels for the selected natural challenge.')
    groups, positives, negatives = [], [], []
    for actor in sorted({row['actor'] for row in rows}):
        owner = db.execute("""SELECT f.person FROM body_observations b JOIN members f ON f.observation=b.face
            WHERE json_extract(b.quality,'$.benchmark_truth')=? GROUP BY f.person
            ORDER BY count(*) DESC,f.person LIMIT 1""",(actor,)).fetchone()[0]
        performer = str(910000+actor)
        db.execute('UPDATE people SET performer=? WHERE id=?',(performer,owner))
        groups.append([owner])
        for row in rows:
            if row['actor'] != actor:
                continue
            positives.append((performer,row['id']))
            partners = db.execute("""SELECT id FROM body_observations WHERE scan=? AND cue=?
                AND json_extract(quality,'$.benchmark_truth')<>?""",(row['scan'],row['cue'],actor))
            negatives.extend((performer,other) for other, in partners)
    return resolve(db,config,groups,by_performer=False,policy='identity-physical',
                   observation_inclusions=positives,observation_exclusions=negatives)


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


def run(config,size,cues,batch,trials,*,occupancy='periodic',varied_crops=False,natural=None):
    if min(size,cues,batch,trials)<1:
        raise ValueError('Scene, cue, batch and trial counts must be positive.')
    data = plain(config.data)
    data['sapiensid']['enabled'] = data['sapiensid']['collection']['enabled'] = True
    data['sapiensid']['identities'].update(enabled=True,tracklet_matching=True,scene_bonus_mode='evidence')
    data['sapiensid']['global_matching']['enabled'] = True
    data['execution']['profile_rebuild'] = True
    validate(data)
    faces,bodies = (features(48,n) for n in (512,4096))
    result = dict(scenes=size,cues_per_scene=cues,occupancy=occupancy,varied_crops=varied_crops,scenarios={})
    with scope(config):
        for mode in ('cold','incremental'):
            with directory(prefix='fragment-pipeline-') as folder:
                cfg=replace(config,data=data,database=Path(folder)/'benchmark.sqlite3')
                with FaceDatabase(cfg.database) as state:
                    selected=[]
                    timings=[]
                    for n in range(1,size+1):
                        number = n+6 if natural=='rear' else n
                        selected.append(seed_scene(state.connection,cfg,number,cues,faces,bodies,occupancy=occupancy,varied_crops=varied_crops))
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
                        goals = (natural_goals(state.connection,cfg,natural) if natural else
                                 freeze_challenge(state.connection, cfg, selected[0]['scan']))
                        state.connection.commit()
                if mode=='cold':
                    if goals is None:
                        report['solver'] = dict(status='insufficient_front_rear_coverage')
                        continue
                    with closing(_open_readonly(cfg.database)) as db, sampled_rss() as memory:
                        selects = Counter()
                        db.set_trace_callback(lambda sql: selects.update(['select']) if sql.lstrip().lower().startswith('select') else None)
                        before=process_io()
                        start=time.perf_counter()
                        solved=solve(cfg,goals,db,maximum=trials,compact_report=True)
                        report['solver']=dict(seconds=time.perf_counter()-start,peak_rss_bytes=memory['sampled_peak_bytes'],
                                              attempted=solved['search']['attempted'],selected=solved['selected_trial'],
                                              baseline=solved['simulation']['trials']['baseline']['goal_result']['fully_solved'],
                                              fully_solved=solved['best_result']['fully_solved'],
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
    parser.add_argument('--occupancy',choices=('periodic','spread','burst','random'),default='periodic')
    parser.add_argument('--varied-crops',action='store_true',help='Vary appearance, clothing and retained crop fingerprints.')
    parser.add_argument('--real-holdout',type=Path,help='Evaluate a manually labelled VTT manifest and its retained database.')
    parser.add_argument('--scope',help='Stash scope for the real holdout.')
    parser.add_argument('--natural',choices=('contact','rear'),help='Replay frozen natural splits on two or four 300-cue random scenes.')
    args=parser.parse_args()
    config=load(args.config)
    if args.real_holdout:
        import sqlite3
        from stash_ai.recognition.solver_validation import validate_winner
        if not args.scope:
            parser.error('--scope is required with --real-holdout')
        with closing(sqlite3.connect(':memory:')) as training:
            training.execute('CREATE TABLE scans(scope,namespace,scene)')
            result = validate_winner(config,training,dict(scope=args.scope,namespace=namespace(config),
                best_result=dict(changes=[])),args.real_holdout,compare_strategies=True)
        print(json.dumps(result),flush=True)
        return
    if args.natural:
        size = 2 if args.natural=='contact' else 4
        print(json.dumps(run(config,size,300,args.batch,args.solver_trials,occupancy='random',
                             varied_crops=True,natural=args.natural)),flush=True)
        return
    for size in args.scenes:
        print(json.dumps(run(config,size,args.cues,args.batch,args.solver_trials,occupancy=args.occupancy,varied_crops=args.varied_crops)),flush=True)


if __name__=='__main__':
    main()
