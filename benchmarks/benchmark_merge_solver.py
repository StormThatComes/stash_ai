"""Deterministic retained-evidence solver benchmark; no models or network."""
import argparse
from dataclasses import replace
import json
import math
from pathlib import Path
import tempfile
import time
import uuid

import numpy as np
import yaml
from stash_ai.recognition.config import load, plain
from stash_ai.recognition.database import FaceDatabase
from stash_ai.recognition.models import namespace
from stash_ai.recognition.scanning import scan_identity
from stash_ai.recognition.clustering import matching_policy
from stash_ai.recognition.person_signal import save, contract
from stash_ai.recognition.merge_goals import resolve, objective
from stash_ai.recognition.merge_solver import solve

CASES = ('face', 'query', 'disjoint', 'missing_body', 'quality', 'body_modes')


def fixture(root, case):
    root.mkdir(parents=True, exist_ok=True)
    path = root/'config.yml'
    path.write_text(yaml.safe_dump(dict(version=1,recognition=dict(match_threshold=.99),
        clustering=dict(cohesion_threshold=.85,max_anchor_pose_deviation=.2,anchor_view_matching=True))))
    config = load(path, database=root/'faces.sqlite3')
    data = plain(config.data)
    data['sapiensid']['enabled'] = case not in ('face','quality')
    data['sapiensid']['matching'].update(face_floor=.8,person_threshold=.99,min_support=4)
    if case == 'quality':
        data['clustering'].update(min_anchor_face_pixels=100,min_auxiliary_face_pixels=100)
    config = replace(config,data=data)
    count = 8 if case == 'body_modes' else 3
    n = 4 if case == 'disjoint' else 2
    people = [str(uuid.uuid5(uuid.NAMESPACE_URL,f'solver-{i}')) for i in range(n)]
    with FaceDatabase(config.database) as state:
        db = state.connection
        for i,p in enumerate(people):
            scan = str(uuid.uuid5(uuid.NAMESPACE_URL,f'solver-scan-{i}'))
            db.execute('INSERT INTO people VALUES (?,?,?,?,?,?)',(p,'scope',namespace(config),i,str(900+i),'benchmark-'+p))
            db.execute('INSERT INTO scans VALUES (?,?,?,?,?,?,?,?,?,?,?)',(scan,'scope',str(i+1),namespace(config),
                'input',scan_identity(config,namespace(config)),'{}','{}',count,count,'complete'))
            score = 1. if case == 'quality' else .431237 if case in ('query','missing_body') else .7
            face = np.zeros(4,dtype='<f4'); offset = (i//2)*2
            face[offset] = 1 if i%2 == 0 else score
            if i%2:face[offset+1] = math.sqrt(1-score**2)
            for cue in range(count):
                oid = str(uuid.uuid5(uuid.UUID(scan),str(cue)))
                quality = dict(pose_deviation=.4 if i%2 and case in ('query','disjoint','missing_body') else .1,
                               pixels=80,sharpness=30)
                db.execute('INSERT INTO observations VALUES (?,?,?,?,?,?,?,?,?,?,?,?)',(oid,scan,cue,float(cue),
                    'benchmark','[]','[]',json.dumps(quality),None,4,face.tobytes(),None))
                db.execute('INSERT INTO members VALUES (?,?)',(oid,p))
                db.execute('INSERT INTO candidates VALUES (?,?,?,?,?,?)',(str(uuid.uuid5(uuid.UUID(oid),'candidate')),
                    scan,matching_policy(config),p,'accepted',json.dumps(dict(observation=oid,reason='candidate_group'))))
                if data['sapiensid']['enabled'] and case != 'missing_body':
                    body = np.zeros(4096,dtype='<f4')
                    offset = 2 if case == 'body_modes' and cue>=4 else offset
                    body[offset] = 1 if i%2 == 0 else .96
                    if i%2:body[offset+1] = .28
                    metadata=dict(person=dict(bounds=[0,0,512,512],score=1.,keypoints=[[100.,100.]]*17),
                        quality=dict(keypoints=17,sharpness=100.,width=512,height=512,overlap=0.,pixel_digest=oid))
                    save(db,oid,contract(config),metadata,body)
        db.commit()
        groups = [people[i:i+2] for i in range(0,n,2)]
        goals = resolve(db,config,groups,by_performer=False)
    return config,goals


def measure(config, goals, maximum, **solver_options):
    from stash_ai.recognition.fragmentation_report import _open_readonly
    from contextlib import closing
    with closing(_open_readonly(config.database)) as db:
        started = time.perf_counter()
        result = solve(config,goals,db,maximum=maximum,**solver_options)
        elapsed = time.perf_counter()-started
    best, cumulative, history = None, 0., []
    for number,(name,t) in enumerate(result['simulation']['trials'].items()):
        cumulative += t['elapsed_seconds']
        rank = objective(t['goal_result'])
        best = max(best,rank) if best is not None else rank
        history.append(dict(trial=name,completed=number+1,seconds=t.get('completed_wall_seconds',cumulative),best_objective=list(best),
            admissible=t['goal_result']['admissible'],fully_solved=t['goal_result']['fully_solved'],
            disjoint=t['goal_result']['disjoint']))
    return dict(seconds=elapsed,selected=result['selected_trial'],quality=result['status'],search=result['search'],
        history=history,first_solution=next((t for t in history if t['fully_solved']),None),
        milestones={str(b): next((t for t in reversed(history) if t['completed']<=b),None) for b in (10,50,100)})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--max-trials',type=int,default=16)
    parser.add_argument('--cases',nargs='+',choices=CASES,default=CASES)
    parser.add_argument('--sampler',choices=('heuristic','tpe','compare'),default='heuristic')
    parser.add_argument('--seed',type=int,default=0)
    parser.add_argument('--targeted-trials',type=int,default=8)
    args = parser.parse_args()
    if args.max_trials<1:parser.error('max-trials must be positive')
    if args.targeted_trials<0:parser.error('targeted-trials must be nonnegative')
    with tempfile.TemporaryDirectory() as tmp:
        modes=('heuristic','tpe') if args.sampler=='compare' else (args.sampler,)
        results={mode:{case:measure(*fixture(Path(tmp)/mode/case,case),args.max_trials,
                    sampler=mode,seed=args.seed,targeted_trials=args.targeted_trials) for case in args.cases} for mode in modes}
    print(json.dumps(dict(format=1,scope='deterministic_synthetic_retained_replay',results=results),indent=2))


if __name__ == '__main__':main()
