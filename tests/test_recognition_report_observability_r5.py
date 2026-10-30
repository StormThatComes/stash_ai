"""Reporting must disclose cost and availability without changing decisions."""
import json
from threading import Event


def test_serial_heartbeat_survives_blocking_work_without_fabricated_progress():
    from stash_ai.recognition.solver_progress import heartbeat, ReplayProgress
    from stash_ai.recognition.progress import observe_updates, stage, work
    snapshots=[]; reported=Event()
    def callback(name, label, value):
        if value['stage']=='Blocking SQL' and value['work'] and value['work'][0]['completed']==3:
            snapshots.append(value)
            if len(snapshots)>=2:reported.set()
    with heartbeat('baseline', callback) as live:
        reporter=ReplayProgress('baseline', live)
        with observe_updates(reporter.update),stage('Blocking SQL'),work('Rows',3) as row:
            row.advance(3)
            assert reported.wait(5)
    assert snapshots[0]['updated']==snapshots[1]['updated']
    assert snapshots[0]['last_completed']==snapshots[1]['last_completed']
    assert all(s['work'][0]['completed']==3 for s in snapshots)
    assert not live.thread.is_alive()


def test_plain_body_report_never_replays_and_simulation_deduplicates_baseline(tmp_path,monkeypatch):
    from tests.test_recognition_body_integration import seeded
    from stash_ai.recognition.merge_report import build
    from stash_ai.recognition import replay_simulation
    from stash_ai.recognition.config import digest
    config,_,_,people=seeded(tmp_path)
    calls=[];real=replay_simulation.rebuild_selected
    def rebuild(database, selected, settings, **kwargs):
        calls.append(digest(settings.data))
        return real(database,selected,settings,**kwargs)
    monkeypatch.setattr(replay_simulation,'rebuild_selected',rebuild)
    before=config.database.read_bytes()
    fast=build(config,*people[:2])
    assert calls==[] and fast['replay_trials']==0
    body=next(p for p in fast['paths'] if p['route']=='independent_person')
    assert body['verification_scope']=='measured_only' and not body['recommendations']
    assert body['proposals'] and not body['proposals'][0]['verification']['replay_verified']
    full=build(config,*people[:2],simulate=True)
    assert full['replay_trials']==len(calls)>0
    assert len(calls)==len(set(calls))
    assert config.database.read_bytes()==before


def test_body_reference_role_uses_person_quality_without_promoting_face_roles(tmp_path):
    from tests.test_recognition_merge_solver_hybrid import prepared
    from stash_ai.recognition.database import FaceDatabase
    from stash_ai.recognition.merge_goals import resolve,assess
    config,people=prepared(tmp_path, [[1.,0,0,0],[0,1.,0,0]], [[1.,0],[0,1.]], count=3)
    with FaceDatabase(config.database) as state:
        db=state.connection
        db.execute("UPDATE person_evidence SET metadata=json_set(metadata,'$.quality.sharpness',20)")
        db.commit()
        goals=resolve(db,config,[people],by_performer=False)
    rows=[r for m in goals['groups'][0]['members'] for r in m['observations']]
    assert all(r['sample_role']=='anchor' and r['body_role']=='auxiliary' for r in rows)
    observations={r['id']:r['scene'] for r in rows}; predicted=dict.fromkeys(observations,'merged')
    assignment=assess(goals,observations,{},predicted)['groups'][0]['sample_assignment']
    assert assignment['sapiensid']['anchor']['total']==len(rows)
    body=assignment['sapiensid_by_body_eligibility']
    assert body['anchor']['total']==0 and body['auxiliary']['total']==len(rows)
    assert body['reasons']['body_anchor_sharpness']==len(rows)


def test_publication_debt_is_scoped_bounded_and_read_only(tmp_path):
    from stash_ai.recognition.database import FaceDatabase,encoded
    from stash_ai.recognition.publication_queue import debt,key
    with FaceDatabase(tmp_path/'faces.sqlite3') as state:
        db=state.connection
        for scope,scene in [('s','1'),('s','2'),('s','3'),('other','4')]:
            db.execute('INSERT INTO metadata VALUES (?,?)',(key(scope,'ns',scene),encoded(dict(scope=scope,namespace='ns',scene=scene))))
        db.commit();before=db.total_changes
        result=debt(db,'s','ns',['1'],limit=1)
        assert result['pending_selected']==1 and result['pending_outside_selection']==2
        assert len(result['outside_scene_ids'])==1 and result['truncated']
        assert result['next_step']=='recognition sync-identities' and db.total_changes==before


def test_solver_profile_includes_copy_feedback_and_process_memory(tmp_path):
    from benchmarks.benchmark_merge_solver import fixture
    from stash_ai.recognition.fragmentation_report import _open_readonly
    from stash_ai.recognition.merge_solver import solve
    config,goals=fixture(tmp_path/'input','disjoint')
    with _open_readonly(config.database) as db:
        result=solve(config,goals,db,maximum=2,execution=dict(cpus=2,trials=1,profile=True))
    profile=result['performance']
    assert profile['elapsed_seconds']>=0 and 'process lifetime' in profile['rss_scope']
    simulation=result['simulation']
    assert simulation['performance']['input_preparation_seconds']>=0
    assert simulation['performance']['consume_seconds']>=0
    for trial in simulation['trials'].values():
        stages=trial['performance']['trial']['stages']
        assert set(stages)=={'Prepare replay','Rebuild selected identities','Collect trial predictions','Collect solver feedback'}
        assert all(v['elapsed_seconds']>=0 and v['cpu_seconds']>=0 for v in stages.values())
    json.dumps(result,allow_nan=False)
