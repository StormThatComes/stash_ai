"""Weak outsiders compete by margin without becoming transitive merge members."""
import math,json
import numpy as np
import pytest
from stash_ai.recognition.database import FaceDatabase
from stash_ai.recognition.clustering import matching_policy
from stash_ai.recognition.global_views import consolidation_plan,merge_views
from tests.test_recognition_anchor_views import config_for,A,X,C
from tests.test_recognition_global_views import seed,views


def vector(degrees):
    angle=math.radians(degrees);return np.array([math.cos(angle),math.sin(angle),0.,0.])


@pytest.mark.parametrize('extra',[False,True])
def test_uncompetitive_weak_neighbour_does_not_veto_supported_pair(tmp_path,extra):
    config=config_for(tmp_path)
    with FaceDatabase(config.database) as db:
        _,owner=seed(db,config,'1',A)
        _,part=seed(db,config,'2',X)
        _,fragment=seed(db,config,'3',C)
        db.connection.execute('UPDATE members SET person=? WHERE person=?',(owner,part))
        db.connection.execute('UPDATE candidates SET person=? WHERE person=?',(owner,part))
        if extra:seed(db,config,'4',vector(114))
        frozen=views(db,config)
        before=[tuple(r) for r in db.connection.execute('SELECT * FROM members ORDER BY observation')]
        plan=consolidation_plan(db.connection,frozen,config)
        assert [tuple(r) for r in db.connection.execute('SELECT * FROM members ORDER BY observation')]==before
        decision=next(p for p in plan if owner in p['people'])
        assert decision['outcome']=='merge' and decision['component_size']==2
        if extra:assert decision['nearest_rival']['gap']>.12
        merge_views(db,frozen,config,matching_policy(config))
        assert db.connection.execute('SELECT COUNT(DISTINCT person) FROM members').fetchone()[0]==(2 if extra else 1)
        proof=json.loads(db.connection.execute('SELECT evidence FROM identity_merges').fetchone()[0])
        assert proof['format']==3 and proof['complete_component']


def test_near_threshold_competitor_still_vetoes_pair(tmp_path):
    config=config_for(tmp_path)
    with FaceDatabase(config.database) as db:
        for scene,angle in enumerate((0,32,68),1):seed(db,config,str(scene),vector(angle))
        frozen=views(db,config)
        decision=next(p for p in consolidation_plan(db.connection,frozen,config) if p['component_size']==2)
        assert decision['blocker']=='competing_view' and decision['nearest_rival']['gap']<.12
        merge_views(db,frozen,config,matching_policy(config))
        assert not db.connection.execute('SELECT 1 FROM identity_merges').fetchone()


def test_high_scoring_but_under_supported_outsider_remains_competition(tmp_path):
    config=config_for(tmp_path,view_min_support=2)
    with FaceDatabase(config.database) as db:
        from tests.test_recognition_rebuild import changed
        from tests.test_recognition_clustering import scan
        from stash_ai.recognition.clustering import cluster_scene
        strict=changed(config,recognition={'match_threshold':.99999,'ambiguity_margin':.000001})
        for scene,angle in enumerate((0,12,6),1):
            sid,_=scan(db,strict,str(scene),[vector(angle)]*3,count=3)
            assert cluster_scene(db,sid,strict)[0]['status']=='accepted'
        db.connection.execute('DELETE FROM members WHERE observation IN (SELECT id FROM observations WHERE scan IN '
            "(SELECT id FROM scans WHERE scene='3') ORDER BY cue LIMIT 1)")
        frozen=views(db,config)
        decision=next(p for p in consolidation_plan(db.connection,frozen,config) if p['component_size']==2)
        assert decision['blocker']=='competing_view'
        merge_views(db,frozen,config,matching_policy(config))
        assert not db.connection.execute('SELECT 1 FROM identity_merges').fetchone()


def test_scene_report_exposes_the_same_frozen_component_without_writes(tmp_path):
    import uuid
    from stash_ai.recognition.fragmentation_report import build
    from stash_ai.recognition.models import namespace
    from stash_ai.recognition.scanning import scan_identity
    config=config_for(tmp_path)
    with FaceDatabase(config.database) as db:
        first,owner=seed(db,config,'1',A)
        _,part=seed(db,config,'2',X)
        fragment_scan,fragment=seed(db,config,'3',C)
        seed(db,config,'4',vector(114))
        db.connection.execute('UPDATE members SET person=? WHERE person=?',(owner,part))
        db.connection.execute('UPDATE candidates SET person=? WHERE person=?',(owner,part))
        obs=db.connection.execute('SELECT id FROM observations WHERE scan=? ORDER BY cue LIMIT 1',(fragment_scan,)).fetchone()[0]
        db.connection.execute('UPDATE observations SET scan=?,cue=50,seconds=50 WHERE id=?',(first,obs))
        ns,policy=namespace(config),matching_policy(config)
        db.connection.execute('UPDATE scans SET namespace=?,policy_sha=?',(ns,scan_identity(config,ns)))
        db.connection.execute('UPDATE people SET namespace=?',(ns,))
        for row in db.connection.execute('SELECT id,payload FROM candidates').fetchall():
            payload=json.loads(row['payload']);observation=payload['observation'];payload['policy']=policy
            scan=db.connection.execute('SELECT scan FROM observations WHERE id=?',(observation,)).fetchone()[0]
            db.connection.execute('UPDATE candidates SET id=?,scan=?,policy=?,payload=? WHERE id=?',
                (str(uuid.uuid5(uuid.UUID(observation),policy)),scan,policy,json.dumps(payload),row['id']))
        db.connection.commit()
    before=config.database.read_bytes()
    report=build(config,'1')
    assert config.database.read_bytes()==before
    pair=report['potential_fragments'][0]
    assert pair['global_view_component']['outcome']=='merge'
    assert pair['global_view_component']['nearest_rival']['gap']>.12
    assert pair['confirmed_view_evidence']['blockers']==[]
    assert pair['can_merge'] is None  # Read-only evidence is not publication.
