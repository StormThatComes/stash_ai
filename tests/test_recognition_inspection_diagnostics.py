"""Diagnostics preserve provenance and never turn missing evidence into facts."""
from contextlib import closing
from dataclasses import replace
import json
from threading import Event

import numpy as np
import pytest

from tests.test_recognition_identity_inspector import scene_fixture
from stash_ai.recognition.database import FaceDatabase
from stash_ai.recognition.identity_inspection import IdentityReader
from stash_ai.recognition.scene_inspection import SceneReader
from stash_ai.recognition.inspection_diagnostics import stages, contribution, retained_rivals, review_queue, unassigned_summary
from stash_ai.recognition.inspection_candidates import calculate
from stash_ai.recognition.inspection_search import SearchReader


def test_decision_chain_provenance_reference_selection_and_unknown_rival(tmp_path):
    config,person=scene_fixture(tmp_path)
    reader=IdentityReader(config,identity_id=person)
    address=reader.page(kind='face',state='anchor')[0][0]['id']
    with FaceDatabase(config.database) as state:
        state.connection.execute("UPDATE candidates SET payload=json_set(payload,'$.reference_role','strict','$.reference_selected',json('true')) WHERE json_extract(payload,'$.observation')=?",(address,))
        state.connection.commit()
    detail=reader.detail('face',address,pixels=False)
    result=stages(detail)
    assert any('historical settings unavailable' in stage['source'] for stage in result)
    assert result[-1]['label']=='Membership' and result[-1]['status']=='accepted'
    text=contribution(detail)
    assert 'bank selection yes' in text and 'not a count of later comparisons' in text
    rivals=retained_rivals(detail)
    assert rivals[0]['score']==.61 and rivals[0]['identity'] is None
    decision=detail['trace']['decisions'][0]
    decision['metrics']['rival_cluster']='retained-owner'
    assert retained_rivals(detail)[0]['identity']=='retained-owner'
    del decision['reference_selected']
    assert 'bank selection not retained' in contribution(detail)


def test_review_queue_searches_complete_scope_before_limit_without_crops(tmp_path,monkeypatch):
    config,person=scene_fixture(tmp_path)
    scene=SceneReader(config,'2')
    monkeypatch.setattr('stash_ai.recognition.identity_inspection.cold_crop',lambda *_:pytest.fail('Decoded pixels'))
    with FaceDatabase(config.database) as state:
        # Highest-priority sample is in another scene, outside any expanded tree page.
        row=state.connection.execute("SELECT o.id FROM observations o JOIN scans s ON s.id=o.scan JOIN members m ON m.observation=o.id WHERE s.scene='10' AND m.person=? LIMIT 1",(person,)).fetchone()
        address=row[0]
        state.connection.execute("UPDATE observations SET rejected='retained-quality-rejection' WHERE id=?",(address,))
        state.connection.execute("UPDATE candidates SET payload=json_set(payload,'$.scores',json('[0.801,0.8]')) WHERE json_extract(payload,'$.observation')=?",(address,))
        state.connection.commit()
    before=config.database.read_bytes()
    result=review_queue(scene,limit=1)
    assert result['seen']==8 and result['flagged']>=1
    assert result['rows'][0]['id']==address and result['rows'][0]['scene']=='10'
    assert len(result['rows'][0]['reasons'])>=2
    assert config.database.read_bytes()==before and not result['changed']


def test_unassigned_summary_streams_only_metadata_and_filters_correct_membership(tmp_path,monkeypatch):
    from stash_ai.recognition import inspection_diagnostics as module
    config,_=scene_fixture(tmp_path);scene=SceneReader(config,'2')
    original=module.open_readonly
    def opened(path):
        import sqlite3
        db=original(path)
        def authorize(action,table,column,*_):
            if action==sqlite3.SQLITE_READ and column in ('crop','embedding'):pytest.fail('Summary loaded image/vector column')
            return sqlite3.SQLITE_OK
        db.set_authorizer(authorize);return db
    monkeypatch.setattr(module,'open_readonly',opened)
    result=unassigned_summary(scene)
    assert result['total']==5 and sum(r['count'] for r in result['states'])==5
    assert any(g['counts']['unknown'] for g in result['gates'])
    assert any(g['counts']['fail'] for g in result['gates'])
    rows=list(SearchReader(scene,dict(kind='body',state='provisional',assignment='unassigned')).iter_rows())
    assert [r['id'] for r in rows]==['provisional-body']
    assert all(r['status']!='accepted' for r in SearchReader(scene,dict(assignment='unassigned')).iter_rows())


def candidate_fixture(tmp_path):
    from stash_ai.recognition.models import namespace
    config,person=scene_fixture(tmp_path)
    reader=IdentityReader(config,identity_id=person)
    selected=reader.page(kind='face',state='anchor')[0][0]['id']
    with FaceDatabase(config.database) as state:
        db=state.connection
        quality=json.dumps(dict(pixels=300,sharpness=300,pose_deviation=.1,alignment_valid=True))
        db.execute('UPDATE observations SET embedding=?,quality=?',(np.array([1,0,0,0],dtype='<f4').tobytes(),quality))
        db.execute('INSERT INTO people VALUES (?,?,?,?,?,?)',('rival','scope',namespace(config),3,'901','rival-token'))
        template=dict(db.execute('SELECT * FROM observations WHERE id=?',(selected,)).fetchone())
        for index in range(2):
            row=dict(template,id='rival-'+str(index),cue=30+index,seconds=40.+index,embedding=np.array([.8,.6,0,0],dtype='<f4').tobytes())
            db.execute('INSERT INTO observations VALUES ('+','.join('?' for _ in row)+')',tuple(row.values()))
            db.execute('INSERT INTO members VALUES (?,?)',(row['id'],'rival'))
        db.commit()
    return config,person,selected


def test_new_face_comparison_is_scoped_excludes_self_and_exposes_witnesses(tmp_path,monkeypatch):
    config,person,selected=candidate_fixture(tmp_path)
    monkeypatch.setattr('stash_ai.recognition.models.load_backend',lambda *_:pytest.fail('Model inference'))
    before=config.database.read_bytes()
    result=calculate(config,'face',selected,'scope')
    assert 'not probabilities' in result['note'] and 'or a replay' in result['note']
    owner=next(r for r in result['rows'] if r['owner'])
    rival=next(r for r in result['rows'] if r['identity']=='rival')
    assert owner['identity']==person and owner['score']==pytest.approx(1)
    assert owner['witness']!=selected and rival['score']==pytest.approx(.8)
    assert owner['selected'] and owner['count']<=config.data['clustering']['max_samples_per_person']
    assert config.database.read_bytes()==before and not result['changed']
    with pytest.raises(ValueError,match='scope/model'):calculate(config,'face',selected,'wrong')


def test_missing_embedding_and_historical_body_contract_are_explicit(tmp_path):
    config,_=scene_fixture(tmp_path)
    result=calculate(config,'body','rejected-body','scope')
    assert not result['rows'] and 'No retained embedding' in result['note']
    with FaceDatabase(config.database) as state:
        vector=np.zeros(4096,dtype='<f4');vector[0]=1
        state.connection.execute("UPDATE body_observations SET embedding=?,contract='historical' WHERE id='accepted-body'",(vector.tobytes(),));state.connection.commit()
    result=calculate(config,'body','accepted-body','scope')
    assert not result['rows'] and 'Historical body contract' in result['note']


def test_body_candidates_only_use_accepted_anchors_and_same_contract(tmp_path):
    config,person=scene_fixture(tmp_path)
    with FaceDatabase(config.database) as state:
        db=state.connection;sample=np.zeros(4096,dtype='<f4');sample[0]=1
        db.execute('UPDATE body_observations SET embedding=?',(sample.tobytes(),))
        db.execute("UPDATE body_members SET role='anchor' WHERE observation='accepted-body'")
        row=dict(db.execute("SELECT * FROM body_observations WHERE id='accepted-body'").fetchone())
        row.update(id='reference-body',cue=10,seconds=50.)
        db.execute('INSERT INTO body_observations VALUES ('+','.join('?' for _ in row)+')',tuple(row.values()))
        db.execute('INSERT INTO body_members SELECT ?,person,role,status,policy,proof FROM body_members WHERE observation=?',('reference-body','accepted-body'));db.commit()
    result=calculate(config,'body','accepted-body','scope')
    owner=next(r for r in result['rows'] if r['owner'])
    assert owner['count']==1 and owner['witness']=='reference-body' and owner['score']==pytest.approx(1)
    assert 'not the production body-bank selection' in result['note']


def test_expensive_diagnostics_honor_cancellation(tmp_path):
    from stash_ai.recognition.inspection_runtime import local, Cancelled
    config,_,selected=candidate_fixture(tmp_path)
    cancel=Event();cancel.set();local.cancel=cancel
    try:
        for task in (lambda:calculate(config,'face',selected,'scope'),lambda:review_queue(SceneReader(config,'2')),lambda:unassigned_summary(SceneReader(config,'2'))):
            with pytest.raises(Cancelled):task()
    finally:del local.cancel


def test_candidates_exclude_foreign_scope_members_and_report_budget_limit(tmp_path):
    from stash_ai.recognition.config import plain
    from stash_ai.recognition.models import namespace
    config,person,selected=candidate_fixture(tmp_path)
    with FaceDatabase(config.database) as state:
        db=state.connection
        db.execute('INSERT INTO people VALUES (?,?,?,?,?,?)',('foreign','foreign-scope',namespace(config),4,'902','foreign-token'))
        scan=dict(db.execute('SELECT * FROM scans LIMIT 1').fetchone());scan.update(id='foreign-scan',scope='foreign-scope')
        db.execute('INSERT INTO scans VALUES ('+','.join('?' for _ in scan)+')',tuple(scan.values()))
        row=dict(db.execute('SELECT * FROM observations WHERE id=?',(selected,)).fetchone());row.update(id='foreign-sample',scan='foreign-scan')
        db.execute('INSERT INTO observations VALUES ('+','.join('?' for _ in row)+')',tuple(row.values()))
        db.execute('INSERT INTO members VALUES (?,?)',('foreign-sample','foreign'));db.commit()
    result=calculate(config,'face',selected,'scope')
    assert all(r['identity']!='foreign' for r in result['rows'])
    data=plain(config.data);data['clustering']['max_people_per_lookup']=1
    limited=calculate(replace(config,data=data),'face',selected,'scope')
    assert limited['truncated'] and limited['compared']==1
    assert limited['rows'][0]['owner']  # Current owner receives the limited budget first.
