import uuid
from pathlib import Path
import pytest
from stash_ai.recognition.database import FaceDatabase
from tests.test_recognition_body_identity import configuration, scene
from stash_ai.recognition.body_identity import rebuild


def test_stable_body_origin(tmp_path):
    t=tmp_path

    c=configuration(Path(t))
    from stash_ai.recognition.body_observations import contract, identifier
    with FaceDatabase(c.database) as st:
        db=st.connection
        item,ids=scene(db,c);db.commit()
        a=rebuild(st,[item],c,{item['scan']:[]})[item['scan']][0]['person']
        old=list(db.execute('SELECT * FROM body_observations'))
        c.data['sapiensid']['collection']['min_sharpness']=1
        for r in old:
            values=list(r);values[0]=str(uuid.uuid5(uuid.NAMESPACE_URL,r['id']+contract(c)));values[4]=contract(c)
            db.execute('INSERT INTO body_observations VALUES (?,?,?,?,?,?,?,?,?,?,?)',values)
            db.execute('INSERT OR IGNORE INTO body_cues VALUES (?,?,?)',(item['scan'],contract(c),r['cue']))
        db.commit();b=rebuild(st,[item],c,{item['scan']:[]})[item['scan']][0]['person']
        assert a==b


def test_failed_hybrid_does_not_publish(tmp_path):
    t=tmp_path

    from tests.test_recognition_clustering import setup,scan
    from stash_ai.recognition.config import plain
    from dataclasses import replace
    from stash_ai.recognition.rebuild import rebuild_selected
    from stash_ai.recognition.publication_queue import plan
    c=setup(Path(t));c=replace(c,data=plain(c.data));c.data['sapiensid']['enabled']=True
    c.data['sapiensid']['execution']['reference_bytes']=1
    with FaceDatabase(c.database) as st:
        sid,_=scan(st,c,'1')
        try:rebuild_selected(st,[dict(scan=sid,scene='1')],c)
        except ValueError as err:
            assert st.connection.execute('SELECT COUNT(*) FROM members').fetchone()[0]==0
            assert not plan(st.connection,'scope','ns',c)


def test_retuning_keeps_historical_rivals(tmp_path):
    t=tmp_path

    from stash_ai.recognition.body_global_banks import banks
    c=configuration(Path(t));c.data['sapiensid']['global_matching']['enabled']=True
    c.data['sapiensid']['identities']['min_support']=2
    with FaceDatabase(c.database) as st:
        db=st.connection;a,_=scene(db,c,'1');b,_=scene(db,c,'2');r,_=scene(db,c,'3',vectors=[[.99,.1]]*2);db.commit()
        rebuild(st,[a,b,r],c,{x['scan']:[] for x in [a,b,r]})
        before=db.execute('SELECT COUNT(*) FROM identity_merges').fetchone()[0]
        c.data['sapiensid']['global_matching']['match_threshold']=.91
        rebuild(st,[a,b],c,{x['scan']:[] for x in [a,b]})
        assert before==db.execute('SELECT COUNT(*) FROM identity_merges').fetchone()[0]==0
        assert len(banks(db,c,'scope','ns'))==3


def test_late_body_failure_restores_prior_assignments_and_queue(tmp_path,monkeypatch):
    from tests.test_recognition_clustering import setup,scan
    from stash_ai.recognition.rebuild import rebuild_selected
    c=setup(tmp_path)
    with FaceDatabase(c.database) as state:
        sid,_=scan(state,c,'1')
        selected=[dict(scan=sid,scene='1')]
        rebuild_selected(state,selected,c)
        tables=('members','candidates','people','identity_origins','metadata','identity_merges')
        before={t:list(state.connection.execute('SELECT * FROM '+t)) for t in tables}
        def fail(*args):
            state.connection.execute('DELETE FROM members')
            raise RuntimeError('late body failure')
        monkeypatch.setattr('stash_ai.recognition.body_identity.rebuild',fail)
        with pytest.raises(RuntimeError,match='late body failure'):
            rebuild_selected(state,selected,c)
        assert before=={t:list(state.connection.execute('SELECT * FROM '+t)) for t in tables}
        assert not state.connection.in_transaction
