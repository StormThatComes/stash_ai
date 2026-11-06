import numpy as np


def test_body_global_append_reuses_means_and_old_negative_nominations(tmp_path):
    from tests.test_recognition_body_scaling import fixture
    from tests.test_recognition_body_identity import scene
    from stash_ai.recognition.body_identity import rebuild
    from stash_ai.recognition.body_global import _report
    from stash_ai.recognition.derived_cache import scope
    config,state=fixture(tmp_path,np.eye(32,33).tolist())
    try:
        db=state.connection
        with scope(state,config):first=_report(db,config,'scope','ns',compact=True)
        assert first['search']['embedding_reads']==96
        item,ids=scene(db,config,'33',vectors=[np.eye(33)[32].tolist()]*3);db.commit()
        rebuild(state,[item],config,{item['scan']:[]})
        with scope(state,config):delta=_report(db,config,'scope','ns',compact=True)
        cold=_report(db,config,'scope','ns',compact=True)
        assert delta['pairs']==cold['pairs']==[]
        assert delta['decisions']==cold['decisions']==[]
        assert delta['search']['embedding_reads']==3
        assert delta['search']['centroid_pairs']==32
        # Changed source bytes invalidate the old negative even with stable IDs.
        source=db.execute('SELECT embedding FROM body_observations WHERE id<>? LIMIT 1',(ids[0],)).fetchone()[0]
        db.executemany('UPDATE body_observations SET embedding=? WHERE id=?',[(source,i) for i in ids]);db.commit()
        with scope(state,config):modified=_report(db,config,'scope','ns',compact=True)
        fresh=_report(db,config,'scope','ns',compact=True)
        assert modified['pairs']==fresh['pairs'] and modified['decisions']==fresh['decisions']
    finally:state.__exit__(None,None,None)


def test_completed_hybrid_modes_reused_without_recomparing_witnesses(tmp_path,monkeypatch):
    from tests.test_recognition_hybrid_merge import prepared
    from stash_ai.recognition.database import FaceDatabase
    from stash_ai.recognition.derived_cache import scope
    from stash_ai.recognition.hybrid_evidence import catalogue
    from stash_ai.recognition.disk_body_modes import Records
    config,_=prepared(tmp_path)
    with FaceDatabase(config.database) as state:
        ns=state.connection.execute('SELECT namespace FROM people LIMIT 1').fetchone()[0]
        with scope(state,config):first=catalogue(state.connection,config,'scope',ns)
        assert any(i.body_modes for i in first)
        def unexpected(*a):raise AssertionError('unchanged appearance bank rebuilt')
        monkeypatch.setattr(Records,'_build_modes',unexpected)
        with scope(state,config):second=catalogue(state.connection,config,'scope',ns)
        assert [i.person for i in first]==[i.person for i in second]
        for a,b in zip(first,second):
            assert len(a.body_modes)==len(b.body_modes)
            for x,y in zip(a.body_modes,b.body_modes):
                assert np.array_equal(x[:],y[:])
                assert np.array_equal(x.weights(slice(None)),y.weights(slice(None)))


def test_historical_scan_digests_survive_new_selection(tmp_path,monkeypatch):
    from tests.test_recognition_rebuild import library
    from tests.test_recognition_run import run
    from stash_ai.recognition.database import FaceDatabase
    from stash_ai.recognition.historical_references import context
    config,_,services,_=library(tmp_path,monkeypatch,{'1':[[1.,0,0,0]]*3,'2':[[0,1.,0,0]]*3})
    assert run(config,services,'1','2')==0
    with FaceDatabase(config.database) as state:
        db=state.connection
        scope,ns=db.execute('SELECT scope,namespace FROM scans LIMIT 1').fetchone()
        before=context(db,[],config,scope,ns,cache=True);db.commit()
        sql=[];db.set_trace_callback(sql.append)
        selected=[dict(scan='not-in-db',scene='999')]
        assert context(db,selected,config,scope,ns,cache=True)==before
        db.set_trace_callback(None)
        assert not any('dimensions,embedding FROM observations' in q for q in sql)
        assert context(db,selected,config,scope,ns,cache=False)==before
        db.execute('UPDATE observations SET embedding=? WHERE rowid=(SELECT min(rowid) FROM observations)',(np.array([0,0,1,0],'<f4').tobytes(),))
        assert context(db,selected,config,scope,ns,cache=True)!=before


def test_new_weak_body_rival_can_veto_a_cached_positive_pair(tmp_path):
    import math
    from tests.test_recognition_body_scaling import fixture
    from stash_ai.recognition.body_global import banks,plan
    from stash_ai.recognition.derived_cache import scope
    config,state=fixture(tmp_path,[[math.cos(a),math.sin(a)] for a in (0.,.02,.01,1.5,3.,4.5)])
    try:
        db=state.connection;bank=banks(db,config,'scope','ns')
        rival=next(p for p,b in bank.items() if db.execute('SELECT scene FROM scans WHERE id=?',(b['refs'][0]['scan'],)).fetchone()[0]=='3')
        original={p:b for p,b in bank.items() if p!=rival}
        with scope(state,config):before=plan(db,config,original,compact=True)
        assert any(d['outcome']=='merge' for d in before['decisions'])
        bank[rival]['support']=1
        with scope(state,config):after=plan(db,config,bank,compact=True)
        cold=plan(db,config,bank,compact=True)
        assert after['search']['centroid_catalogue_hits']==5
        assert after['decisions']==cold['decisions']
        assert all(d['outcome']=='blocked' for d in after['decisions'])
        assert any('body_ambiguity' in d['blockers'] for d in after['decisions'])
    finally:state.__exit__(None,None,None)


def test_incomplete_cached_mode_is_removed_before_rebuild(tmp_path,monkeypatch):
    import base64,zlib
    from types import SimpleNamespace
    from stash_ai.recognition.config import defaults
    from stash_ai.recognition.disk_body_modes import Records
    from stash_ai.recognition import body_mode_cache
    payload=base64.b64encode(zlib.compress(bytes(4096*4))).decode()
    record=dict(modes=[[1,payload,'invalid']],support=1,selected=['one'])
    store=SimpleNamespace(budget=2**26,get=lambda *a:record)
    monkeypatch.setattr(body_mode_cache,'current',lambda:store)
    spool=Records()
    try:
        assert body_mode_cache.load(spool,'key',SimpleNamespace(data=defaults())) is None
        assert not (spool.path/'0.bin').exists()
    finally:spool.close()
