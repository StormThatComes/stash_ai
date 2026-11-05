from tests.test_recognition_rebuild import changed, library, partition, two_people, run
from stash_ai.recognition import rebuild


def test_append_reuses_independent_strict_scene_and_matches_cold(tmp_path,monkeypatch):
    cfg,_,services,_=library(tmp_path/'warm',monkeypatch,two_people())
    cfg=changed(cfg,clustering={'anchor_view_matching':True})
    assert run(cfg,services,'1')==0
    called=[]
    original=rebuild.cluster_scene
    def record(database,scan,*args,**kwargs):
        called.append(database.connection.execute('SELECT scene FROM scans WHERE id=?',(scan,)).fetchone()[0])
        return original(database,scan,*args,**kwargs)
    monkeypatch.setattr(rebuild,'cluster_scene',record)
    assert run(cfg,services,'1','2')==0
    assert called==['2']
    cold,_,services,_=library(tmp_path/'cold',monkeypatch,two_people())
    cold=changed(cold,clustering={'anchor_view_matching':True},execution={'rebuild_cache_bytes':0})
    assert run(cold,services,'1','2')==0
    assert partition(cfg)==partition(cold)


def test_parallel_body_plans_reuse_independent_scenes(tmp_path):
    from stash_ai.recognition.database import FaceDatabase
    from stash_ai.recognition.compute import compute_scope
    from stash_ai.recognition.derived_cache import scope
    from stash_ai.recognition.body_scene_jobs import planned
    from stash_ai.recognition.body_identity import scene_plan
    from tests.test_recognition_body_global import configured,scene
    config=configured(tmp_path)
    config.data['execution']['rebuild_workers']=2
    with FaceDatabase(config.database) as state:
        selected=[scene(state.connection,config,str(i))[0] for i in range(1,4)]
        expected=[scene_plan(state.connection,i['scan'],config)[2] for i in selected]
        with scope(state,config),compute_scope(config):
            assert [p[2] for p in planned(state.connection,selected[:2],config)]==expected[:2]
            assert [p[2] for p in planned(state.connection,selected,config)]==expected
