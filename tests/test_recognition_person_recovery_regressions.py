import json
import pytest
from tests.test_recognition_body_collection import settings,Persons
from tests.helpers.recognition import Backend,MediaClient
from stash_ai.recognition.database import FaceDatabase
from stash_ai.recognition.person_collection import PersonCollector
from stash_ai.recognition.media import SpriteInput
from stash_ai.recognition.scanning import scan_scene


def test_person_failure_closes_poisoned_model_and_next_scene_gets_fresh_owner(tmp_path):
    c=settings(tmp_path);client=MediaClient();models=[]
    class Model(Persons):
        def close(self):self.closed=True
    def factory(*args):
        model=Model(1);model.closed=False;model.fail=not models;models.append(model);return model
    collector=PersonCollector(c,{},factory)
    try:
        with FaceDatabase(c.database) as state,SpriteInput(client,client.scene,c) as source:
            with pytest.raises(RuntimeError,match='body inference failed'):
                scan_scene(state,'scope',client.scene,c,'ns',source,Backend,on_cue=collector.collect_frame)
            assert collector.model is None and collector.pipeline is None and models[0].closed
            scan_scene(state,'scope',client.scene,c,'ns',source,Backend,on_cue=collector.collect_frame)
            assert len(models)==2 and models[1].embedded==3
    finally:collector.close()


@pytest.mark.parametrize('initial_budget',[None,1])
def test_retention_change_backfills_crops_without_inference(tmp_path,initial_budget):
    c=settings(tmp_path);c.data['sapiensid']['storage']['retain_person_crops']=initial_budget is not None
    normal_budget=c.data['storage']['max_crop_bytes']
    client=MediaClient();model=Persons(1);collector=PersonCollector(c,{},lambda *a:model)
    try:
        with FaceDatabase(c.database) as state,SpriteInput(client,client.scene,c) as source:
            scan,_=scan_scene(state,'scope',client.scene,c,'ns',source,Backend)
            if initial_budget is not None:c.data['storage']['max_crop_bytes']=initial_budget
            collector.enrich(state.connection,scan,source)
            assert state.connection.execute('SELECT COUNT(*) FROM body_observations WHERE crop IS NOT NULL').fetchone()[0]==0
            before=[tuple(r) for r in state.connection.execute('SELECT id,embedding FROM body_observations ORDER BY id')]
            calls=model.calls,model.embedded
            collector.close()
            collector.factory=lambda *a:pytest.fail('Crop backfill must not load models')
            c.data['sapiensid']['storage']['retain_person_crops']=True
            c.data['storage']['max_crop_bytes']=normal_budget
            collector.enrich(state.connection,scan,source)
            for table in ('body_observations','person_evidence'):
                assert state.connection.execute('SELECT COUNT(*) FROM '+table+' WHERE crop IS NOT NULL').fetchone()[0]==3
            assert before==[tuple(r) for r in state.connection.execute('SELECT id,embedding FROM body_observations ORDER BY id')]
            assert calls==(model.calls,model.embedded)
            collector.enrich(state.connection,scan,source)
    finally:collector.close()


@pytest.mark.parametrize('people,body_rejected,unavailable',[(0,0,3),(1,3,3)])
def test_run_reports_body_rejections_and_unavailable_person_evidence(tmp_path,monkeypatch,people,body_rejected,unavailable):
    from dataclasses import replace
    from stash_ai.recognition.config import plain
    from tests.test_recognition_run import setup,run
    c,client,services,_=setup(tmp_path,monkeypatch)
    c=replace(c,data=plain(c.data));c.data['sapiensid']['enabled']=True
    c.data['sapiensid']['collection'].update(enabled=True,min_person_pixels=200,min_sharpness=0,max_overlap=1)
    monkeypatch.setattr('stash_ai.recognition.person_collection.PersonCollector',
        lambda config,paths:PersonCollector(config,paths,lambda *a:Persons(people)))
    assert run(c,services,'1')==0
    with FaceDatabase(c.database) as state:
        summary=json.loads(state.connection.execute('SELECT summary FROM runs ORDER BY started DESC LIMIT 1').fetchone()[0])
        assert summary['counts']['body_observations_rejected']==body_rejected
        assert summary['counts']['person_evidence_unavailable']==unavailable
        assert sum(summary['rejections']['body'].values())==body_rejected
        assert sum(summary['rejections']['person_evidence_unavailable'].values())==unavailable
