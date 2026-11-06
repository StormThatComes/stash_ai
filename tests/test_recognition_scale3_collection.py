from dataclasses import replace
from io import BytesIO
from types import SimpleNamespace

from PIL import Image
import pytest

from stash_ai.recognition.config import plain
from stash_ai.recognition.database import FaceDatabase
from stash_ai.recognition.demand_crops import Crops
from stash_ai.recognition.media import SpriteInput
from stash_ai.recognition.person_collection import PersonCollector
from stash_ai.recognition.run import instance_scope
from stash_ai.recognition.scanning import scan_scene
from tests.test_recognition_scanning import config
from tests.helpers.recognition import Backend,MediaClient
from tests.test_recognition_person_collection import Person


def test_resume_reads_each_cue_once_without_feature_blobs_and_commits_a_batch(tmp_path):
    c=config(tmp_path);data=plain(c.data);data['sapiensid']['enabled']=True;data['execution']['checkpoint_cues']=32
    c=replace(c,data=data);client=MediaClient();collector=PersonCollector(c,{},factory=lambda *_:Person())
    try:
        with FaceDatabase(c.database) as state,SpriteInput(client,client.scene,c) as source:
            scan,_=scan_scene(state,'scope',client.scene,c,'ns',source,Backend)
            sql=[];state.connection.set_trace_callback(sql.append)
            collector.enrich(state.connection,scan,source)
            reads=[q for q in sql if q.startswith('SELECT id,geometry,rejected FROM observations')]
            assert len(reads)==3
            assert not any('SELECT * FROM observations' in q for q in sql)
            assert sum(q=='COMMIT' for q in sql)==1
            assert state.connection.execute('SELECT COUNT(*) FROM person_evidence').fetchone()[0]==3
    finally:collector.close()


def test_selected_crops_preserve_pixel_evidence_and_reconstruct_only_requested_face(tmp_path):
    c=config(tmp_path);data=plain(c.data);data['storage']['crop_policy']='selected'
    c=replace(c,data=data);client=MediaClient();backend=Backend()
    with FaceDatabase(c.database) as state,SpriteInput(client,client.scene,c) as source:
        scan,_=scan_scene(state,instance_scope(client.url),client.scene,c,'ns',source,lambda:backend)
        db=state.connection
        assert not db.execute('SELECT 1 FROM observations WHERE crop IS NOT NULL').fetchone()
        assert db.execute("SELECT COUNT(*) FROM observations WHERE json_extract(quality,'$.evidence_pixel_sha256') IS NOT NULL").fetchone()[0]==3
        observation=db.execute('SELECT id FROM observations LIMIT 1').fetchone()[0]
        reader=SimpleNamespace(url=client.url,scene=lambda _:client.scene,media=client.media)
        crops=Crops(db,c,reader)
        try:
            with Image.open(BytesIO(crops.face(observation))) as image:assert image.width>0
            assert backend.calls==3
            assert not db.execute('SELECT 1 FROM observations WHERE crop IS NOT NULL').fetchone()
            db.execute("UPDATE observations SET quality=json_set(quality,'$.evidence_pixel_sha256','changed') WHERE id=?",(observation,))
            with pytest.raises(ValueError,match='pixels differ'):crops.face(observation)
        finally:crops.close()


def test_selected_body_crop_is_recovered_without_another_pose_or_embedding(tmp_path):
    pytest.importorskip('torch')
    from tests.test_recognition_body_collection import settings,Persons
    c=settings(tmp_path);c.data['storage']['crop_policy']='selected'
    client=MediaClient(count=1);model=Persons(1);collector=PersonCollector(c,{},lambda *_:model)
    try:
        with FaceDatabase(c.database) as state,SpriteInput(client,client.scene,c) as source:
            scan,_=scan_scene(state,instance_scope(client.url),client.scene,c,'ns',source,Backend,on_cue=collector.collect_frame)
            collector.enrich(state.connection,scan,source)
            row=state.connection.execute('SELECT id,crop FROM body_observations').fetchone()
            assert row['crop'] is None
            reader=SimpleNamespace(url=client.url,scene=lambda _:client.scene,media=client.media)
            crops=Crops(state.connection,c,reader)
            try:
                with Image.open(BytesIO(crops.body(row['id']))) as image:assert image.size==(160,160)
                assert (model.calls,model.embedded)==(1,1)
            finally:crops.close()
    finally:collector.close()
