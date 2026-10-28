from dataclasses import replace
import json
import sqlite3
import uuid
import numpy as np
import pytest

from tests.test_recognition_scanning import config
from tests.helpers.recognition import Backend,MediaClient
from stash_ai.recognition.config import plain,load
from stash_ai.recognition.database import FaceDatabase,SCHEMA_V8,APPLICATION_ID,SCHEMA_VERSION
from stash_ai.recognition.media import SpriteInput
from stash_ai.recognition.scanning import scan_scene
from stash_ai.recognition.person_collection import PersonCollector
from stash_ai.recognition.body_observations import contract,coverage
from stash_ai.recognition.sapiensid_backend import associate,SapiensIDFaces


def settings(tmp_path):
    c=config(tmp_path);d=plain(c.data)
    d['sapiensid']['enabled']=True
    d['sapiensid']['collection'].update(enabled=True,min_sharpness=0,max_overlap=1)
    d['sapiensid']['execution']['batch_size']=2
    return replace(c,data=d)


class Persons:
    def __init__(self,count=2):self.calls=0;self.embedded=0;self.count=count;self.fail=False;self.inputs=[]
    def _owner(self):pass
    person_geometry=SapiensIDFaces.person_geometry
    def detect(self,image,faces=None):
        self.calls+=1
        self.frame_people=[]
        for i in range(self.count):
            points=[[-1.,-1.]]*17
            if i==0:points[:3]=[[80.,70.],[60.,60.],[100.,60.]]
            points[5:13]=[[40.,90.],[120.,90.],[30.,110.],[130.,110.],[20.,130.],[140.,130.],[60.,145.],[100.,145.]]
            self.frame_people.append(dict(bounds=[0,0,160,160],keypoints=points,score=.99))
        self.face_matches=associate(faces,self.frame_people)
        return faces
    def embed_inputs(self,prepared):
        if self.fail:raise RuntimeError('body inference failed')
        self.embedded+=len(prepared);self.inputs.extend(prepared)
        return [np.ones(4096) for _ in prepared]
    def close(self):pass


def test_shared_pass_one_pose_per_frame_one_feature_per_person(tmp_path):
    pytest.importorskip('torch');pytest.importorskip('torchvision')
    c=settings(tmp_path);client=MediaClient();model=Persons()
    collector=PersonCollector(c,{},lambda *args:model)
    with FaceDatabase(c.database) as state,SpriteInput(client,client.scene,c) as source:
        scan,_=scan_scene(state,'scope',client.scene,c,'ns',source,Backend,on_cue=collector.collect_frame)
        collector.enrich(state.connection,scan,source)
        assert (model.calls,model.embedded)==(3,6)
        assert coverage(state.connection,c,[scan])['ready']==6
        assert state.connection.execute('SELECT COUNT(*) FROM person_evidence').fetchone()[0]==3
        for b in state.connection.execute('SELECT b.embedding,p.embedding FROM body_observations b JOIN person_evidence p ON p.observation=b.face'):
            assert b[0]==b[1]
        collector.enrich(state.connection,scan,source)
        assert (model.calls,model.embedded)==(3,6)
        assert any((points[0,17:]==-1).all() for _,points,_ in model.inputs)
    collector.close()


@pytest.mark.parametrize('people',[0,1])
def test_faceless_frames_and_empty_cues_are_reusable(tmp_path,people):
    pytest.importorskip('torch');pytest.importorskip('torchvision')
    c=settings(tmp_path);client=MediaClient();model=Persons(people)
    collector=PersonCollector(c,{},lambda *args:model)
    backend=Backend();backend.detect=lambda image:[]
    with FaceDatabase(c.database) as state,SpriteInput(client,client.scene,c) as source:
        scan,_=scan_scene(state,'scope',client.scene,c,'ns',source,lambda:backend)
        collector.enrich(state.connection,scan,source)
        assert (model.calls,model.embedded)==(3,3*people)
        collector.enrich(state.connection,scan,source)
        assert model.calls==3
        assert coverage(state.connection,c,[scan])['complete_cues']==3
        assert state.connection.execute('SELECT COUNT(*) FROM observations').fetchone()[0]==0
    collector.close()


def test_failure_does_not_checkpoint_partial_body_or_face_cue(tmp_path):
    pytest.importorskip('torch');pytest.importorskip('torchvision')
    c=settings(tmp_path);client=MediaClient();model=Persons();model.fail=True
    collector=PersonCollector(c,{},lambda *args:model)
    with FaceDatabase(c.database) as state,SpriteInput(client,client.scene,c) as source:
        with pytest.raises(RuntimeError,match='body inference failed'):
            scan_scene(state,'scope',client.scene,c,'ns',source,Backend,on_cue=collector.collect_frame)
        for table in ('observations','body_observations','body_cues','person_evidence'):
            assert state.connection.execute('SELECT COUNT(*) FROM '+table).fetchone()[0]==0
        assert state.connection.execute('SELECT cursor FROM scans').fetchone()[0]==0
        model.fail=False
        scan_scene(state,'scope',client.scene,c,'ns',source,Backend,on_cue=collector.collect_frame)
        assert state.connection.execute('SELECT COUNT(*) FROM body_cues').fetchone()[0]==3
    collector.close()


def test_v8_migration_preserves_installation(tmp_path):
    path=tmp_path/'old.sqlite';installation=str(uuid.uuid4())
    with sqlite3.connect(path) as db:
        db.executescript(SCHEMA_V8)
        db.execute('INSERT INTO metadata VALUES (?,?)',('installation',installation))
        db.execute(f'PRAGMA application_id={APPLICATION_ID}');db.execute('PRAGMA user_version=8')
    with FaceDatabase(path) as state:
        assert state.installation==installation
        assert state.connection.execute('PRAGMA user_version').fetchone()[0]==SCHEMA_VERSION
        assert state.connection.execute('SELECT COUNT(*) FROM body_observations').fetchone()[0]==0


def test_collection_validation_and_contract(tmp_path):
    c=settings(tmp_path);before=contract(c)
    c.data['sapiensid']['matching']['person_threshold']=.5
    assert contract(c)==before
    c.data['sapiensid']['collection']['min_keypoints']=5
    assert contract(c)!=before
    path=tmp_path/'bad.yml';path.write_text('version: 1\nsapiensid: {collection: {enabled: true}}')
    with pytest.raises(ValueError,match='requires sapiensid.enabled'):load(path)
