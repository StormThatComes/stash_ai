from dataclasses import replace
import numpy as np
import pytest

from tests.test_recognition_scanning import config
from tests.helpers.recognition import Backend, MediaClient
from stash_ai.recognition.config import plain
from stash_ai.recognition.database import FaceDatabase
from stash_ai.recognition.media import SpriteInput
from stash_ai.recognition.scanning import scan_scene
from stash_ai.recognition.person_collection import PersonCollector


class Person:
    def __init__(self,*args): self.calls=0;self.closed=False;self.fail=False
    def detect(self,image,faces=None): return faces
    def observation_quality(self,image,face):
        return dict(identity_rejection=None,person=dict(bounds=[0,0,image.width,image.height]))
    def embed(self,image,face):
        self.calls+=1
        if self.fail and self.calls==2:raise RuntimeError('broken inference')
        return np.ones(4096)
    def close(self):self.closed=True


def test_collection_resumes_without_reembedding_faces(tmp_path):
    settings=config(tmp_path);data=plain(settings.data);data['sapiensid']['enabled']=True
    settings=replace(settings,data=data)
    client=MediaClient();model=Person();collector=PersonCollector(settings,{},factory=lambda *args:model)
    with FaceDatabase(settings.database) as state,SpriteInput(client,client.scene,settings) as source:
        scan,_=scan_scene(state,'scope',client.scene,settings,'ns',source,Backend)
        before=list(state.connection.execute('SELECT id,embedding FROM observations'))
        model.fail=True
        with pytest.raises(RuntimeError,match='broken'):collector.enrich(state.connection,scan,source)
        assert state.connection.execute('SELECT COUNT(*) FROM person_evidence').fetchone()[0]==1
        model.fail=False
        collector.enrich(state.connection,scan,source)
        assert state.connection.execute('SELECT COUNT(*) FROM person_evidence').fetchone()[0]==3
        calls=model.calls
        collector.enrich(state.connection,scan,source)
        assert model.calls==calls
        assert before==list(state.connection.execute('SELECT id,embedding FROM observations'))
        assert all(r[0] for r in state.connection.execute('SELECT crop FROM person_evidence'))
    collector.close();assert model.closed
