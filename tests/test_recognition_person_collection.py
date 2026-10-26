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


def test_variant_change_collects_separate_evidence_and_switch_back_reuses_cache(tmp_path):
    from stash_ai.recognition.person_signal import contract, coverage
    settings=config(tmp_path);data=plain(settings.data);data['sapiensid']['enabled']=True
    settings=replace(settings,data=data)
    client=MediaClient();models=[];contracts={}
    with FaceDatabase(settings.database) as state,SpriteInput(client,client.scene,settings) as source:
        db=state.connection
        scan,_=scan_scene(state,'scope',client.scene,settings,'ns',source,Backend)
        faces=list(db.execute('SELECT id,embedding FROM observations'))
        for variant in ('wb4m','wb12m','wb4m'):
            settings.data['sapiensid']['model']['variant']=variant
            if variant not in contracts:
                assert coverage(db,settings,[scan])['ready']==0
            model=Person();models.append(model)
            collector=PersonCollector(settings,{},factory=lambda *args:model)
            try:collector.enrich(db,scan,source)
            finally:collector.close()
            contracts[variant]=contract(settings)
            assert db.execute('SELECT COUNT(*) FROM person_evidence WHERE contract=?',
                              (contract(settings),)).fetchone()[0]==3
        assert contracts['wb4m']!=contracts['wb12m']
        assert [m.calls for m in models]==[3,3,0]
        assert db.execute('SELECT COUNT(*) FROM person_evidence').fetchone()[0]==6
        assert faces==list(db.execute('SELECT id,embedding FROM observations'))


def test_multi_face_batches_are_bounded_and_failed_cue_is_atomic(tmp_path):
    import uuid
    settings=config(tmp_path);data=plain(settings.data)
    data['sapiensid']['enabled']=True;data['sapiensid']['execution']['batch_size']=2
    settings=replace(settings,data=data)
    class Batched(Person):
        def __init__(self):super().__init__();self.sizes=[];self.fail=True
        def embed_faces(self,image,faces):
            self.sizes.append(len(faces))
            if self.fail and len(self.sizes)==2:return []
            return [np.ones(4096) for face in faces]
        def embed(self,*a):pytest.fail('batch adapter fell back to scalar')
    client=MediaClient();model=Batched();collector=PersonCollector(settings,{},factory=lambda *a:model)
    with FaceDatabase(settings.database) as state,SpriteInput(client,client.scene,settings) as source:
        scan,_=scan_scene(state,'scope',client.scene,settings,'ns',source,Backend)
        db=state.connection
        first=list(db.execute('SELECT * FROM observations WHERE cue=0').fetchone())
        for _ in range(2):
            first[0]=str(uuid.uuid4())
            db.execute('INSERT INTO observations VALUES (?,?,?,?,?,?,?,?,?,?,?,?)',first)
        db.commit()
        with pytest.raises(ValueError,match='feature count'):collector.enrich(db,scan,source)
        assert model.sizes==[2,1]
        assert db.execute('SELECT COUNT(*) FROM person_evidence').fetchone()[0]==0
        model.fail=False
        collector.enrich(db,scan,source)
        assert max(model.sizes)==2
        assert db.execute('SELECT COUNT(*) FROM person_evidence').fetchone()[0]==5
    collector.close()


def test_prepared_pipeline_batches_across_cues_on_one_inference_owner(tmp_path,monkeypatch):
    import threading
    owner=threading.get_ident();workers=[]
    def prepare(image,points):
        workers.append(threading.get_ident());return (np.zeros(1),np.zeros(1),np.ones(1))
    monkeypatch.setattr('stash_ai.recognition.sapiensid_v1.prepare_person_input',prepare)
    settings=config(tmp_path);data=plain(settings.data)
    data['sapiensid']['enabled']=True;data['sapiensid']['execution']['batch_size']=2
    settings=replace(settings,data=data)
    class Prepared(Person):
        def __init__(self):super().__init__();self.batches=[]
        def person_input(self,image,face):
            assert threading.get_ident()==owner
            return [0,0,image.width,image.height],[]
        def embed_inputs(self,inputs):
            assert threading.get_ident()==owner
            self.batches.append(len(inputs));return [np.ones(4096) for _ in inputs]
    settings.data['execution']['checkpoint_cues']=3
    model=Prepared();collector=PersonCollector(settings,{},factory=lambda *a:model)
    client=MediaClient()
    try:
        with FaceDatabase(settings.database) as state,SpriteInput(client,client.scene,settings) as source:
            scan,_=scan_scene(state,'scope',client.scene,settings,'ns',source,Backend,on_cue=collector.collect_frame)
            collector.flush()
            source.frames_indices=lambda *a:pytest.fail('completed inline evidence reopened VTT')
            collector.enrich(state.connection,scan,source)
            assert model.batches==[2,1]
            assert state.connection.execute('SELECT COUNT(*) FROM person_evidence').fetchone()[0]==3
            assert workers and all(thread!=owner for thread in workers)
    finally:collector.close()


def test_resume_reads_only_missing_cue_indices(tmp_path):
    settings=config(tmp_path);data=plain(settings.data);data['sapiensid']['enabled']=True
    settings=replace(settings,data=data);client=MediaClient();collector=PersonCollector(settings,{},factory=Person)
    try:
        with FaceDatabase(settings.database) as state,SpriteInput(client,client.scene,settings) as source:
            scan,_=scan_scene(state,'scope',client.scene,settings,'ns',source,Backend)
            collector.enrich(state.connection,scan,source)
            state.connection.execute('DELETE FROM person_evidence WHERE observation IN (SELECT id FROM observations WHERE cue=1)');state.connection.commit()
            original=source.frames_indices;seen=[]
            def selected(indices):
                seen.extend(indices);yield from original(indices)
            source.frames_indices=selected
            collector.enrich(state.connection,scan,source)
            assert seen==[1]
    finally:collector.close()


def test_prepared_failure_does_not_publish_partial_cue_and_can_resume(tmp_path,monkeypatch):
    import uuid
    monkeypatch.setattr('stash_ai.recognition.sapiensid_v1.prepare_person_input',lambda *a:(np.zeros(1),)*3)
    settings=config(tmp_path);data=plain(settings.data);data['sapiensid']['enabled']=True
    data['sapiensid']['execution']['batch_size']=2;settings=replace(settings,data=data)
    class Prepared(Person):
        def person_input(self,image,face):return [0,0,image.width,image.height],[]
        def embed_inputs(self,inputs):
            self.calls+=1
            if self.fail and self.calls==2:raise RuntimeError('prepared inference failed')
            return [np.ones(4096) for _ in inputs]
    model=Prepared();model.fail=True
    collector=PersonCollector(settings,{},factory=lambda *a:model);client=MediaClient()
    try:
        with FaceDatabase(settings.database) as state,SpriteInput(client,client.scene,settings) as source:
            scan,_=scan_scene(state,'scope',client.scene,settings,'ns',source,Backend)
            db=state.connection;row=list(db.execute('SELECT * FROM observations WHERE cue=0').fetchone())
            for _ in range(2):
                row[0]=str(uuid.uuid4());db.execute('INSERT INTO observations VALUES (?,?,?,?,?,?,?,?,?,?,?,?)',row)
            db.commit()
            with pytest.raises(RuntimeError,match='prepared inference'):collector.enrich(db,scan,source)
            assert collector.pipeline is None
            assert db.execute('SELECT COUNT(*) FROM person_evidence').fetchone()[0]==0
            model.fail=False;collector.enrich(db,scan,source)
            assert db.execute('SELECT COUNT(*) FROM person_evidence').fetchone()[0]==5
    finally:collector.close()


def test_person_cleanup_releases_model_when_cpu_shutdown_fails(tmp_path):
    from types import SimpleNamespace
    collector = PersonCollector(config(tmp_path), {})
    def broken(): raise RuntimeError('shutdown failed')
    collector.pipeline = SimpleNamespace(close=broken)
    model = Person();collector.model = model
    with pytest.raises(RuntimeError, match='shutdown failed'): collector.close()
    assert model.closed and collector.pipeline is None and collector.model is None
