import gc
import pickle
import numpy as np
import pytest
from stash_ai.recognition.database import FaceDatabase
from stash_ai.recognition.body_identity import scene_plan,rebuild
from tests.test_recognition_body_identity import configuration,scene


def test_1415_compatible_body_observations_fit_default_comparison_budget(tmp_path):
    c=configuration(tmp_path)
    with FaceDatabase(c.database) as state:
        item,ids=scene(state.connection,c,vectors=[[1,0]]*1415)
        _,_,decisions=scene_plan(state.connection,item['scan'],c)
        assert {d['status'] for d in decisions.values()}=={'accepted'}
        assert len({d['person'] for d in decisions.values()})==1


def test_local_score_weights_match_distinct_pixel_reference(tmp_path):
    from stash_ai.recognition.body_identity import Scores,distinct_pixels
    rng=np.random.default_rng(12);rows=[]
    for i in range(30):
        vector=rng.normal(size=4096).astype('float32');vector/=np.linalg.norm(vector)
        rows.append(dict(id=str(i),quality='{}',_vector=vector))
    # Equal embeddings with distinct physical pixels still carry their weight.
    rows.extend(dict(row,id='copy-'+row['id']) for row in rows[:5])
    scores=Scores(configuration(tmp_path))
    for size in range(1,len(rows)+1):
        bank=rows[:size];first=rows[-2:]
        values=[float(np.dot(a['_vector'],b['_vector'])) for a in first for b in bank]
        assert scores.compare(first,bank)==pytest.approx((sum(values)/len(values),min(values)),abs=1e-12)


def test_warm_body_cache_and_unselected_rival_invalidation(tmp_path,monkeypatch):
    from stash_ai.recognition import body_identity
    c=configuration(tmp_path)
    with FaceDatabase(c.database) as state:
        item,_=scene(state.connection,c);rival,ids=scene(state.connection,c,'2');state.connection.commit()
        rebuild(state,[item],c,{item['scan']:[]})
        def forbidden(*a):raise AssertionError('replanned')
        monkeypatch.setattr(body_identity,'scene_plan',forbidden)
        rebuild(state,[item],c,{item['scan']:[]})
        state.connection.execute("UPDATE body_observations SET quality='{}' WHERE id=?",(ids[0],));state.connection.commit()
        with pytest.raises(AssertionError,match='replanned'):rebuild(state,[item],c,{item['scan']:[]})


def test_large_hybrid_owner_streams_every_witness_and_releases_spool(tmp_path,monkeypatch):
    from stash_ai.recognition.hybrid_references import catalogue_rows
    from stash_ai.recognition.hybrid_evidence import scores
    from stash_ai.recognition.disk_body_modes import Matrix
    import tempfile
    monkeypatch.setattr(tempfile,'tempdir',str(tmp_path))
    c=configuration(tmp_path);c.data['clustering']['max_samples_per_person']=2
    c.data['sapiensid']['execution']['reference_bytes']=16*1024**2
    vector=np.zeros(4096,dtype='<f4');vector[0]=1;blob=vector.tobytes()
    def rows(*args):
        for i in range(5000):
            yield dict(owner='p',id=str(i),scan='s',seconds=i,scene='1',geometry='[]',quality='{}',person_metadata='{}',embedding=blob)
    monkeypatch.setattr('stash_ai.recognition.hybrid_references._rows',rows)
    bank=catalogue_rows(None,c,'scope','ns',['p'])
    mode=bank['p'].modes[0]
    assert len(bank['p'])==2 and bank['p'].support==5000 and len(mode)==5000
    assert isinstance(mode,Matrix) and len(pickle.dumps(mode))<512
    assert scores(np.array([vector]),mode)==dict(mean=1.,minimum=1.)
    del mode,bank;gc.collect()
    assert not list(tmp_path.glob('stash-hybrid-witnesses-*'))


def test_independent_body_collection_batches_across_cues(tmp_path):
    from tests.test_recognition_body_collection import settings,Persons
    from tests.helpers.recognition import Backend,MediaClient
    from stash_ai.recognition.person_collection import PersonCollector
    from stash_ai.recognition.media import SpriteInput
    from stash_ai.recognition.scanning import scan_scene
    c=settings(tmp_path);c.data['execution']['checkpoint_cues']=10
    client=MediaClient();batches=[]
    class Model(Persons):
        def embed_inputs(self,values):
            batches.append(len(values));return super().embed_inputs(values)
    collector=PersonCollector(c,{},lambda *args:Model(1))
    try:
        with FaceDatabase(c.database) as state,SpriteInput(client,client.scene,c) as source:
            scan_scene(state,'scope',client.scene,c,'ns',source,Backend,on_cue=collector.collect_frame)
            assert batches==[2,1]
            assert state.connection.execute('SELECT COUNT(*) FROM body_cues').fetchone()[0]==3
    finally:collector.close()
