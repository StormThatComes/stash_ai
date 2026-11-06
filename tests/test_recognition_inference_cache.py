from contextlib import closing
from types import SimpleNamespace
import numpy as np
from PIL import Image

from stash_ai.recognition.inference_cache import Store,using,many,fingerprint
from stash_ai.recognition.config import defaults


def test_reopen_reuses_exact_pixels_but_preserves_independent_outputs(tmp_path):
    path=tmp_path/'cache.sqlite';calls=[]
    def compute(values):calls.extend(values);return [{'vector':np.arange(4,dtype=np.float32)} for _ in values]
    with closing(Store(path,1024**2)) as store,using(store):
        with Image.new('RGB',(8,8),'red') as a,Image.new('RGB',(8,8),'red') as b:
            outputs=many('test','v1',[a,b],compute)
            assert len(calls)==1 and len(outputs)==2
            outputs[0]['vector'][0]=8
            assert outputs[1]['vector'][0]==0
    with closing(Store(path,1024**2)) as store,using(store):
        with Image.new('RGB',(8,8),'red') as image:
            assert many('test','v1',[image],compute)[0]['vector'][0]==0
            assert len(calls)==1
            many('test','v2',[image],compute)
            image.putpixel((0,0),(1,2,3));many('test','v1',[image],compute)
            assert len(calls)==3


def test_contracts_invalidate_only_dependent_stages():
    from stash_ai.recognition.inference_contracts import detector,pose,body_encoder,face
    config=SimpleNamespace(data=defaults());config.data['sapiensid']['pose']['backend']='vitpose_plus_large'
    stages=lambda:[fingerprint(f(config)) for f in (face,detector,pose,body_encoder)]
    before=stages()
    config.data['sapiensid']['collection']['min_sharpness']+=1
    config.data['detection']['min_face_pixels']+=1
    config.data['recognition']['match_threshold']=.99
    assert stages()==before
    config.data['sapiensid']['pose']['score_threshold']+=.01
    after=stages();assert [a!=b for a,b in zip(before,after)]==[False,True,False,False]
    config.data['sapiensid']['model']['variant']='wb4m'
    assert [a!=b for a,b in zip(after,stages())]==[False,False,False,True]


def test_face_adapter_deduplicates_without_reusing_quality_policy(tmp_path):
    from stash_ai.recognition.models import YuNetDetector
    from tests.helpers.recognition import FACE
    from stash_ai.recognition.quality import assess,rejection
    config=SimpleNamespace(data=defaults())
    owner=YuNetDetector.__new__(YuNetDetector);owner.config=config;owner.provenance={}
    calls=[]
    owner._detect_faces_many=lambda images:(calls.append(len(images)) or [[FACE] for _ in images])
    with closing(Store(tmp_path/'cache.sqlite',1024**2)) as store,using(store):
        with Image.new('RGB',(100,100),'red') as image:
            first,second=owner.detect_faces_many([image,image.copy()])
            assert calls==[1] and first==second
            q=assess(image,first[0]);config.data['detection']['min_face_pixels']=10000
            assert assess(image,first[0])==q
            assert rejection(q,config.data['detection']) is not None
            owner.detect(image);assert calls==[1]


def test_corrupt_or_overbudget_cache_is_disposable(tmp_path):
    with closing(Store(tmp_path/'cache.sqlite',256*1024)) as store:
        for n in range(80):store.put(str(n),{'data':'x'*8192})
        assert store.used<=store.limit and store.db.execute('PRAGMA page_count').fetchone()[0]*4096<=store.limit
        store.put('last',{'ok':True});store.db.execute("UPDATE entries SET payload=? WHERE key='last'",(b'bad',))
        assert store.get('last') is None


def test_quality_cache_context_is_available_on_cpu_workers(tmp_path):
    from stash_ai.recognition.scan_pipeline import CpuStage
    from stash_ai.recognition.inference_cache import current
    with closing(Store(tmp_path/'cache.sqlite',1024**2)) as store,using(store),closing(CpuStage(2,1024**2)) as stage:
        with Image.new('RGB',(8,8)) as image:
            assert list(stage.ordered(lambda pixels,value:current() is store,image,[1,2]))==[True,True]


def test_pose_cache_survives_quality_retuning_and_boxes_change(tmp_path):
    from stash_ai.recognition.cached_pose import Pose
    cfg=SimpleNamespace(data=defaults());cfg.data['sapiensid']['pose']['backend']='vitpose_plus_large'
    calls=[]
    model=SimpleNamespace(predict_many=lambda frames:(calls.append(len(frames)) or
        [(np.zeros((len(boxes),17,2),dtype=np.float32),np.ones((len(boxes),17),dtype=np.float32)) for _,boxes in frames]))
    with closing(Store(tmp_path/'cache.sqlite',1024**2)) as store,using(store):
        with Image.new('RGB',(32,32)) as image:
            bounds=[[0,0,16,16]]
            Pose(model,cfg,'cpu').predict_many([(image,bounds),(image.copy(),bounds)])
            cfg.data['sapiensid']['collection']['min_sharpness']+=1
            Pose(model,cfg,'cpu').predict(image,bounds)
            assert calls==[1]
            Pose(model,cfg,'cpu').predict(image,[[0,0,17,16]])
            assert calls==[1,1]


def test_duplicate_frame_cache_preserves_cue_records_and_geometry(tmp_path):
    from stash_ai.recognition.models import YuNetDetector
    from stash_ai.recognition.scan_pipeline import CpuStage,process_cues
    from stash_ai.recognition.quality import assess,crop_bytes
    from stash_ai.recognition.scanning import geometry_contract
    from tests.helpers.recognition import FACE,textured_image
    config=SimpleNamespace(data=defaults())
    config.data['detection']['min_visible_face_fraction']=0
    config.data['clustering']['min_anchor_visible_face_fraction']=0
    config.data['clustering']['min_auxiliary_visible_face_fraction']=0
    owner=YuNetDetector.__new__(YuNetDetector);owner.config=config;owner.provenance={};owner.dimensions=4
    calls=[]
    owner._detect_faces_many=lambda images:(calls.append(len(images)) or [[FACE] for _ in images])
    owner.embed=lambda *args:np.array([1,0,0,0],dtype=np.float32)
    cues=[SimpleNamespace(rect=[0,0,160,160],seconds=0),SimpleNamespace(rect=[160,0,160,160],seconds=1)]
    with closing(Store(tmp_path/'cache.sqlite',1024**2)) as store,using(store),closing(CpuStage(1,16*1024**2)) as stage:
        with textured_image() as image,image.copy() as other:
            result=list(process_cues([(0,cues[0],image),(1,cues[1],other)],owner,None,config,stage=stage,
                assess=assess,rejection=lambda *args:None,crop_bytes=crop_bytes,geometry_contract=geometry_contract))
    assert calls==[1] and [r[0] for r in result]==[0,1]
    assert [r[1].seconds for r in result]==[0,1]
    assert result[0][2][0][2]['geometry_contract']!=result[1][2][0][2]['geometry_contract']
