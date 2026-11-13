"""Foreground blur cannot be rescued by textured backgrounds or mask edges."""
import json
from types import SimpleNamespace

import numpy as np
from PIL import Image
import pytest

from stash_ai.recognition.foreground_quality import laplacian
from stash_ai.recognition.visibility import measurement, device_measurements
from tests.helpers.recognition import FACE, Backend, MediaClient, Visibility
from tests.test_recognition_body_mask_overlap import person


def blob_from_gray(gray):
    rgb=np.repeat(np.asarray(gray,dtype=np.float32)[...,None],3,axis=2)/255
    rgb=(rgb-np.asarray([.485,.456,.406],dtype=np.float32))/np.asarray([.229,.224,.225],dtype=np.float32)
    return np.ascontiguousarray(rgb.transpose(2,0,1)[None])


def test_body_mask_ignores_background_without_masking_the_identity_input():
    from stash_ai.recognition.body_collection import quality
    mask=np.indices((32,32))[1]<16
    pixels=(np.indices((32,32)).sum(axis=0)%2*200+25).astype('uint8')
    pixels[mask]=128
    selected=person([0,0,32,32],mask)
    raw=person([0,0,32,32])
    policy=dict(min_person_pixels=1,min_keypoints=1,min_sharpness=10,max_overlap=1)
    with Image.fromarray(pixels).convert('RGB') as image:
        measured,reason=quality(image,selected,[selected],policy)
        original,_=quality(image,raw,[raw],policy)
        assert measured['sharpness']==0 and reason=='person_sharpness'
        assert measured['sharpness_method']=='sam_mask' and measured['sharpness_pixels']>0
        assert original['sharpness']>10 and original['sharpness_method']=='body_crop'
        assert measured['pixel_digest']==original['pixel_digest']
        selected['pose_refinement']['encoder_mask']=selected['pose_refinement']['overlap_mask']
        masked,_=quality(image,selected,[selected],policy)
        assert masked['sharpness']==measured['sharpness']
        assert masked['sharpness_pixels']==measured['sharpness_pixels']


def test_face_ellipse_excludes_sharp_crop_corners():
    from stash_ai.recognition.quality import assess
    rows,cols=np.indices((160,160))
    background=((cols-80)/57.6)**2+((rows-80)/61.44)**2>1.3
    values=np.full((160,160),128,dtype='uint8')
    values[background]=(rows[background]+cols[background])%2*200+25
    with Image.fromarray(values).convert('RGB') as image:
        measured=assess(image,FACE)
    assert measured['crop_sharpness']>10
    assert measured['sharpness']<1e-5
    assert measured['sharpness_method']=='face_ellipse'


@pytest.mark.parametrize('empty',[False,True])
def test_faceocc_mask_ignores_textured_occluder_and_hard_mask_boundary(empty):
    rows,cols=np.indices((256,256))
    mask=(rows>60)&(rows<190)&(cols>60)&(cols<190)
    values=(rows+cols)%2*200+25
    values[mask]=128
    if empty:mask[:]=False
    measured=measurement(np.where(mask,1.,-1.)[None,None],np.ones_like(mask),'cpu',blob=blob_from_gray(values))
    assert measured['sharpness']<1e-5
    assert measured['sharpness_method']=='faceocc_mask'
    assert (measured['sharpness_pixels']==0)==empty


def test_blurring_foreground_lowers_measurement_while_background_stays_sharp():
    import cv2
    values=np.random.default_rng(7).integers(20,230,(100,100)).astype('float32')
    mask=np.zeros((100,100),bool);mask[10:90,10:90]=True
    blurred=cv2.blur(values,(15,1))
    blurred[~mask]=values[~mask]
    assert laplacian(blurred,mask)['sharpness']<laplacian(values,mask)['sharpness']/5


def test_device_mask_sharpness_matches_cpu_without_transferring_images(monkeypatch):
    import torch
    rng=np.random.default_rng(5)
    pixels=rng.integers(10,240,(256,256))
    blobs=np.concatenate([blob_from_gray(pixels),blob_from_gray(np.full_like(pixels,128))])
    masks=np.zeros((2,256,256),bool);masks[0,30:200,40:220]=True
    logits=np.where(masks,1.,-1.).astype('float32')[:,None]
    regions=[np.ones((256,256),bool)]*2
    copied=[];original=torch.Tensor.cpu
    def cpu(tensor,*args,**kwargs):
        copied.append(tensor.numel());return original(tensor,*args,**kwargs)
    monkeypatch.setattr(torch.Tensor,'cpu',cpu)
    actual=device_measurements(torch.from_numpy(logits),regions,'cpu',inputs=torch.from_numpy(blobs))
    expected=[measurement(logits[i:i+1],regions[i],'cpu',blob=blobs[i:i+1]) for i in range(2)]
    for a,b in zip(actual,expected):
        assert a['sharpness']==pytest.approx(b['sharpness'],rel=1e-5,abs=1e-5)
        assert a['sharpness_pixels']==b['sharpness_pixels']
        assert a['visible_face_fraction']==b['visible_face_fraction']
    assert copied==[2,4]


@pytest.mark.parametrize('batch_size',[1,2])
@pytest.mark.parametrize('ellipse,foreground,reason',[(0.,200.,None),(200.,0.,'face_blurred')])
def test_scan_applies_face_threshold_after_semantic_mask(tmp_path,monkeypatch,batch_size,ellipse,foreground,reason):
    from stash_ai.recognition import scanning
    from stash_ai.recognition.database import FaceDatabase
    from stash_ai.recognition.media import SpriteInput
    from tests.test_recognition_detection_quality import settings
    from dataclasses import replace
    from stash_ai.recognition.config import plain
    config=settings(tmp_path,min_sharpness=10,min_visible_face_fraction=.5)
    data=plain(config.data);data['execution']['recognizer_batch_size']=batch_size
    config=replace(config,data=data)
    assess=scanning.assess
    monkeypatch.setattr(scanning,'assess',lambda *a:dict(assess(*a),sharpness=ellipse))
    class ForegroundVisibility(Visibility):
        def measure(self,*args):
            return dict(super().measure(*args),sharpness=foreground,sharpness_method='faceocc_mask')
    client=MediaClient();backend=Backend()
    with FaceDatabase(config.database) as state,SpriteInput(client,client.scene,config) as source:
        scanning.scan_scene(state,'scope',client.scene,config,'ns',source,lambda:backend,visibility=ForegroundVisibility())
        rows=state.connection.execute('SELECT quality,rejected,embedding FROM observations').fetchall()
        assert len(rows)==3
        for row in rows:
            assert row['rejected']==reason
            assert (row['embedding'] is None)==(reason is not None)
            quality=json.loads(row['quality'])
            assert quality['sharpness']==foreground and quality['sharpness_method']=='faceocc_mask'


def test_body_sharpness_contract_refresh_preserves_previous_mask_overlap_banks():
    from stash_ai.recognition.config import defaults,plain,digest
    from stash_ai.recognition.person_signal import contract as person_contract
    from stash_ai.recognition.body_observations import contract,compatible_contracts
    data=defaults();data['sapiensid']['pose']['backend']='pmpose'
    data['sapiensid']['pose']['refinement']['enabled']=True
    config=SimpleNamespace(data=data)
    previous=person_contract(config,legacy_foreground_sharpness=True)
    assert previous!=person_contract(config)
    collection=plain(data['sapiensid']['collection']);collection.pop('enabled')
    old=digest(['independent-person-v4-independent-torso',previous,collection])
    assert old in compatible_contracts(config)[1:] and old!=contract(config)


def test_face_scan_and_portrait_policies_refresh_even_without_faceocc(tmp_path):
    from dataclasses import replace
    from stash_ai.recognition.config import digest,plain
    from stash_ai.recognition.scanning import scan_identity
    from stash_ai.recognition.portraits import policy_identity,_policy_value
    from tests.test_recognition_scanning import config as configuration
    config=configuration(tmp_path);data=plain(config.data)
    data['portraits']['weights']['visibility']=0
    config=replace(config,data=data)
    old_scan=digest(dict(namespace='ns',detection=data['detection'],
        max_cues=data['execution']['max_cues'],quality_format=3,visibility_contract=None))
    assert scan_identity(config,'ns')!=old_scan
    old_portrait=dict(_policy_value(config),version=8)
    assert old_portrait['visibility_contract'] is None
    assert policy_identity(config)!=digest(old_portrait)
