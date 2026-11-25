"""Recognition quality follows real encoder pixels and independent mask geometry."""
from dataclasses import replace
import json

import cv2
import numpy as np
from PIL import Image
import pytest
import torch

from stash_ai.recognition.blur_quality import measure
from stash_ai.recognition.model_input_quality import body, project_masks, SPACE
from tests.helpers.recognition import Backend, FACE, MediaClient, textured_image


def checker(size, block):
    return (40+175*((np.indices((size, size))//block).sum(axis=0)%2)).astype('uint8')


def test_upscaled_small_crop_loses_detail_at_encoder_scale():
    large = checker(384, 24)
    small = cv2.resize(large, (48, 48), interpolation=cv2.INTER_AREA)
    with Image.fromarray(large).convert('RGB') as full, Image.fromarray(small).convert('RGB') as tiny:
        full_quality, small_quality = body(full), body(tiny)
    assert measure(small)['sharpness'] > measure(large)['sharpness']
    assert small_quality['sharpness'] < full_quality['sharpness']*.7
    assert small_quality['sharpness_scale'] == 8
    assert small_quality['sharpness_input_size'] == [384, 384]


def test_reduction_can_improve_effective_sharpness():
    pixels = cv2.GaussianBlur(checker(768, 32), (0, 0), 2)
    with Image.fromarray(pixels).convert('RGB') as image:
        quality = body(image)
    assert quality['sharpness'] > measure(pixels)['sharpness']*1.4
    assert quality['sharpness_scale'] == .5


@pytest.mark.parametrize('recognizer', ['auraface', 'lvface_b', 'adaface_ir50', 'sface'])
def test_recognizer_pixels_restore_rgb_for_quality(recognizer):
    from stash_ai.recognition.aura import AuraFaces, input_blob as aura_blob
    from stash_ai.recognition.onnx_recognizers import OnnxEmbeddingFaces, input_blob as onnx_blob
    from stash_ai.recognition.models import OpenCVFaces
    bgr = np.random.default_rng(32).integers(0, 256, (112,112,3), dtype=np.uint8)
    if recognizer == 'auraface':
        pixels = AuraFaces.embedding_pixels(None, aura_blob(bgr))
    elif recognizer == 'sface':
        pixels = OpenCVFaces.embedding_pixels(None, bgr)
    else:
        backend = object.__new__(OnnxEmbeddingFaces);backend.recognizer_key = recognizer
        pixels = backend.embedding_pixels(onnx_blob(bgr, recognizer=recognizer))
    np.testing.assert_allclose(pixels, bgr[:,:,::-1], atol=1e-5)


@pytest.mark.parametrize('size', [(71, 120), (241, 119), (512, 700)])
def test_body_pixels_match_encoder_with_odd_square_padding(size):
    from stash_ai.recognition.sapiensid_v1 import preprocess, square_resize
    width, height = size
    pixels = checker(max(size), 6)[:height, :width]
    with Image.fromarray(pixels).convert('RGB') as image:
        actual = body(image)
        tensor = preprocess(image)[0]
    gray = (((tensor+1)*127.5)*tensor.new_tensor([.299,.587,.114])[:,None,None]).sum(0).numpy()
    valid = square_resize(torch.ones(1,height,width), fill=0)[0].numpy() >= 1-1e-6
    expected = measure(gray, valid)
    assert actual['sharpness'] == pytest.approx(expected['sharpness'], abs=1e-4)
    assert actual['sharpness_pixels'] == expected['sharpness_pixels']
    # White square padding next to a constant person must not create edges.
    with Image.new('RGB', size, (80,80,80)) as flat:
        assert body(flat)['sharpness'] == 0


@pytest.mark.parametrize('size', [64, 384, 600])
def test_resized_mask_excludes_holes_and_external_texture(size):
    pixels = checker(size, max(2,size//32))
    mask = np.zeros((size,size), bool)
    mask[size//8:-size//8,size//8:-size//8] = True
    mask[size//3:size//2,size//3:size//2] = False
    with Image.fromarray(pixels).convert('RGB') as image:
        before = body(image,mask)
    pixels[~mask] = np.random.default_rng(72).integers(0,256,(~mask).sum())
    with Image.fromarray(pixels).convert('RGB') as image:
        after = body(image,mask)
    assert before == after


def test_encoder_mask_outline_is_not_sharpness_even_in_full_crop_mode():
    mask = np.zeros((96,128),bool);mask[16:80,24:104] = True
    with Image.new('RGB',(128,96),(40,40,40)) as image:
        for selected in (None,mask):
            value = body(image,selected,mask)
            assert value['sharpness'] == 0 and value['sharpness_pixels'] > 0


def test_face_encoder_mask_projection_cpu_device_parity_and_scalar_transfers(monkeypatch):
    from stash_ai.recognition.visibility import prepare_measurement, measurement, device_measurements
    with textured_image() as image:
        prepared = prepare_measurement(image, FACE, 0.)
    blob, region, count = prepared
    logits = np.ones((1,1,256,256),np.float32)
    logits[:,:,90:125,110:160] = -1
    expected = measurement(logits,region,'cpu',blob=blob,face_pixels=count,sharpness_input=prepared.sharpness_input)
    copied=[];original=torch.Tensor.cpu
    def cpu(tensor,*args,**kwargs):
        copied.append(tensor.numel());return original(tensor,*args,**kwargs)
    monkeypatch.setattr(torch.Tensor,'cpu',cpu)
    actual = device_measurements(torch.from_numpy(logits),[region],'cpu',inputs=torch.from_numpy(blob),
        face_pixels=[count],sharpness_inputs=[prepared.sharpness_input])[0]
    for name in ('sharpness','portrait_sharpness','sharpness_pixels','sharpness_noise','visible_face_fraction'):
        assert actual[name] == pytest.approx(expected[name],abs=1e-4)
    assert copied == [1,7,7]
    assert actual['sharpness_input_size'] == [112,112] and actual['sharpness_space'] == SPACE
    projected = project_masks((logits[:,0]>0)&region[None],[prepared.sharpness_input])[0]
    assert measure(prepared.sharpness_input[0],projected)['sharpness'] == pytest.approx(actual['sharpness'],abs=1e-4)


def test_device_batch_keeps_each_items_measurement_space():
    from stash_ai.recognition.visibility import prepare_measurement, measurement, device_measurements
    with textured_image() as image:
        prepared = prepare_measurement(image, FACE, 0.)
    blob, region, count = prepared
    logits = np.ones((1,1,256,256),np.float32)
    actual = device_measurements(torch.from_numpy(np.repeat(logits,2,axis=0)),[region,region],'cpu',
        inputs=torch.from_numpy(np.repeat(blob,2,axis=0)),face_pixels=[count,count],
        sharpness_inputs=[None,prepared.sharpness_input])
    for index,quality_input in enumerate((None,prepared.sharpness_input)):
        expected = measurement(logits,region,'cpu',blob=blob,face_pixels=count,sharpness_input=quality_input)
        assert actual[index]['sharpness'] == pytest.approx(expected['sharpness'],abs=1e-4)
        assert actual[index].get('sharpness_space') == expected.get('sharpness_space')


@pytest.mark.parametrize('flat', [False, True])
@pytest.mark.parametrize('batch_size', [1,3])
@pytest.mark.parametrize('masked', [False, True])
def test_scan_gates_actual_prepared_pixels_and_reuses_encoder_tensor(tmp_path,flat,batch_size,masked):
    from stash_ai.recognition.aura import input_blob, AuraFaces
    from stash_ai.recognition.database import FaceDatabase
    from stash_ai.recognition.media import SpriteInput
    from stash_ai.recognition.scanning import scan_scene
    from stash_ai.recognition.visibility import FaceVisibility
    from stash_ai.recognition.config import plain
    from tests.test_recognition_detection_quality import settings
    config = settings(tmp_path,min_sharpness=5,min_visible_face_fraction=.5)
    data = plain(config.data)
    if not masked:
        data['detection']['min_visible_face_fraction'] = 0
        data['clustering'].update(min_anchor_visible_face_fraction=0,min_auxiliary_visible_face_fraction=0)
    data['execution'].update(recognizer_batch_size=batch_size,scan_cpu_workers=2,detector_batch_size=batch_size)
    config = replace(config,data=data)
    class Prepared(Backend):
        prepared = 0
        tensors = []
        def prepare_embedding(self,image,face):
            self.prepared += 1
            gray = np.full((112,112),100,np.uint8) if flat else checker(112,5)
            result = input_blob(np.repeat(gray[:,:,None],3,axis=2))
            self.tensors.append(result)
            return result
        embedding_pixels = AuraFaces.embedding_pixels
        def embed_prepared_many(self,tensors):
            assert all(any(t is original for original in self.tensors) for t in tensors)
            self.calls += len(tensors)
            return [np.asarray(self.vector,np.float32) for _ in tensors]
    visibility = object.__new__(FaceVisibility)
    visibility.device='cpu';visibility.minimum_source_fraction=0.;visibility.logit_threshold=0.;visibility.diagnostics=False
    visibility.model=lambda pixels: torch.ones((len(pixels),1,256,256))
    backend,client = Prepared(),MediaClient()
    if not masked:
        visibility = None
    with FaceDatabase(config.database) as db, SpriteInput(client,client.scene,config) as source:
        scan_scene(db,'scope',client.scene,config,'ns',source,lambda:backend,visibility=visibility)
        rows = db.connection.execute('SELECT rejected,quality,embedding FROM observations').fetchall()
    assert backend.prepared == 3 and backend.calls == (0 if flat else 3)
    for reason,raw,embedding in rows:
        quality = json.loads(raw)
        assert quality['sharpness_space'] == SPACE and quality['sharpness_input_size'] == [112,112]
        assert quality['portrait_sharpness'] > 0
        assert reason == ('face_blurred' if flat else None)
        assert (embedding is None) is flat


def test_portrait_preference_is_independent_of_recognition_sharpness():
    from stash_ai.recognition.quality import portrait_components
    from stash_ai.recognition.config import defaults
    from stash_ai.recognition.body_portraits import score
    policy = defaults()['portraits']
    quality = dict(sharpness=0.,portrait_sharpness=35.,pixels=128,pose_deviation=0.,exposure=128.,clipped_fraction=0.)
    assert portrait_components(quality,policy)['sharpness'] == portrait_components(dict(quality,sharpness=100.),policy)['sharpness']
    assert score(quality) == score(dict(quality,sharpness=100.))


def test_prior_sharpness_space_is_a_frozen_contract_even_without_masks():
    from types import SimpleNamespace
    from stash_ai.recognition.config import defaults,plain,digest
    from stash_ai.recognition.person_signal import contract as person_contract
    from stash_ai.recognition.body_observations import contract,compatible_contracts
    data=defaults();data['sapiensid']['pose']['refinement']['enabled']=False
    config=SimpleNamespace(data=data)
    collection=plain(data['sapiensid']['collection']);collection.pop('enabled')
    for key in ('sharpness_mode', 'min_focus_sharpness', 'min_motion_sharpness'):collection.pop(key)
    previous=digest(['independent-person-v9-directional-sharpness',person_contract(config,legacy_input_sharpness=True),collection])
    assert contract(config)!=previous and previous in compatible_contracts(config)[1:]
