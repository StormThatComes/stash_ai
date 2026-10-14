"""Expression ranking, policy-independent measurements and safe image refresh."""
from io import BytesIO
import json
from types import SimpleNamespace

import numpy as np
from PIL import Image
import pytest
import yaml

from stash_ai.recognition.config import load
from stash_ai.recognition.database import FaceDatabase
from stash_ai.recognition.expression import (REQUIRED, LazyExpression, cached, associated, measurement)
from stash_ai.recognition.expression_onnx import prepare, decode_mesh, LANDMARKS_SUBSET, BLENDSHAPE_NAMES
from stash_ai.recognition.models import namespace
from stash_ai.recognition.portraits import policy_identity
from stash_ai.recognition.scanning import scan_identity
from tests.helpers.recognition import FACE
from tests.test_recognition_metadata import setup as video_setup, Decoder
from tests.test_recognition_portrait_removal import change, last_counts
from tests.test_recognition_vtt_portraits import setup as vtt_setup, run, image_writes

BASE = ('frontal', 'sharpness', 'resolution', 'exposure', 'visibility')


class Expression:
    def __init__(self, profiles=None):
        self.profiles = profiles or [{'jawOpen': .8}, {'jawOpen': .1}, {'jawOpen': .5}]
        self.calls, self.closed = 0, False

    def measure(self, image, face):
        profile = self.profiles[self.calls % len(self.profiles)]
        self.calls += 1
        return measurement({**dict.fromkeys(REQUIRED, 0.), **profile})

    def close(self):
        self.closed = True


def setup(tmp_path, monkeypatch, source='vtt', **weights):
    config, stash, services = (vtt_setup(tmp_path, monkeypatch)[:3] if source == 'vtt'
                                else video_setup(tmp_path, monkeypatch))
    config = change(config, weights={**dict.fromkeys(BASE, 0.), **weights})
    model = Expression()
    services.expression = lambda *args: model
    if source == 'vtt':
        # Distinct source pixels make changing the winning cue a real Stash update.
        pixels = np.random.default_rng(7).integers(30, 225, (1050, 500, 3), dtype='uint8')
        with Image.fromarray(pixels) as image:
            output = BytesIO(); image.save(output, format='PNG'); stash.sprite = output.getvalue()
        stash.vtt = ('WEBVTT\n\n' + '\n\n'.join(
            f'00:00:{i:02d}.000 --> 00:00:{i+1:02d}.000\nsprite.jpg#xywh=0,{350*i},500,350'
            for i in range(3))).encode()
    return config, stash, services, model


def winner(config):
    with FaceDatabase(config.database) as db:
        operation = db.connection.execute("SELECT payload FROM publications WHERE kind='image'").fetchone()
        identifier = json.loads(operation[0])['portrait']
        row = db.connection.execute('SELECT * FROM portraits WHERE id=?', (identifier,)).fetchone()
        return dict(row), json.loads(row['metadata'])


def no_inference(*args, **kwargs):
    pytest.fail('weight changes must reuse measurements and identity evidence')


def test_vtt_weights_refresh_owned_image_without_expression_or_recognition_inference(tmp_path, monkeypatch):
    config, stash, services, model = setup(tmp_path, monkeypatch, closed_mouth=2)
    assert run(config, services, '1') == 0
    assert winner(config)[1]['cue'] == 1
    assert model.calls == 3 and model.closed
    original = dict(stash.images)
    services.expression = services.backend = no_inference
    changed = change(config, weights={**dict.fromkeys(BASE, 0.), 'closed_mouth': 0, 'open_mouth': 2})
    assert namespace(config) == namespace(changed)
    assert scan_identity(config, namespace(config)) == scan_identity(changed, namespace(changed))
    assert policy_identity(config) != policy_identity(changed)
    assert run(changed, services, '1') == 0
    assert winner(config)[1]['cue'] == 0
    assert stash.images != original and len(image_writes(stash)) == 2
    assert last_counts(config)['portraits_updated'] == 1
    with FaceDatabase(config.database) as db:
        assert db.connection.execute('SELECT COUNT(*) FROM scans').fetchone()[0] == 1
        assert all(cached(json.loads(row[0])) for row in db.connection.execute('SELECT quality FROM observations'))
    writes = list(stash.writes)
    assert run(changed, services, '1') == 0 and stash.writes == writes


@pytest.mark.parametrize('weight,profiles,expected', [
    ('neutral_expression', [{'mouthSmileLeft': .9}, {}, {'jawOpen': .8}], 1),
    ('open_eyes', [{'eyeBlinkLeft': .8}, {'eyeBlinkRight': .6}, {'eyeBlinkLeft': .1}], 2),
    ('open_mouth', [{'jawOpen': .4}, {'jawOpen': .9}, {'jawOpen': .1}], 1),
])
def test_visible_cues_choose_expected_vtt_portrait(tmp_path, monkeypatch, weight, profiles, expected):
    config, _, services, _ = setup(tmp_path, monkeypatch, **{weight: 3})
    model = Expression(profiles)
    services.expression = lambda *args: model
    assert run(config, services, '1') == 0
    assert winner(config)[1]['cue'] == expected
    assert model.calls == 3 and model.closed


def test_video_weight_changes_reuse_actual_frame_scores_jpegs_and_embeddings(tmp_path, monkeypatch):
    config, stash, services, model = setup(tmp_path, monkeypatch, 'video', closed_mouth=2)
    assert run(config, services, '1') == 0
    _, initial = winner(config)
    assert initial['requested_seconds'] == 2 and initial['quality']['closed_mouth'] == .9
    assert model.calls == 2 and model.closed
    services.expression = services.backend = services.decoder = no_inference
    changed = change(config, weights={**dict.fromkeys(BASE, 0.), 'closed_mouth': 0, 'open_mouth': 2})
    assert run(changed, services, '1') == 0
    _, current = winner(config)
    assert current['requested_seconds'] == 0 and current['quality']['open_mouth'] == .8
    assert current['reused_from'] and current['source_type'] == 'video'
    assert all(decoder.closed for decoder in Decoder.instances)
    writes = list(stash.writes)
    assert run(changed, services, '1') == 0 and stash.writes == writes


@pytest.mark.parametrize('source', ['vtt', 'video'])
def test_expression_failure_keeps_a_portrait_and_closes_model(tmp_path, monkeypatch, source):
    config, stash, services = (vtt_setup(tmp_path, monkeypatch)[:3] if source == 'vtt'
                                else video_setup(tmp_path, monkeypatch))
    assert run(config, services, '1') == 0
    images, writes = dict(stash.images), list(stash.writes)
    model = Expression()
    model.measure = lambda *args: {'open_mouth': float('nan')}
    services.expression = lambda *args: model
    changed = change(config, weights={'closed_mouth': 2})
    assert run(changed, services, '1') == 0
    assert stash.images == images and stash.writes == writes and model.closed
    assert last_counts(config).get('portraits_failed', 0) == 0
    assert last_counts(config).get('portraits_removed', 0) == 0


@pytest.mark.parametrize('source', ['vtt', 'video'])
def test_imperfect_face_still_receives_expression_preferences(tmp_path, monkeypatch, source):
    config, stash, services, model = setup(tmp_path, monkeypatch, source, closed_mouth=2)
    config = change(config, min_face_pixels=200)
    assert run(config, services, '1') == 0 and stash.images
    assert model.calls > 0 and model.closed


def test_zero_expression_weights_do_not_initialize_model(tmp_path, monkeypatch):
    config, _, services = vtt_setup(tmp_path, monkeypatch)[:3]
    services.expression = no_inference
    assert run(config, services, '1') == 0
    assert 'expression' not in winner(config)[1]['quality']


def test_manual_portrait_is_preserved_after_enabling_expression_preferences(tmp_path, monkeypatch):
    config, stash, services = vtt_setup(tmp_path, monkeypatch)[:3]
    assert run(config, services, '1') == 0
    performer = stash.links['1'][-1]
    stash.images[performer] = b'manual photo'
    services.expression = no_inference
    assert run(change(config, weights={'closed_mouth': 2}), services, '1') == 0
    assert stash.images[performer] == b'manual photo'


def test_incompatible_expression_cache_recomputes_only_expression(tmp_path, monkeypatch):
    config, _, services, _ = setup(tmp_path, monkeypatch, closed_mouth=2)
    assert run(config, services, '1') == 0
    with FaceDatabase(config.database) as db:
        for row in db.connection.execute('SELECT id,quality FROM observations').fetchall():
            quality = json.loads(row['quality']); quality['expression']['contract'] = 'old'
            db.connection.execute('UPDATE observations SET quality=? WHERE id=?', (json.dumps(quality), row['id']))
        db.connection.commit()
    model = Expression()
    services.expression, services.backend = lambda *args: model, no_inference
    changed = change(config, weights={**dict.fromkeys(BASE, 0.), 'closed_mouth': 3})
    assert run(changed, services, '1') == 0 and model.calls == 3 and model.closed


def test_measurement_is_a_visible_activity_heuristic_not_neutral_class_probability():
    raw = dict.fromkeys(REQUIRED, 0.)
    raw.update(jawOpen=.7, eyeBlinkLeft=.8, mouthSmileRight=.6)
    result = measurement(raw)
    assert result['neutral_expression'] == pytest.approx(.3)
    assert result['closed_mouth'] == pytest.approx(.3) and result['open_mouth'] == .7
    assert result['open_eyes'] == pytest.approx(.2) and cached(result)
    raw['jawOpen'] = 0.; raw['mouthSmileRight'] = 0.
    assert measurement(raw)['neutral_expression'] == 1  # blinking is ranked separately
    result['open_eyes'] = .9
    assert not cached(result)
    for bad in ({}, {**raw, 'jawOpen': float('nan')}, {**raw, 'jawOpen': 1.1}):
        with pytest.raises(ValueError): measurement(bad)


def landmarks_for(face, *, offset=0.):
    landmarks = np.zeros((478,2), dtype='float32')
    for indexes, point in zip(((33,133), (362,263), (1,), (61,), (291,)), face.landmarks):
        for index in indexes:
            landmarks[index] = [point[0]+offset, point[1]]
    return landmarks


def test_mesh_is_associated_to_verified_face_and_rejects_neighbor():
    with Image.new('RGB', (160,160), (20,30,40)) as image:
        pixels, transform = prepare(image, FACE)
    assert pixels.shape == (1,256,256,3) and pixels.dtype == np.float32
    assert pixels[0,128,128].tolist() == pytest.approx([20/255,30/255,40/255])
    points = landmarks_for(FACE)
    associated(points, FACE)
    with pytest.raises(ValueError, match='associated'):
        associated(landmarks_for(FACE,offset=100), FACE)
    points[1,0] = float('nan')
    with pytest.raises(ValueError, match='non-finite'):
        associated(points,FACE)


def test_lazy_expression_shares_and_closes_one_task():
    model = Expression()
    calls = []
    def factory(*args): calls.append(1); return model
    lazy = LazyExpression(None, {}, factory)
    for _ in range(3): assert cached(lazy.measure(None,None))
    lazy.close(); lazy.close()
    assert calls == [1] and model.calls == 3 and model.closed


@pytest.mark.parametrize('weight', ['neutral_expression','closed_mouth','open_mouth','open_eyes'])
@pytest.mark.parametrize('value', [-1,101,float('nan'),True])
def test_invalid_expression_weights_fail_before_work(tmp_path, weight, value):
    path=tmp_path/'bad.yml'; path.write_text(yaml.safe_dump({'version':1,'portraits':{'weights':{weight:value}}}))
    with pytest.raises(ValueError): load(path)
    assert not (tmp_path/'faces_state.sqlite3').exists()


def test_expression_asset_is_downloaded_only_when_enabled(tmp_path, monkeypatch):
    import stash_ai.recognition.models as models
    import stash_ai.recognition.runtime as runtime
    import stash_ai.backends.downloads as downloads
    calls=[]
    monkeypatch.setattr(runtime,'runtime',lambda *args: None)
    monkeypatch.setattr(models,'verify',lambda *args: None)
    monkeypatch.setattr(downloads,'install_hf_asset',lambda *a,**k: calls.append(a))
    config=load(root=tmp_path)
    assert not {'expression_mesh','expression_blendshapes'} & models.ensure_assets(config).keys()
    assert len(calls)==3
    path=tmp_path/'enabled.yml'; path.write_text('version: 1\nportraits: {weights: {closed_mouth: 2}}\n')
    assert {'expression_mesh','expression_blendshapes'} <= models.ensure_assets(load(path)).keys()
    assert len(calls)==8
    path.write_text('version: 1\nportraits: {enabled: false, weights: {closed_mouth: 2}}\n')
    assert not {'expression_mesh','expression_blendshapes'} & models.ensure_assets(load(path)).keys()
    assert len(calls)==10


def test_mesh_decoding_preserves_pixel_coordinates_and_validates_presence():
    with Image.new('RGB',(160,160)) as image:
        _,transform=prepare(image,FACE)
    source=landmarks_for(FACE)
    crop=np.linalg.solve(transform[:,:2],(source-transform[:,2]).T).T
    points=np.concatenate((crop,np.zeros((478,1))),axis=1).astype('float32').reshape(1,1,1,1434)
    mapped=decode_mesh(points,np.ones((1,1,1,1)),transform)
    assert np.allclose(mapped,source,atol=1e-5)
    associated(mapped,FACE)
    with pytest.raises(ValueError,match='confirm'):
        decode_mesh(points,-np.ones((1,1,1,1)),transform)
    with pytest.raises(ValueError,match='invalid'):
        decode_mesh(points,np.full((1,1,1,1),np.nan),transform)
    assert len(LANDMARKS_SUBSET)==len(set(LANDMARKS_SUBSET))==146
    assert len(BLENDSHAPE_NAMES)==52 and BLENDSHAPE_NAMES[25]=='jawOpen'


def test_expression_sessions_disable_cpu_node_fallback_when_cuda_is_requested(tmp_path):
    from stash_ai.recognition.runtime import session
    settings, calls = {}, []
    options=SimpleNamespace(add_session_config_entry=lambda key,value: settings.update({key:value}))
    def create(model,*,providers,**kwargs):
        calls.append(providers)
        return SimpleNamespace(get_providers=lambda: providers)
    ort=SimpleNamespace(SessionOptions=lambda: options, InferenceSession=create)
    _,device=session(load(root=tmp_path),b'model',runtime_info=(ort,['CUDAExecutionProvider','CPUExecutionProvider']),cpu_fallback=False)
    assert device=='cuda' and calls==[['CUDAExecutionProvider']]
    assert settings=={'session.disable_cpu_ep_fallback':'1'}
    def fallback(*args,**kwargs): return SimpleNamespace(get_providers=lambda: ['CPUExecutionProvider'])
    ort.InferenceSession=fallback
    with pytest.raises(ValueError,match='fell back'):
        session(load(root=tmp_path),b'model',runtime_info=(ort,['CUDAExecutionProvider','CPUExecutionProvider']),cpu_fallback=False)


def test_both_expression_networks_use_the_cuda_adapter_and_original_pixel_subset(tmp_path, monkeypatch):
    import stash_ai.recognition.expression_onnx as adapter
    from stash_ai.recognition.expression import FaceExpression
    calls=[]
    source=landmarks_for(FACE)
    with Image.new('RGB',(160,160)) as image:
        _,transform=prepare(image,FACE)
    crop=np.linalg.solve(transform[:,:2],(source-transform[:,2]).T).T
    points=np.concatenate((crop,np.zeros((478,1))),axis=1).astype('float32').reshape(1,1,1,1434)
    def item(name,shape): return SimpleNamespace(name=name,shape=shape,type='tensor(float)')
    class Model:
        def __init__(self,kind): self.kind=kind
        def get_providers(self): return ['CUDAExecutionProvider']
        def get_inputs(self):
            return [item('input_12',[1,256,256,3])] if self.kind=='mesh' else [item('serving_default_input_points:0',[1,146,2])]
        def get_outputs(self):
            return ([item('Identity',[1,1,1,1434]),item('Identity_1',[1,1,1,1]),item('Identity_2',[1,1])]
                    if self.kind=='mesh' else [item('StatefulPartitionedCall:0',[52])])
        def run(self,names,inputs):
            if self.kind=='mesh':
                assert inputs['input_12'].shape==(1,256,256,3)
                return [points,np.ones((1,1,1,1),dtype='float32')]
            subset=inputs['serving_default_input_points:0']
            assert subset.dtype==np.float32 and np.allclose(subset[0],source[list(LANDMARKS_SUBSET)],atol=1e-5)
            result=np.zeros(52,dtype='float32'); result[25]=.8
            return [result]
    def create(config,path,**kwargs):
        calls.append((config.data['models']['device'],kwargs['cpu_fallback']))
        return Model('mesh' if path==b'mesh' else 'blend'),'cuda'
    monkeypatch.setattr(adapter,'verified_bytes',lambda path,asset: str(path).encode())
    monkeypatch.setattr(adapter,'runtime',lambda *a: (SimpleNamespace(__version__='1.26.0'),['CUDAExecutionProvider']))
    monkeypatch.setattr(adapter,'session',create)
    model=FaceExpression(load(root=tmp_path),{'expression_mesh':'mesh','expression_blendshapes':'blend'})
    with Image.new('RGB',(160,160)) as image:
        result=model.measure(image,FACE)
    assert calls==[('cuda',False),('cuda',False)]
    assert model.device==result['expression']['device']=='cuda'
    assert result['open_mouth']==pytest.approx(.8) and cached(result)
    model.close()
