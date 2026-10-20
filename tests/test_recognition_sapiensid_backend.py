"""Person association, geometry, namespaces and production scan integration."""
from dataclasses import replace
from threading import get_ident
from types import SimpleNamespace
import json

import numpy as np
from PIL import Image
import pytest

from stash_ai.recognition.config import defaults, load, validate, _freeze
from stash_ai.recognition.models import checked_detection, namespace, asset_selection, model_provenance
from stash_ai.recognition.sapiensid_backend import SapiensIDFaces, associate, adjust_pose, CONTRACT
from tests.helpers.recognition import FACE, Backend, MediaClient


def settings(tmp_path):
    data = defaults()
    data['models'].update(recognizer='sapiensid_v1', device='cpu')
    data['sapiensid'].update(source_directory='upstream', checkpoint_directory='wb4m',
                            model_yaml_sha256='a'*64, pose_path='pose.pt', pose_sha256='b'*64)
    data['detection'].update(min_sharpness=0, max_pose_deviation=None,
                             min_visible_face_fraction=0, require_complete_face=False)
    config = load(root=tmp_path)
    return replace(config, data=_freeze(data))


def person(bounds=(0,0,160,160), nose=(80,84)):
    points = [[-1.,-1.] for _ in range(17)]
    points[0] = list(nose)
    points[5],points[6] = [120.,140.],[40.,140.]
    return {'bounds':list(bounds), 'keypoints':points, 'score':.99}


def backend(config, faces, people):
    """Exercise the real backend without external weights or network."""
    torch = pytest.importorskip('torch')
    from stash_ai.recognition.sapiensid_inference import SapiensIDV1Encoder
    class Model(torch.nn.Module):
        def forward(self, images, *, foreground_masks, ldmks):
            self.seen = images.clone(), ldmks.clone(), foreground_masks.clone()
            output = torch.zeros((len(images),4096),device=images.device)
            output[:,0] = 1
            return output
    model = Model()
    result = SimpleNamespace(
        boxes=SimpleNamespace(xyxy=torch.tensor([p['bounds'] for p in people],dtype=torch.float32).reshape(-1,4),
                              conf=torch.tensor([p['score'] for p in people])),
        keypoints=SimpleNamespace(xy=torch.tensor([p['keypoints'] for p in people],dtype=torch.float32).reshape(-1,17,2),
                                  conf=torch.ones((len(people),17))))
    class Pose:
        calls = 0
        def predict(self, **kwargs):
            self.calls += 1
            return [result]
    instance = object.__new__(SapiensIDFaces)
    instance.owner = get_ident()
    instance.config = config
    instance.device = 'cpu'
    instance.pose = Pose()
    instance.encoder = SapiensIDV1Encoder(model,None,device='cpu')
    instance.detector = None
    instance.ort_detector = SimpleNamespace(detect=lambda data:faces,close=lambda:None)
    instance._person_frame = None
    return instance, model, result


def test_selectable_with_required_local_assets_and_no_extra_face_asset(tmp_path):
    c = settings(tmp_path)
    from stash_ai.recognition.config import plain
    validate(plain(c.data))
    assert 'sapiensid_v1' not in asset_selection(c)[0]
    assert model_provenance(c)['sapiensid_v1']['contract'] == CONTRACT
    assert namespace(c) != namespace(load(root=tmp_path))
    modified = defaults()
    modified.update({k:dict(v) for k,v in c.data.items() if k in ('models','sapiensid')})
    modified['sapiensid']['pose_sha256'] = 'c'*64
    assert namespace(replace(c,data=_freeze(modified))) != namespace(c)
    modified['sapiensid']['person_score_threshold'] = .7
    assert namespace(replace(c,data=_freeze(modified))) != namespace(c)


@pytest.mark.parametrize('field,value', [('source_directory',None),('pose_path',''),('pose_sha256','bad'),
                                         ('person_score_threshold',float('nan')),('keypoint_score_threshold',True)])
def test_bad_configuration_fails_before_model_loading(tmp_path,field,value):
    from stash_ai.recognition.config import plain
    data = plain(settings(tmp_path).data)
    data['sapiensid'][field] = value
    with pytest.raises(ValueError,match='sapiensid'):
        validate(data)


def test_unique_measured_head_association_never_uses_detection_order():
    second = checked_detection([176,16,128,128,209,60,271,60,240,84,216,108,264,108,.99])
    a, b = person(), person((160,0,320,160),(240,84))
    assert associate([FACE,second],[b,a]) == [1,0]
    assert associate([FACE],[person(nose=(150,150))]) == [None]
    assert associate([FACE],[a,a]) == [None]
    assert associate([FACE,FACE],[a]) == [None,None]
    assert associate([FACE],[]) == [None]


def test_person_crop_and_22_real_landmarks_feed_model_and_keep_face_geometry(tmp_path):
    c = settings(tmp_path)
    p = person((8,4,156,156))
    adapter, model, _ = backend(c,[FACE],[p])
    with Image.new('RGB',(160,160),(50,80,120)) as image:
        assert adapter.detect(image) == [FACE]
        vector = adapter.embed(image,FACE)
        assert adapter.pose.calls == 1
        assert vector.shape == (4096,) and vector[0] == 1
        batch,points,mask = model.seen
        assert batch.shape == (1,3,384,384) and mask.dtype == adapter.encoder.torch.bool
        # 148x152 body crop: 2px horizontal square padding before normalization.
        expected = (np.array(FACE.landmarks)-[8,4]+[2,0])/152
        np.testing.assert_allclose(points[0,17:].numpy(),expected,atol=1e-6)
        np.testing.assert_allclose(points[0,0].numpy(),[(80-8+2)/152,(84-4)/152],atol=1e-6)
        assert (points[0,1:5] == -1).all()
        quality = adapter.observation_quality(image,FACE)
        assert quality['person']['bounds'] == [8,4,156,156]
        assert quality['identity_rejection'] is None
    adapter.close()


def test_ambiguous_body_is_withheld_without_inference(tmp_path):
    adapter, model, _ = backend(settings(tmp_path),[FACE],[person(),person()])
    with Image.new('RGB',(160,160)) as image:
        adapter.detect(image)
        assert adapter.observation_quality(image,FACE)['identity_rejection'] == 'person_face_association_unavailable'
        with pytest.raises(ValueError,match='unambiguous'):
            adapter.embed(image,FACE)
        assert not hasattr(model,'seen')
    adapter.close()


@pytest.mark.parametrize('invalid', ['nan','missing_points','bad_score','empty_box'])
def test_malformed_pose_is_not_silently_reused(tmp_path,invalid):
    adapter, _, result = backend(settings(tmp_path),[FACE],[person()])
    if invalid == 'nan': result.keypoints.xy[0,0,0] = float('nan')
    if invalid == 'missing_points': result.keypoints.xy = result.keypoints.xy[:,:16]
    if invalid == 'bad_score': result.boxes.conf[0] = 2
    if invalid == 'empty_box': result.boxes.xyxy[0] = 0
    with Image.new('RGB',(160,160)) as image, pytest.raises(ValueError):
        adapter.detect(image)
    adapter.close()


def test_pose_canonicalization_preserves_missing_pairs():
    points = [[-1.,-1.] for _ in range(17)]
    points[5],points[6] = [.3,.5],[.7,.5]
    actual = adjust_pose(points)
    assert actual[5] == [.7,.5] and actual[6] == [.3,.5]
    assert actual[1] == [-1.,-1.]


@pytest.mark.parametrize('ambiguous',[False,True])
def test_scan_persists_person_provenance_and_only_eligible_4096d_vectors(tmp_path,ambiguous):
    from stash_ai.recognition.database import FaceDatabase
    from stash_ai.recognition.media import SpriteInput
    from stash_ai.recognition.scanning import scan_scene
    c = settings(tmp_path)
    client = MediaClient(count=1)
    adapter, model, _ = backend(c,[FACE],[person()]*(2 if ambiguous else 1))
    with FaceDatabase(c.database) as db, SpriteInput(client,client.scene,c) as source:
        scan_scene(db,'scope',client.scene,c,namespace(c),source,lambda:adapter)
        row = db.connection.execute('SELECT * FROM observations').fetchone()
        assert json.loads(row['geometry']) == list(FACE.row)
        assert json.loads(row['quality'])['identity_input'] == CONTRACT
        if ambiguous:
            assert row['rejected'] == 'person_face_association_unavailable'
            assert row['embedding'] is None and not hasattr(model,'seen')
        else:
            assert row['rejected'] is None and row['dimensions'] == 4096
            assert len(row['embedding']) == 16384
        scan_scene(db,'scope',client.scene,c,namespace(c),source,lambda:pytest.fail('complete scan loaded a model'))


def test_detector_quality_gate_prevents_sapiens_inference(tmp_path):
    from stash_ai.recognition.database import FaceDatabase
    from stash_ai.recognition.media import SpriteInput
    from stash_ai.recognition.scanning import scan_scene
    from stash_ai.recognition.config import plain
    c = settings(tmp_path)
    data = plain(c.data); data['detection']['min_sharpness'] = 100000
    c = replace(c,data=_freeze(data))
    client = MediaClient(count=1)
    adapter, model, _ = backend(c,[FACE],[person()])
    with FaceDatabase(c.database) as db, SpriteInput(client,client.scene,c) as source:
        scan_scene(db,'scope',client.scene,c,namespace(c),source,lambda:adapter)
        assert db.connection.execute('SELECT rejected FROM observations').fetchone()[0] == 'face_blurred'
        assert not hasattr(model,'seen')


def test_normal_run_loads_one_backend_reuses_4096d_evidence_and_publishes(tmp_path,monkeypatch):
    from tests.test_recognition_run import setup,run
    from stash_ai.recognition.database import FaceDatabase
    base,client,services,_ = setup(tmp_path,monkeypatch)
    data = __import__('stash_ai.recognition.config',fromlist=['plain']).plain(settings(tmp_path).data)
    data['portraits']['enabled'] = False
    data['clustering']['same_scene_min_support'] = 1
    config = replace(base,data=_freeze(data))
    created = []
    def factory(*args):
        value,model,_ = backend(config,[FACE],[person()])
        created.append(value)
        return value
    services.backend = factory
    assert run(config,services,'1') == 0
    assert len(client.links['1']) == 2 and client.links['1'][0] == '77'
    assert len(created) == 1 and created[0].encoder is None
    writes = list(client.writes)
    assert run(config,services,'1') == 0
    assert client.writes == writes and len(created) == 1
    with FaceDatabase(config.database) as db:
        assert db.connection.execute('SELECT COUNT(*) FROM observations WHERE dimensions=4096').fetchone()[0] == 3
        provenance = json.loads(db.connection.execute('SELECT provenance FROM models').fetchone()[0])
        assert provenance['sapiensid_v1']['contract'] == CONTRACT


def test_no_pose_detections_are_retained_as_ineligible_faces(tmp_path):
    adapter,model,_ = backend(settings(tmp_path),[FACE],[])
    with Image.new('RGB',(160,160)) as image:
        assert adapter.detect(image) == [FACE]
        assert adapter.observation_quality(image,FACE)['identity_rejection'] is not None
    adapter.close()
