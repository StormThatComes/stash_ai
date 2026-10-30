from types import SimpleNamespace
from dataclasses import replace
import pytest
import numpy as np
from PIL import Image


def test_detection_only_preserves_exact_boxes_without_running_pose_branches():
    torch = pytest.importorskip('torch')
    pytest.importorskip('ultralytics')
    from ultralytics.nn.modules.head import Detect, Pose26
    from stash_ai.recognition.person_detector import detection_only
    head = Pose26(nc=1, ch=(8,16), reg_max=1, end2end=True).eval()
    head.stride = torch.tensor([8.,16.]); head.end2end = False
    features = [torch.randn(1,8,8,8),torch.randn(1,16,4,4)]
    with torch.inference_mode():
        expected = head(features)[0][:,:5].clone()
        model = SimpleNamespace(model=SimpleNamespace(model=[head],task='pose'), task='pose',overrides={})
        detection_only(model)
        actual = head(features)[0]
    assert type(head) is Detect and model.task == 'detect'
    assert not any('cv4' in name for name in head._modules)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_vitpose_batches_people_across_images_and_preserves_order():
    torch = pytest.importorskip('torch')
    from transformers import BatchFeature
    from stash_ai.recognition.vitpose_pose import ViTPoseLarge
    calls = []
    class Processor:
        def __call__(self, *, images, boxes, return_tensors):
            size = sum(map(len,boxes)); calls.append((len(boxes),size))
            return BatchFeature({'pixel_values':torch.zeros(size,3,8,8)})
        def post_process_pose_estimation(self, outputs, *, boxes, threshold):
            return [[dict(keypoints=torch.tensor([[x+1,y+2]]*17),scores=torch.ones(17),labels=torch.arange(17))
                     for x,y,w,h in group] for group in boxes]
    def model(**inputs):
        assert len(inputs['dataset_index']) == len(inputs['pixel_values'])
        return None
    refiner = ViTPoseLarge(model,Processor(),device='cpu',batch_size=3)
    with Image.new('RGB',(100,100)) as a, Image.new('RGB',(100,100)) as b:
        result = refiner.predict_many([(a,[[0,0,10,10]]),(b,[[10,20,30,40],[30,40,50,60]]),(a,[])])
    assert calls == [(2,3)]
    np.testing.assert_array_equal(result[0][0][:,0], [[1,2]])
    np.testing.assert_array_equal(result[1][0][:,0], [[11,22],[31,42]])
    assert result[2][0].shape == (0,17,2)


def test_collector_batches_owned_frames_and_commits_complete_cues(tmp_path):
    from tests.test_recognition_scanning import config
    from tests.test_recognition_person_collection import Person
    from tests.helpers.recognition import Backend, MediaClient
    from stash_ai.recognition.config import plain
    from stash_ai.recognition.database import FaceDatabase
    from stash_ai.recognition.media import SpriteInput
    from stash_ai.recognition.person_collection import PersonCollector
    from stash_ai.recognition.scanning import scan_scene
    settings = config(tmp_path); data = plain(settings.data)
    data['execution']['checkpoint_cues'] = 100
    data['sapiensid']['enabled'] = True
    data['sapiensid']['pose'].update(backend='vitpose_plus_large',batch_size=4)
    settings = replace(settings,data=data)
    class Batched(Person):
        def detect_many(self, frames):self.frames=list(frames)
    model = Batched(); client = MediaClient()
    collector = PersonCollector(settings,{},factory=lambda *args:model)
    try:
        with FaceDatabase(settings.database) as state, SpriteInput(client,client.scene,settings) as source:
            scan_scene(state,'scope',client.scene,settings,'ns',source,Backend,on_cue=collector.collect_frame)
            assert state.connection.execute('SELECT COUNT(*) FROM person_evidence').fetchone()[0] == 3
            assert state.connection.execute('SELECT cursor FROM scans').fetchone()[0] == 3
        assert len(model.frames) == 3 and collector.pending_bytes() == 0
        for image,_ in model.frames:
            with pytest.raises(ValueError):image.getpixel((0,0))
    finally:
        collector.close()
