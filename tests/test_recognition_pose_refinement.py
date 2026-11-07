"""Mask ownership, missing-limb recovery and shared single/batch contracts."""
from contextlib import contextmanager
from types import SimpleNamespace

import numpy as np
from PIL import Image
import pytest

from stash_ai.recognition.config import defaults, validate
from stash_ai.recognition.pose_refinement import RefinedPMPose, check_mask
from stash_ai.recognition.pose_refinement_config import defaults as refinement_defaults
from tests.test_recognition_pmpose import fingerprints


BOX = [20, 10, 60, 90]


def pose_points():
    return np.array([[35,20], [32,18], [38,18], [30,20], [40,20], [28,35], [48,35],
                     [25,48], [52,48], [22,55], [55,55], [30,60], [45,60],
                     [30,75], [45,75], [30,85], [45,85]], dtype=np.float32)


def prediction(points):
    points = np.asarray(points, dtype=np.float32).reshape(-1, 17, 2)
    scores = np.full(points.shape[:2], .95, dtype=np.float32)
    return points, scores, {k: scores.copy() for k in ('presence', 'visibility')}


class Pose:
    batch_size = 2
    def __init__(self, drift=False):
        self.calls = []
        self.drift = drift

    def predict_many(self, frames):
        self.calls.append(('first', len(frames)))
        return [prediction([pose_points() + [b[0]-20, 0] for b in bounds]) for _, bounds in frames]

    def predict(self, image, bounds, masks=None):
        assert len(masks) == len(bounds) <= self.batch_size
        self.calls.append(('masked', len(bounds)))
        points = [pose_points() + [b[0]-20, 0] for b in bounds]
        for row in points:
            row[10] += [20, 0]  # Recover the right wrist beyond the original box.
            if self.drift:
                row[:7] += [25, 0]
        return prediction(points)


class Segmenter:
    def __init__(self, *, foreign=False, malformed=False, failure=False):
        self.frames = self.prompts = self.resets = 0
        self.foreign, self.malformed, self.failure = foreign, malformed, failure

    @contextmanager
    def frame(self, image):
        self.frames += 1
        self.size = image.size
        try:
            yield self
        finally:
            self.resets += 1

    def predict(self, points):
        self.prompts += 1
        if self.failure:
            raise RuntimeError('inference failed')
        w, h = self.size
        mask = np.zeros((h, w), dtype=bool)
        offset = int(points[0,0]-35)
        mask[10:90, 20+offset:80+offset] = True
        if self.foreign:
            mask[:, 100:140] = True
        return mask[:2] if self.malformed else mask, .95


def wrapper(**kwargs):
    return RefinedPMPose(Pose(), Segmenter(**kwargs), refinement_defaults())


def test_expands_box_recovers_wrist_and_retains_original_geometry_for_audit():
    from stash_ai.recognition.sapiensid_backend import SapiensIDFaces
    model = wrapper()
    with Image.new('RGB', (160,100)) as image:
        first, empty = model.predict_many([(image, [BOX]), (image, [])])
    assert empty[0].shape == (0,17,2) and empty[3] == []
    assert model.segmenter.frames == model.segmenter.resets == 1
    assert model.pose.calls == [('first', 2), ('masked', 1)]
    record = first[3][0]
    assert record['status'] == 'refined' and record['box_expanded']
    assert record['bounds'] == [20,10,80,90] and record['detector_bounds'] == BOX
    detector = object.__new__(SapiensIDFaces)
    data = defaults()
    data['sapiensid']['pose'].update(backend='pmpose')
    data['sapiensid']['pose']['refinement']['enabled'] = True
    detector.config = SimpleNamespace(data=data)
    person = detector._people([BOX], [.98], *first)[0]
    assert person['bounds'] == [20,10,80,90]
    assert person['keypoints'][10] == [75.,55.]
    assert person['score'] == .98  # Detection and mask confidences stay distinct.
    assert person['pose_refinement'] == record


@pytest.mark.parametrize('kind,reason', [('foreign','foreign_core_in_mask'), ('malformed',None), ('failure',None)])
def test_foreign_mask_rejected_but_corrupt_outputs_and_runtime_errors_propagate(kind, reason):
    model = wrapper(**{kind: True})
    with Image.new('RGB', (180,100)) as image:
        if kind != 'foreign':
            with pytest.raises((ValueError,RuntimeError)):
                model.predict(image, [BOX])
        else:
            result = model.predict(image, [BOX, [100,10,140,90]])
            assert result[3][0]['reason'] == reason
            assert result[3][0]['bounds'] == BOX
            np.testing.assert_array_equal(result[0][0], pose_points())
    assert model.segmenter.resets == 1


def test_bad_second_pose_keeps_both_original_box_and_original_points():
    model = wrapper()
    model.pose.drift = True
    with Image.new('RGB', (160,100)) as image:
        result = model.predict(image,[BOX])
    assert result[3][0]['reason'] == 'second_pass_core_drift'
    assert result[3][0]['bounds'] == BOX
    np.testing.assert_array_equal(result[0][0], pose_points())


def test_second_pass_batches_people_but_segments_frame_once():
    model = wrapper()
    boxes = [[20+i*80,10,60+i*80,90] for i in range(5)]
    with Image.new('RGB',(460,100)) as image:
        result = model.predict(image,boxes)
    assert model.pose.calls == [('first',1),('masked',2),('masked',2),('masked',1)]
    assert all(r['status']=='refined' for r in result[3])
    assert model.segmenter.frames == model.segmenter.resets == 1
    assert model.segmenter.prompts == 5


def test_overlap_mode_skips_isolated_boxes_and_detects_containment():
    model=wrapper()
    model.settings['mode']='overlap'
    with Image.new('RGB',(180,100)) as image:
        result=model.predict(image,[BOX])
    assert result[3][0]['reason']=='no_overlap'
    assert model.segmenter.frames==0
    from stash_ai.recognition.pose_refinement import overlaps
    assert overlaps([0,0,100,100],[10,10,20,20],.5)
    assert not overlaps([0,0,10,10],[10,0,20,10],0)


def test_no_reliable_or_ambiguous_core_points_skip_segmentation():
    model=wrapper()
    with Image.new('RGB',(160,100)) as image:
        result=model.predict(image,[BOX,BOX])
        assert all(r['reason']=='no_reliable_prompts' for r in result[3])
        model.settings['min_visibility']=1.
        result=model.predict(image,[BOX])
    assert model.segmenter.frames==0
    assert result[3][0]['status']=='skipped'


def test_expansion_disabled_still_refines_pose_without_changing_box():
    model=wrapper()
    model.settings['expand_boxes']=False
    with Image.new('RGB',(160,100)) as image:
        result=model.predict(image,[BOX])
    assert result[3][0]['status']=='refined'
    assert result[3][0]['bounds']==BOX
    assert not result[3][0]['box_expanded']


@pytest.mark.parametrize('case,reason', [('empty','empty_mask'),('low_score','low_mask_score'),
    ('missing','missing_prompt_support'),('big','excessive_expansion')])
def test_mask_admission_guards(case, reason):
    mask=np.zeros((100,160),dtype=bool)
    mask[10:90,20:80]=True
    score=.95
    if case=='empty':mask[:]=False
    if case=='low_score':score=.1
    if case=='missing':mask[10:30]=False
    if case=='big':mask[:]=True
    _,_,actual=check_mask(mask,score,pose_points()[:3],np.empty((0,2)),BOX,refinement_defaults(),(160,100))
    assert actual==reason


def test_speckles_do_not_expand_box_and_disconnected_hand_is_kept():
    mask=np.zeros((100,160),dtype=bool)
    mask[10:90,20:60]=True
    mask[50:55,70:75]=True  # Disconnected hand must survive component cleanup.
    mask[5,150]=True
    result,box,reason=check_mask(mask,.9,pose_points()[:3],np.empty((0,2)),BOX,refinement_defaults(),(160,100))
    assert reason is None and box==[20,10,75,90]
    assert result[52,72] and not result[5,150]


def test_mask_preparation_matches_upstream_alpha_before_affine_and_keeps_input_intact():
    import cv2
    from stash_ai.recognition.pmpose_geometry import prepare
    pixels=np.full((100,160,3),[101,151,201],dtype=np.uint8)
    mask=np.zeros((100,160),dtype=bool)
    mask[10:60,20:50]=True
    original=pixels.copy()
    expected=cv2.addWeighted(pixels,.75,pixels*mask[:,:,None],.25,0)
    with Image.fromarray(pixels) as image, Image.fromarray(expected) as masked:
        result=prepare(image,BOX,mask)[0]
        np.testing.assert_array_equal(result,prepare(masked,BOX)[0])
        np.testing.assert_array_equal(np.asarray(image),original)
        with pytest.raises(ValueError,match='full-frame boolean'):
            prepare(image,BOX,mask.astype(float))


@pytest.mark.parametrize('backend',['yolo','vitpose_plus_large','sapiens','pmpose'])
def test_disabled_refinement_preserves_historical_evidence(backend):
    data=defaults();data['sapiensid']['pose']['backend']=backend
    expected=fingerprints(data)
    data['sapiensid']['pose'].pop('refinement')
    assert fingerprints(data)==expected


def test_active_settings_invalidate_person_body_and_identity_contracts_but_paths_do_not():
    data=defaults();pose=data['sapiensid']['pose'];pose['backend']='pmpose'
    before=fingerprints(data);pose['refinement']['enabled']=True
    active=fingerprints(data)
    assert all(a!=b for a,b in zip(before,active))
    pose['refinement']['directory']='elsewhere'
    assert fingerprints(data)==active
    for key,value in [('mode','overlap'),('min_mask_score',.7),('expand_boxes',False),('max_side_expansion',.25)]:
        original=pose['refinement'][key];pose['refinement'][key]=value
        assert all(a!=b for a,b in zip(active,fingerprints(data)))
        pose['refinement'][key]=original


@pytest.mark.parametrize('key,value',[('enabled',1),('expand_boxes','true'),('mode','sometimes'),
    ('min_visibility',float('nan')),('min_mask_score',1.1),('min_presence',True),('min_prompt_points',4),
    ('max_prompt_points',9),('max_box_area_ratio',.9),('max_side_expansion',-1),('directory',None)])
def test_invalid_refinement_options_rejected(key,value):
    data=defaults();data['sapiensid']['pose']['refinement'][key]=value
    with pytest.raises(ValueError,match='sapiensid.pose.refinement'):
        validate(data)


def test_refinement_requires_pmpose():
    data=defaults();data['sapiensid']['pose']['refinement']['enabled']=True
    with pytest.raises(ValueError,match='backend: pmpose'):
        validate(data)


def test_production_single_and_batched_collection_use_identical_refined_geometry(monkeypatch):
    torch=pytest.importorskip('torch')
    from stash_ai.recognition import sapiensid_backend as backend, pmpose_pose
    data=defaults();data['models']['device']='cpu';data['sapiensid']['enabled']=True
    data['sapiensid']['collection']['enabled']=True
    data['sapiensid']['pose'].update(backend='pmpose',detector='dfine_x')
    data['sapiensid']['pose']['refinement']['enabled']=True
    class Detector:
        def predict(self,source,**kwargs):
            images=source if isinstance(source,list) else [source]
            return [SimpleNamespace(boxes=SimpleNamespace(xyxy=torch.tensor([BOX]),conf=torch.tensor([.98]))) for _ in images]
    monkeypatch.setattr(backend,'load_pose',lambda _:Detector())
    model=wrapper()
    monkeypatch.setattr(pmpose_pose,'load',lambda *_:model)
    detector=backend.SapiensIDFaces(SimpleNamespace(data=data),{},load_identity=False,load_detector=False)
    try:
        with Image.new('RGB',(160,100)) as image, Image.new('RGB',(160,100)) as other:
            detector.detect(image,faces=[])
            expected=detector.frame_people
            detector.detect_many([(image,[]),(other,[])])
            for frame in (image,other):
                detector.detect(frame,faces=[])
                assert detector.frame_people==expected
                assert detector.frame_people[0]['bounds']==[20,10,80,90]
                assert detector.frame_people[0]['keypoints'][10]==[75.,55.]
    finally:
        detector.close()
    assert detector.pose_refiner is None


def test_identity_mask_preserves_only_the_selected_pixels():
    import base64
    from stash_ai.recognition.person_geometry import encoder_crop
    image = Image.new('RGB', (4, 2), (251, 12, 19))
    mask = base64.b64encode(bytes([0b10000000])).decode()
    with encoder_crop(image, (1, 0, 3, 1), mask) as crop:
        assert crop.getpixel((0, 0)) == (251, 12, 19)
        assert crop.getpixel((1, 0)) == (127, 127, 127)
