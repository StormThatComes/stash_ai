"""Independent mask consumers, legacy migration and scalar/batched parity."""
from copy import deepcopy
from types import SimpleNamespace

import numpy as np
from PIL import Image
import pytest

from stash_ai.recognition.config import defaults, effective_data
from stash_ai.recognition.config_layout import v2_document
from stash_ai.recognition.mask_usage import applications
from stash_ai.recognition.pose_refinement import RefinedPMPose
from stash_ai.recognition.pose_masks import PoseMasks
from stash_ai.recognition.pose_refinement_config import defaults as mask_defaults
from tests.test_recognition_pose_refinement import BOX, Pose, Segmenter, pose_points, prediction


@pytest.mark.parametrize('backend', ['pmpose', 'vitpose_plus_large', 'sapiens', 'rtmo'])
@pytest.mark.parametrize('second', [False, True])
def test_legacy_settings_export_as_independent_booleans(backend, second):
    if backend == 'rtmo' and second:
        return
    doc = dict(version=2, pose=dict(estimator=dict(backend=backend),
        refinement=dict(enabled=True, generic_second_pass=second, masked_identity_crop=True, expand_boxes=False)))
    if backend == 'rtmo':
        doc['pose']['detector'] = dict(backend='rtmo')
    _, old = effective_data(doc)
    exported = v2_document(old)
    assert 'refinement' not in exported['pose']
    use = exported['pose']['mask']['apply_to']
    assert all(type(v) is bool for v in use.values())
    assert use['pose_refinement'] is (backend == 'pmpose' or second)
    assert use['body_encoder_input'] and not use['box_expansion']
    _, restored = effective_data(exported)
    assert old == restored


@pytest.mark.parametrize('value', [None, 1, 'true'])
def test_consumer_requires_a_boolean_even_when_generation_disabled(value):
    with pytest.raises(ValueError, match='true or false'):
        effective_data(dict(version=2, pose=dict(mask=dict(enabled=False, apply_to=dict(sharpness=value)))))


def test_conflicting_legacy_and_new_switches_are_rejected():
    for legacy, modern in [('expand_boxes', 'box_expansion'), ('masked_identity_crop', 'body_encoder_input'),
                           ('generic_second_pass', 'pose_refinement')]:
        with pytest.raises(ValueError, match='combine'):
            effective_data(dict(version=2, pose=dict(refinement={legacy: True},
                mask=dict(apply_to={modern: False}))))


def test_crop_sharpness_does_not_require_mask_generation():
    _, data = effective_data(dict(version=2, pose=dict(mask=dict(enabled=False, apply_to=dict(sharpness=False)))))
    assert data['sapiensid']['collection']['min_sharpness'] > 0
    with pytest.raises(ValueError, match='requires SAM masks'):
        effective_data(dict(version=2, pose=dict(mask=dict(enabled=False, apply_to=dict(sharpness=True)))))


class BatchedSegmenter(Segmenter):
    def predict_frames(self, requests):
        output = []
        for image, prompts in requests:
            with self.frame(image):
                output.append([self.predict(p, n) for p, n in prompts])
        return output


@pytest.mark.parametrize('adapter', [RefinedPMPose, PoseMasks])
@pytest.mark.parametrize('batched', [False, True])
@pytest.mark.parametrize('missing', [False, True])
def test_sharpness_only_preserves_all_pose_values_and_boxes(adapter, batched, missing):
    pose = Pose()
    segmenter = (BatchedSegmenter if batched else Segmenter)()
    if missing:
        segmenter.predict = lambda *a: (np.zeros((100,160), bool), .99)
    settings = dict(mask_defaults(), pose_second_pass=False, mask_keypoint_filtering=False,
                    expand_boxes=False, masked_identity_crop=False, mask_overlap=False, mask_body_admission=False,
                    joint_limb_ownership=True)
    model = adapter(pose, segmenter, settings)
    with Image.new('RGB', (160,100)) as image:
        results = model.predict_many([(image, [BOX, [70,10,110,90]])] * 2)
    assert pose.calls == [('first', 2)]
    expected = prediction([pose_points(), pose_points() + [50,0]])
    for result in results:
        np.testing.assert_array_equal(result[0], expected[0])
        np.testing.assert_array_equal(result[1], expected[1])
        for key in expected[2]:
            np.testing.assert_array_equal(result[2][key], expected[2][key])
        assert [r['bounds'] for r in result[3]] == [BOX, [70,10,110,90]]
        assert all('encoder_mask' not in r for r in result[3])
        assert all(not r['mask_applied']['keypoint_filtering'] for r in result[3])


@pytest.mark.parametrize('second', [False, True])
@pytest.mark.parametrize('filter_points', [False, True])
def test_pose_pass_and_point_filtering_are_independent(second, filter_points):
    pose = Pose()
    settings = dict(mask_defaults(), pose_second_pass=second, mask_keypoint_filtering=filter_points, expand_boxes=False)
    segmenter = Segmenter()
    # This valid torso mask deliberately excludes the left wrist.
    def mask(*args):
        pixels = np.zeros((100,160), bool); pixels[10:90,26:60] = True
        return pixels, .99
    segmenter.predict = mask
    model = RefinedPMPose(pose, segmenter, settings)
    with Image.new('RGB',(160,100)) as image:
        result = model.predict(image, [BOX])
    assert len(pose.calls) == (2 if second else 1)
    assert bool(result[1][0,9] == 0) == filter_points
    if not filter_points:
        assert result[0][0,9].tolist() == pose_points()[9].tolist()
    assert result[3][0]['mask_valid']


def test_sharpness_uses_source_pixels_independently_of_encoder_mask():
    from stash_ai.recognition.body_collection import quality
    from stash_ai.recognition.foreground_quality import laplacian
    from tests.test_recognition_body_mask_overlap import person
    mask = np.zeros((32,32), bool); mask[8:24,8:24] = True
    pixels = np.full((32,32), 220, np.uint8); pixels[8:24,8:24] = 80
    item = person([0,0,32,32], mask)
    item['pose_refinement']['encoder_mask'] = item['pose_refinement']['overlap_mask']
    item['mask_usage'] = dict(sharpness=False, body_encoder_input=True)
    policy = dict(min_person_pixels=1, min_sharpness=1, min_keypoints=1, max_overlap=1)
    with Image.fromarray(pixels).convert('RGB') as image:
        crop, reason = quality(image,item,[item],policy)
        assert reason is None and crop['sharpness_method'] == 'person_crop'
        assert crop['sharpness'] == laplacian(pixels)['sharpness']
        item['mask_usage']['sharpness'] = True
        masked, reason = quality(image,item,[item],policy)
        assert masked['sharpness'] == 0 and reason == 'person_sharpness'
        item['pose_refinement']['mask_conflict'] = True
        missing, reason = quality(image,item,[item],policy)
        assert missing['sharpness_method'] == 'unavailable'
        assert reason == 'person_sharpness_mask_unavailable'


def test_overlap_switch_ignores_both_own_and_peer_masks():
    from tests.test_recognition_body_mask_overlap import split_people
    from stash_ai.recognition.body_overlap import measure
    people = split_people()
    assert measure(people[0], people)['overlap'] == 0
    people[0]['mask_usage'] = dict(overlap=False)
    assert measure(people[0], people) == dict(overlap=1.,box_overlap=1.,overlap_basis='box')


def test_body_admission_switch_does_not_bypass_mask_validity():
    from stash_ai.recognition.body_collection import collect
    from stash_ai.recognition.body_overlap import foreground_mask
    from tests.test_recognition_body_mask_overlap import person
    item = person([0,0,32,32], np.ones((32,32),bool), mask_conflict=True)
    data = defaults()
    data['sapiensid']['collection'].update(min_person_pixels=1,min_keypoints=1,min_sharpness=0)
    for active, rejection in [(True,'person_mask_conflict'),(False,None)]:
        item['mask_usage'] = dict(body_admission=active, sharpness=False)
        assert foreground_mask(item) is None
        items = []
        model = SimpleNamespace(frame_people=[item],face_matches=[],person_geometry=lambda *a: ('input',))
        pipe = SimpleNamespace(collect_items=lambda db,image,rows: items.extend(rows))
        with Image.new('RGB',(32,32)) as image:
            collect(pipe,None,model,SimpleNamespace(data=data),'00000000-0000-0000-0000-000000000001',0,0,image,[],[])
        record = next(row for row,_ in items if row[0] is not None)
        assert record[1]['identity_rejection'] == rejection


def test_each_application_changes_evidence_contract_and_export_roundtrips():
    from stash_ai.recognition.person_signal import contract
    _, original = effective_data(dict(version=2))
    before = contract(SimpleNamespace(data=original))
    document = v2_document(original)
    for key in document['pose']['mask']['apply_to']:
        changed = deepcopy(document)
        use = changed['pose']['mask']['apply_to']; use[key] = not use[key]
        _, data = effective_data(changed)
        assert contract(SimpleNamespace(data=data)) != before, key
        assert effective_data(v2_document(data))[1] == data


def test_disabled_mask_crop_sharpness_has_a_distinct_contract():
    from stash_ai.recognition.person_signal import contract
    data = defaults(); before = contract(SimpleNamespace(data=data))
    data['sapiensid']['pose']['refinement']['mask_sharpness'] = False
    assert contract(SimpleNamespace(data=data)) != before


@pytest.mark.parametrize('encoder', [False, True])
def test_encoder_mask_does_not_require_pose_refinement_or_point_filters(encoder):
    settings = dict(mask_defaults(), pose_second_pass=False, mask_keypoint_filtering=False,
                    expand_boxes=False, masked_identity_crop=encoder)
    segmenter = Segmenter()
    def segment(points, negative=()):
        mask = np.zeros((100,160), bool)
        left,right = (20,52) if points[0,0] < 50 else (53,85)
        mask[10:90,left:right] = True
        return mask, .99
    segmenter.predict = segment
    model = RefinedPMPose(Pose(),segmenter,settings)
    with Image.new('RGB',(160,100), 'white') as image:
        result = model.predict(image,[BOX,[45,10,85,90]])
        for record in result[3]:
            assert record['separated'] and record['mask_valid']
            assert bool(record.get('encoder_mask')) == encoder
            from stash_ai.recognition.person_geometry import encoder_crop
            with encoder_crop(image,record['bounds'],record.get('encoder_mask')) as crop:
                values = np.asarray(crop)
                assert bool(np.any(values == 127)) == encoder
    assert model.pose.calls == [('first', 1)]
    np.testing.assert_array_equal(result[0], prediction([pose_points(),pose_points()+[25,0]])[0])


@pytest.mark.parametrize('expand', [False, True])
def test_box_expansion_is_independent_of_both_pose_consumers(expand):
    model = RefinedPMPose(Pose(),Segmenter(),dict(mask_defaults(),pose_second_pass=False,
        mask_keypoint_filtering=False,expand_boxes=expand))
    with Image.new('RGB',(160,100)) as image:
        result = model.predict(image,[BOX])
    assert result[3][0]['bounds'] == ([20,10,80,90] if expand else BOX)
    np.testing.assert_array_equal(result[0][0],pose_points())
    assert model.pose.calls == [('first',1)]


def test_sharpness_only_deduplication_does_not_enter_contact_pose_recovery(monkeypatch):
    from stash_ai.recognition.pose_deduplication import DeduplicatedPMPose
    from stash_ai.recognition.pose_deduplication_config import defaults as dedup_defaults
    from stash_ai.recognition import contact_pose
    monkeypatch.setattr(contact_pose,'recover_pair',lambda *a: pytest.fail('mask-only entered pose recovery'))
    pose = Pose(); pose.supports_rival_mask = True
    settings = dict(mask_defaults(),pose_second_pass=False,mask_keypoint_filtering=False,expand_boxes=False)
    refined = RefinedPMPose(pose,Segmenter(),settings)
    dedup = dict(dedup_defaults(),box_context_retry=False,partial_pose_recovery=False)
    masked = DeduplicatedPMPose(pose,dedup,refined)
    baseline = DeduplicatedPMPose(Pose(),dedup)
    with Image.new('RGB',(160,100)) as image:
        a,b = masked.predict(image,[BOX,[45,10,85,90]]),baseline.predict(image,[BOX,[45,10,85,90]])
    for left,right in zip(a[:2],b[:2]):
        np.testing.assert_array_equal(left,right)
    for key in b[2]:
        np.testing.assert_array_equal(a[2][key],b[2][key])
    assert a[-1] == b[-1]
