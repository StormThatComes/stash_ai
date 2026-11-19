"""Displaced boxes, partial copied limbs, crop pixels and bounded recovery."""
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image

from stash_ai.recognition.pose_collision_recovery import conditioned, recover, NEUTRAL
from stash_ai.recognition.pose_deduplication import DeduplicatedPMPose
from stash_ai.recognition.pose_deduplication_config import defaults, contract


STANDING = np.array([[85,15],[82,13],[88,13],[79,15],[91,15],[75,40],[95,40],
                     [70,60],[100,60],[65,75],[105,75],[75,80],[95,80],
                     [75,105],[95,105],[75,130],[95,130]], dtype=float)
LYING = np.array([[15,100],[13,97],[13,103],[15,94],[15,106],[30,95],[30,110],
                  [45,90],[45,120],[60,85],[60,130],[75,95],[75,110],
                  [110,80],[110,120],[135,95],[135,130]], dtype=float)
BOXES = [[0,65,150,145], [60,0,110,140]]


def prediction(points):
    xy = np.asarray(points, dtype=float).reshape(-1,17,2)
    score = np.full(xy.shape[:2], .95)
    return xy, score, dict(presence=score.copy(), visibility=score.copy())


def copied():
    xy = LYING.copy()
    xy[[11,13,15]] = STANDING[[11,13,15]]
    return prediction([xy, STANDING])


class Pose:
    def __init__(self, candidate=LYING):
        self.candidate = candidate
        self.calls = []

    def predict_many(self, frames):
        self.calls.extend((np.asarray(image).copy(), boxes) for image, boxes in frames)
        return [prediction([self.candidate]) for _ in frames]


@pytest.mark.parametrize('retention', [0., .25, 1.])
def test_context_changes_only_outside_original_box_without_moving_geometry(retention):
    with Image.new('RGB',(160,160),(220,80,30)) as image:
        original = np.asarray(image).copy()
        with conditioned(image, BOXES[0], retention) as result:
            assert result.size == image.size
            pixels = np.asarray(result)
            np.testing.assert_array_equal(pixels[65:145,:150], original[65:145,:150])
            np.testing.assert_array_equal(pixels[5,5], np.rint(NEUTRAL+retention*(original[5,5]-NEUTRAL)))
        np.testing.assert_array_equal(np.asarray(image), original)


@pytest.mark.parametrize('mode', ['box_context_retry', 'partial_pose_recovery'])
def test_each_option_repairs_low_iou_partial_chain_and_keeps_other_person(mode):
    settings = dict(defaults(), **{mode:True})
    model = Pose()
    raw = copied()
    with Image.new('RGB',(160,160),(220,80,30)) as image:
        output, attempts, used = recover(model,image,BOXES,raw,settings)
    assert used == 1 and attempts[0]['accepted']
    assert attempts[0]['shared_indices'] == [11,13,15]
    np.testing.assert_array_equal(output[0][0],LYING)
    np.testing.assert_array_equal(output[0][1],STANDING)
    np.testing.assert_array_equal(raw[0],copied()[0])
    assert model.calls[0][1] == [BOXES[0]]
    if mode == 'partial_pose_recovery':
        np.testing.assert_array_equal(model.calls[0][0][100,15], [220,80,30])  # own head protected
        assert not np.array_equal(model.calls[0][0][105,75], [220,80,30])  # rival knee corridor


def test_single_touching_joint_and_distinct_people_do_not_trigger():
    for indices in ([], [9]):
        target=LYING.copy();target[indices]=STANDING[indices]
        model=Pose()
        with Image.new('RGB',(160,160)) as image:
            result, records, used=recover(model,image,BOXES,prediction([target,STANDING]),
                                         dict(defaults(),partial_pose_recovery=True,box_context_retry=True))
        assert not records and not model.calls and used==0


@pytest.mark.parametrize('failure', ['drift','still_copied','lost_points','bad_geometry'])
def test_rejects_unverified_repair_without_destroying_original(failure):
    candidate=LYING.copy()
    if failure=='drift': candidate[:7] += [30,0]
    if failure=='still_copied':candidate=copied()[0][0]
    if failure=='lost_points':candidate[10]=[-1,-1]
    if failure=='bad_geometry':candidate[0]=[np.nan,0]
    model=Pose(candidate)
    settings=dict(defaults(),partial_pose_recovery=True,max_contact_attempts=1)
    with Image.new('RGB',(160,160)) as image:
        if failure=='bad_geometry':
            with pytest.raises(ValueError,match='malformed'):
                recover(model,image,BOXES,copied(),settings)
            return
        output, records, used=recover(model,image,BOXES,copied(),settings)
    assert used==1 and not records[0]['accepted']
    np.testing.assert_array_equal(output[0],copied()[0])


def test_disabled_and_exhausted_retry_budgets_do_not_infer():
    for overrides in ({},dict(partial_pose_recovery=True,max_contact_attempts=0),
                      dict(partial_pose_recovery=True,max_retry_jobs_per_cue=0)):
        model=Pose()
        with Image.new('RGB',(160,160)) as image:
            output, records, used=recover(model,image,BOXES,copied(),dict(defaults(),**overrides))
        assert not model.calls and used==0
        np.testing.assert_array_equal(output[0],copied()[0])


def test_empty_and_isolated_cues_do_not_retry():
    for boxes, raw in (([],prediction([])),([BOXES[0]],prediction([LYING]))):
        model=Pose()
        with Image.new('RGB',(160,160)) as image:
            output, records, used=recover(model,image,boxes,raw,
                dict(defaults(),box_context_retry=True,partial_pose_recovery=True))
        assert output is raw and not records and used==0 and not model.calls


def test_accepted_complete_recovery_reaches_dedup_before_second_person_is_removed():
    boxes=[[0,0,150,150],BOXES[1]]
    class Model(Pose):
        def predict_many(self, frames):
            if len(frames[0][1])==2:
                return [prediction([STANDING,STANDING])]
            return super().predict_many(frames)
    model=Model()
    pipeline=DeduplicatedPMPose(model,dict(defaults(),box_context_retry=True))
    with Image.new('RGB',(160,160)) as image:
        output=pipeline.predict(image,boxes)
    assert output[4]['kept_indices']==[0,1]
    assert output[4]['collision_recovery'][0]['accepted']
    np.testing.assert_array_equal(output[0], [LYING,STANDING])


def test_reversed_detection_order_recovers_same_physical_person():
    raw=copied()
    raw=raw[0][::-1],raw[1][::-1],{k:v[::-1] for k,v in raw[2].items()}
    with Image.new('RGB',(160,160)) as image:
        output, records, _=recover(Pose(),image,BOXES[::-1],raw,dict(defaults(),partial_pose_recovery=True))
    assert any(r.get('accepted') and r['owner']==1 for r in records)
    np.testing.assert_array_equal(output[0], [STANDING,LYING])


def test_generic_confidence_adapter_and_frame_batch_match_single_results():
    from stash_ai.recognition.pose_masks import ConfidencePose
    class Generic:
        def predict_many(self, frames):
            return [(copied() if len(boxes)==2 else prediction([LYING]))[:2]
                    for _,boxes in frames]
    pipeline=DeduplicatedPMPose(ConfidencePose(Generic()),dict(defaults(),partial_pose_recovery=True))
    with Image.new('RGB',(160,160)) as a, Image.new('RGB',(160,160)) as b:
        single=pipeline.predict(a,BOXES)
        batch=pipeline.predict_many([(a,BOXES),(b,BOXES)])
    for result in batch:
        np.testing.assert_array_equal(result[0],single[0])
        assert result[4]==single[4]
        np.testing.assert_array_equal(result[0],[LYING,STANDING])


def test_retry_limit_is_shared_with_legacy_contact_recovery(monkeypatch):
    from stash_ai.recognition import contact_pose
    class Raw(Pose):
        supports_rival_mask=True
        def predict_many(self,frames):
            return [copied()] if len(frames[0][1])==2 else super().predict_many(frames)
    def legacy(pose,image,bounds,prediction,reserves,settings):
        assert settings['max_contact_attempts']==0
        return prediction,None
    monkeypatch.setattr(contact_pose,'recover_pair',legacy)
    refined=SimpleNamespace(settings={},refine_many=lambda frames,values:[
        (*v,[{} for _ in bounds]) for (_,bounds),v in zip(frames,values)])
    pipeline=DeduplicatedPMPose(Raw(),dict(defaults(),partial_pose_recovery=True,max_contact_attempts=1),refined)
    with Image.new('RGB',(160,160)) as image:
        result=pipeline.predict(image,BOXES)
    assert result[3][0]['collision_recovery'][0]['accepted']


def test_full_duplicate_with_no_exclusive_core_is_not_invented():
    boxes=[[55,0,110,145],BOXES[1]]
    with Image.new('RGB',(160,160)) as image:
        output, records, _=recover(Pose(STANDING),image,boxes,prediction([STANDING,STANDING]),
                                  dict(defaults(),box_context_retry=True))
    assert records and all(not r.get('accepted') for r in records)
    np.testing.assert_array_equal(output[0],[STANDING,STANDING])


def test_configuration_roundtrip_validation_and_contract(tmp_path):
    from stash_ai.recognition.config import load
    from stash_ai.recognition.config_layout import v2_document
    from stash_ai.recognition.config_render import render_v2
    path=tmp_path/'config.yml'
    path.write_text('version: 2\npose:\n  deduplication:\n    enabled: true\n'
                    '    box_context_retry: true\n    partial_pose_recovery: true\n')
    config=load(path)
    from stash_ai.recognition.detection_parameters import apply_edits
    edited,_=apply_edits(config,{'pose.deduplication.box_context_retention':0.},detection_preview=True)
    assert edited.data['sapiensid']['pose']['deduplication']['box_context_retention']==0.
    from stash_ai.recognition.inference_contracts import pose
    assert pose(edited)!=pose(config)
    data=config.data['sapiensid']['pose']['deduplication']
    assert data['box_context_retry'] and data['partial_pose_recovery']
    assert contract(data)!=contract(defaults())
    path.write_text(render_v2(v2_document(config.data,source=config.source)))
    assert load(path).data==config.data
    from stash_ai.recognition.pose_deduplication_config import validate
    for key,value in [('box_context_retention',1.1),('box_context_retry',1),
                      ('partial_point_distance',0),('partial_min_overlap',-1),
                      ('partial_min_shared_points',2),('partial_min_shared_points',True)]:
        with pytest.raises(ValueError):
            validate(dict(backend='pmpose',deduplication=dict(defaults(),**{key:value})))
    with pytest.raises(ValueError,match='requires'):
        validate(dict(backend='pmpose',deduplication=dict(defaults(),box_context_retry=True)))
    legacy={k:v for k,v in defaults().items() if not k.startswith(('box_context_', 'partial_'))}
    assert contract(legacy)==contract(defaults())
