"""Intermediate snapshots must explain real decisions without changing inference."""
import json
from types import SimpleNamespace

import numpy as np
from PIL import Image
import pytest

from stash_ai.recognition import detection_phases as phases
from stash_ai.recognition.pose_refinement import RefinedPMPose
from stash_ai.recognition.pose_refinement_config import defaults
from tests.test_recognition_pose_refinement import BOX, Pose, Segmenter, pose_points, prediction
from tests.test_recognition_profile3_batching import BatchedPose, BatchedSegmenter


@pytest.mark.parametrize('batched', [False, True])
def test_snapshots_are_detached_and_capture_both_masked_pose_and_accepted_points(batched):
    pose = BatchedPose() if batched else Pose(drift=True)
    model = RefinedPMPose(pose, BatchedSegmenter() if batched else Segmenter(), defaults())
    with Image.new('RGB', (160, 100)) as image:
        with phases.capture(image) as history:
            output = model.predict(image, [BOX])
        assert phases.active(image) is None and history.image is None
        titles = [p.title for p in history.phases]
        initial = next(p for p in history.phases if p.title.startswith('Initial keypoints'))
        raw = next(p for p in history.phases if p.title.startswith('Masked pose pass'))
        final = next(p for p in history.phases if p.title.startswith('After refinement acceptance'))
        assert any('raw candidate' in t for t in titles)
        assert any('after cleanup' in t for t in titles)
        if not batched:
            assert raw.people[0].points != initial.people[0].points
            assert final.people[0].points == initial.people[0].points
            assert 'second_pass_core_drift' in final.people[0].note
        else:
            assert raw.people[0].points == final.people[0].points
        before = initial.people[0].points
        output[0][:] = 999
        assert initial.people[0].points == before
        calls = pose.calls.copy()
        for index in range(len(history.phases)):
            with history.render(image, index) as rendered:
                assert rendered.width >= image.width and rendered.height > image.height
        assert pose.calls == calls
    assert len([c for c in pose.calls if c[0].startswith('masked')]) == 1


def test_retries_keep_earlier_masks_and_their_actual_prompts():
    model = RefinedPMPose(Pose(), Segmenter(), dict(defaults(), min_prompt_points=1, max_prompt_points=2))
    def segment(positive, negative=()):
        pixels = np.zeros((100, 160), bool)
        pixels[10:90, 20:80 if len(positive) == 2 else 60] = True
        return pixels, .95 if len(positive) == 2 else .1
    model.segmenter.predict = segment
    with Image.new('RGB', (160, 100)) as image, phases.capture(image) as history:
        model.predict(image, [BOX])
    raw = [p.people[0] for p in history.phases if 'raw candidate' in p.title]
    assert len(raw) == 2 and [len(p.positive) for p in raw] == [1, 2]
    assert raw[0].mask != raw[1].mask and raw[0].rejected and not raw[1].rejected
    assert 'low_mask_score' in raw[0].note


def test_rejected_after_cleanup_still_has_a_cleaned_snapshot():
    model = RefinedPMPose(Pose(), Segmenter(), dict(defaults(), min_prompt_points=1, max_prompt_points=1))
    def segment(*args):
        pixels = np.zeros((100, 160), bool)
        pixels[70:90, 30:40] = True  # Valid pixels, but no head prompt support.
        return pixels, .99
    model.segmenter.predict = segment
    with Image.new('RGB', (160, 100)) as image, phases.capture(image) as history:
        output = model.predict(image, [BOX])
    cleaned = next(p for p in history.phases if 'after cleanup' in p.title)
    assert output[3][0]['reason'] == 'missing_prompt_support'
    assert cleaned.people[0].mask and cleaned.people[0].rejected
    assert 'missing_prompt_support' in cleaned.people[0].note


def test_capture_limits_and_exception_cleanup_do_not_change_detection(monkeypatch):
    monkeypatch.setattr(phases, 'MAX_BYTES', 5000)
    model = RefinedPMPose(Pose(), Segmenter(), defaults())
    with Image.new('RGB', (160, 100)) as image:
        with phases.capture(image) as history:
            output = model.predict(image, [BOX])
        assert history.omitted > 0 and history.bytes <= 5000
        assert output[3][0]['status'] == 'refined'
        model.segmenter.failure = True
        with pytest.raises(RuntimeError, match='inference failed'):
            with phases.capture(image) as failed:
                model.predict(image, [BOX])
        assert phases.active(image) is None and failed.image is None
        with phases.capture(image, enabled=False) as disabled:
            assert disabled is None and phases.active(image) is None


def test_deduplication_keeps_original_candidate_labels_and_pre_filter_points():
    from stash_ai.recognition.pose_deduplication import DeduplicatedPMPose
    from stash_ai.recognition.pose_deduplication_config import defaults as dedup_defaults
    raw = prediction([pose_points(), pose_points()])
    model = DeduplicatedPMPose(SimpleNamespace(predict_many=lambda frames: [raw]), dedup_defaults())
    with Image.new('RGB', (100, 100)) as image, phases.capture(image) as history:
        output = model.predict(image, [BOX, BOX])
        person = dict(bounds=BOX, keypoints=output[0][0].tolist())
        person['keypoints'][0] = [-1, -1]
        phases.output_people(image, [person])
    before, after, filtered = history.phases[1:]
    assert [p.label for p in before.people] == ['D0', 'D1']
    assert len(after.people) == 1 and after.people[0].label == 'P0 (D0)'
    assert before.people[0].points[0] == (35, 20)
    assert filtered.people[0].points[0] == (-1, -1)


@pytest.mark.parametrize('second_pass', [False, True])
def test_generic_estimators_only_show_stages_they_execute(second_pass):
    from stash_ai.recognition.pose_masks import PoseMasks
    model = PoseMasks(BatchedPose(), BatchedSegmenter(), dict(defaults(), generic_second_pass=second_pass))
    with Image.new('RGB', (160, 100)) as image, phases.capture(image) as history:
        model.predict(image, [BOX])
    titles = [p.title for p in history.phases]
    assert any(t.startswith('Masked pose pass') for t in titles) == second_pass
    assert any('no second pose pass' in t for t in titles) != second_pass


def test_final_pose_cache_is_bypassed_only_while_capturing(monkeypatch):
    from stash_ai.recognition.cached_pose import Pose as CachedPose
    from stash_ai.recognition.config import defaults as config_defaults
    import stash_ai.recognition.cached_pose as cached
    calls = []
    model = CachedPose(SimpleNamespace(predict_many=lambda frames: calls.append('model') or []),
                       SimpleNamespace(data=config_defaults()), 'cpu')
    monkeypatch.setattr(cached, 'many', lambda *a, **k: calls.append('cache') or [])
    with Image.new('RGB', (160, 100)) as image:
        model.predict_many([(image, [BOX])])
        with phases.capture(image):
            model.predict_many([(image, [BOX])])
        model.predict_many([(image, [BOX])])
    assert calls == ['cache', 'model', 'cache']


def test_desktop_result_keeps_json_metadata_serializable():
    with Image.new('RGB', (20, 20)) as image, phases.capture(image) as history:
        result = phases.DetectionResult(image, {'faces': [], 'people': []}, history)
        rendered, record = result
        assert rendered is image and result[1] is record
        assert json.loads(json.dumps(record)) == record and result.phases is history
