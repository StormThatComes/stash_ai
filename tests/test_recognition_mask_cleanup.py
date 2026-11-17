"""SAM cleanup in source pixels, including exact replay without the first pose."""
import json
from pathlib import Path

import numpy as np
from PIL import Image
import pytest

from stash_ai.recognition.config import load
from stash_ai.recognition.mask_candidates import check_mask
from stash_ai.recognition.pose_refinement import RefinedPMPose
from stash_ai.recognition.pose_refinement_config import contract, defaults
from tests.test_recognition_pose_refinement import BOX, Pose, Segmenter


def check(raw, settings, **kwargs):
    mask, _, reason = check_mask(raw, .98, [[10, 10]], [], [0, 0, raw.shape[1], raw.shape[0]],
                                 settings, raw.shape[::-1], **kwargs)
    assert reason is None
    return mask


@pytest.mark.parametrize('pixels,fraction,kept', [(6, 0., True), (7, 0., False),
                                                  (0, .01, True), (0, .02, False)])
def test_absolute_and_relative_size_floors(pixels, fraction, kept):
    raw = np.zeros((40, 55), bool)
    raw[5:25, 5:25] = True
    raw[10:12, 40:43] = True
    settings = dict(defaults(), min_island_pixels=pixels, min_island_fraction=fraction)
    result = check(raw, settings)
    assert result[10, 40] == kept and result[5:25, 5:25].all()


def test_distance_is_edge_to_edge_scaled_by_box_and_respects_own_pose():
    raw = np.zeros((60, 80), bool)
    raw[5:30, 5:25] = True
    raw[8:14, 34:39] = True  # 10 px between the nearest foreground pixels.
    raw[20:25, 45:51] = True
    raw[30:38, 68:75] = True
    settings = dict(defaults(), max_island_gap_fraction=.1)  # 100 px diagonal.
    edits = []
    result = check(raw, settings, own=[[70, 33]], edits=edits)
    assert result[8, 34] and result[33, 70] and not result[20, 45]
    assert not check(raw, dict(settings, max_island_gap_fraction=.09))[8, 34]
    assert check(raw, defaults())[20, 45]
    np.testing.assert_array_equal(check(raw, settings, replay=json.loads(json.dumps(edits))), result)


def bridged_mask(thickness=1):
    raw = np.zeros((40, 75), bool)
    raw[2:32, 2:32] = True
    raw[0:2, 7] = True  # A thin strand without a lobe must keep its original outline.
    raw[15:15+thickness, 32:48] = True
    raw[12:19, 48:54] = True
    return raw


def test_bridge_removes_whole_attachment_only_and_respects_area_cap_and_support():
    raw = bridged_mask()
    settings = dict(defaults(), bridge_radius_pixels=1, max_bridge_lobe_fraction=.1)
    expected = raw.copy()
    expected[:, 32:] = False
    cuts = []
    np.testing.assert_array_equal(check(raw, settings, cuts=cuts), expected)
    assert len(cuts) == 1
    np.testing.assert_array_equal(check(raw, settings, replay=[], replay_cuts=cuts), expected)
    for point in ([50, 15], [40, 15]):  # Wrist inside the lobe or the thin forearm.
        np.testing.assert_array_equal(check(raw, settings, own=[point]), raw)
    np.testing.assert_array_equal(check(raw, settings, replay=[], replay_cuts=[]), raw)
    for overrides in ({'max_bridge_lobe_fraction': .05}, {'bridge_radius_pixels': 0},
                      {'max_bridge_lobe_fraction': 0}):
        np.testing.assert_array_equal(check(raw, dict(settings, **overrides)), raw)
    thick = bridged_mask(3)
    np.testing.assert_array_equal(check(thick, settings), thick)
    np.testing.assert_array_equal(check(thick, dict(settings, bridge_radius_pixels=2)), expected)


def test_shared_neck_between_retained_cores_and_diagonal_attachments():
    settings = dict(defaults(), bridge_radius_pixels=1, max_bridge_lobe_fraction=.2)
    raw = bridged_mask()
    raw[2:32, 62:72] = True
    raw[15, 54:62] = True
    np.testing.assert_array_equal(check(raw, settings), raw)
    raw = np.zeros_like(raw)
    raw[:25, :25] = True
    raw[25:30, 25:30] = True  # Diagonal contact is 8-connected in the raw mask.
    cleaned = check(raw, settings)
    assert cleaned[:25, :25].all() and not cleaned[25:, :].any()


def test_refinement_records_only_actual_cuts_for_replay_and_quality():
    settings = dict(defaults(), bridge_radius_pixels=1, max_island_gap_fraction=.1)
    raw = np.zeros((100, 160), bool)
    raw[10:90, 20:60] = True
    raw[40, 60:69] = True
    raw[37:44, 69:75] = True
    raw[3:10, 120:127] = True
    segmenter = Segmenter()
    segmenter.predict = lambda *args: (raw, .98)
    pipeline = RefinedPMPose(Pose(), segmenter, settings)
    with Image.new('RGB', (160, 100)) as image:
        _, _, _, (record,) = pipeline.predict(image, [BOX])
    assert record['status'] == 'refined' and record['bounds'] == BOX
    assert record['mask_pixels'] == 3200 and record['overlap_mask']
    assert len(record['bridge_cuts']) == len(record['component_edits']) == 1
    mask, box, reason = check_mask(raw, .98, record['prompt_points'], (), BOX, settings, (160, 100),
                                   replay=record['component_edits'], replay_cuts=record['bridge_cuts'])
    assert reason is None and box == BOX and mask.sum() == record['mask_pixels']
    assert mask[10:90, 20:60].all()


def test_yaml_profiles_and_old_contract(tmp_path):
    path = tmp_path / 'old.yml'
    path.write_text('version: 2\npose: {refinement: {enabled: true}}\n')
    old = load(path).data['sapiensid']['pose']['refinement']
    keys = {'min_island_fraction', 'min_island_pixels', 'max_island_gap_fraction',
            'bridge_radius_pixels', 'max_bridge_lobe_fraction'}
    before = contract({k: v for k, v in old.items() if k not in keys})
    assert contract(old) == before
    for profile in ('facial_recognition.yml', 'facial_recognition_solver.yml'):
        settings = load(Path(profile)).data['sapiensid']['pose']['refinement']
        assert settings['min_island_pixels'] == 8 and settings['min_island_fraction'] == .002
        assert settings['bridge_radius_pixels'] == 1 and settings['max_island_gap_fraction'] == .12
        assert keys <= contract(settings).keys()
    assert contract(dict(old, max_bridge_lobe_fraction=.5)) == before


@pytest.mark.parametrize('key,value', [
    ('min_island_pixels', 1.5), ('min_island_fraction', 1.1),
    ('max_island_gap_fraction', -.1), ('bridge_radius_pixels', True),
    ('max_bridge_lobe_fraction', -1), ('min_limb_path_coverage', 1.1),
    ('min_limb_foreign_distance_pixels', -1), ('min_foreign_island_points', 0),
])
def test_invalid_cleanup_thresholds_name_the_yaml_field(tmp_path, key, value):
    path = tmp_path / 'bad.yml'
    path.write_text(f'version: 2\npose: {{refinement: {{enabled: true, {key}: {value}}}}}\n')
    with pytest.raises(ValueError, match=key):
        load(path)


def test_limb_rescue_and_rival_point_thresholds():
    raw = np.zeros((25, 40), bool)
    raw[3:13, 1:14] = True
    raw[8, 16] = True
    raw[4:15, 25:34] = True
    prompts = np.array([[6, 8]])
    limbs = [(prompts[0], np.array([16, 8]))]
    foreign = np.array([[19, 8], [27, 8]])
    def cleaned(**overrides):
        settings = dict(defaults(), max_foreign_coverage=1.)
        settings.update(overrides)
        mask, _, reason = check_mask(raw, .98, prompts, foreign, [0, 0, 40, 25],
                                     settings, (40, 25), own=prompts, limbs=limbs)
        assert reason is None
        return mask
    assert not cleaned()[8, 16]
    assert cleaned(min_limb_foreign_distance_pixels=2)[8, 16]
    assert not cleaned(min_limb_foreign_distance_pixels=2, min_limb_path_coverage=1.)[8, 16]
    assert cleaned()[8, 27] and not cleaned(min_foreign_island_points=1)[8, 27]
