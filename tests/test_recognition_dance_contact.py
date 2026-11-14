import base64
import json

import numpy as np
import pytest
from PIL import Image

from benchmarks.benchmark_dance_contact import DATA, assign_people, correct_joints, read_profiles, score
from stash_ai.recognition.pose_mask_capture import capture_masks


@pytest.mark.parametrize('directory,tune,holdout,counts', [
    (DATA, 3, 2, [2] * 5),
    (DATA / 'extended', 7, 5, [1] * 2 + [2] * 8 + [3] * 2),
])
def test_real_fixtures_match_annotations(directory, tune, holdout, counts):
    photos = json.loads((directory / 'labels.json').read_text())
    assert {f.stem for f in directory.glob('*.jpg')} == photos.keys()
    assert sum(p['split'] == 'tune' for p in photos.values()) == tune
    assert sum(p['split'] == 'holdout' for p in photos.values()) == holdout
    assert sorted(len(p['people']) for p in photos.values()) == counts
    for name, photo in photos.items():
        with Image.open(directory / f'{name}.jpg') as image:
            assert list(image.size) == photo['size']
            assert image.width == 500
            image.verify()
        assert all(0 <= x < photo['size'][0] and 0 <= y < photo['size'][1] for x, y in photo['background'])
        for person in photo['people']:
            left, top, right, bottom = person['box']
            assert 0 <= left < right <= photo['size'][0] and 0 <= top < bottom <= photo['size'][1]
            points = person['foreground'] + sum(person['joints'].values(), [])
            if person['head'] is not None:
                points.append(person['head'])
            assert all(left <= x < right and top <= y < bottom for x, y in points)


def test_public_profiles():
    profiles = read_profiles(DATA / 'profiles.yml')
    assert profiles['stock/stock'] == {}
    assert profiles['points/coverage']['pose.estimator.keypoint_score_threshold'] == 0.4


def test_shared_boxes_match_heads_and_each_joint_is_used_once():
    bounds = [5, 5, 25, 25]
    labels = [dict(head=head, box=bounds) for head in ([20, 10], [9, 9])]
    keypoints = [[-1, -1]] * 17
    people = [dict(bounds=bounds, keypoints=[h] + keypoints[1:]) for h in ([9, 9], [20, 10])]
    assert assign_people(labels, people) == {0: 1, 1: 0}
    assert assign_people(labels, []) == {}
    keypoints[15] = [10, 20]
    assert correct_joints(dict(box=bounds, joints={'ankle': [[10, 20], [11, 20]]}), keypoints) == 1


def test_admitted_and_rejected_masks_use_separate_scores(tmp_path):
    from benchmarks.benchmark_dance_contact import save_overlay

    bounds = [10, 10, 20, 20]
    points = [[14, 12]] + [[-1, -1]] * 16
    people = [dict(bounds=bounds, keypoints=points)]
    own = [[13, 12], [14, 12], [15, 12], [16, 12]]
    labels = [dict(box=bounds, head=points[0], joints={'wrist': [[12, 15]]}, foreground=own),
              dict(box=bounds, head=[19, 19], joints={}, foreground=[[18, 17]])]
    photo = dict(size=[25, 25], people=labels, background=[[10, 10], [24, 24]])
    crop = np.zeros((10, 10), dtype=bool)
    crop[2, 3:7] = True
    details = dict(status='refined', bounds=bounds, overlap_mask=base64.b64encode(np.packbits(crop)).decode())
    people[0]['pose_refinement'] = details
    counts, reasons, _ = score(photo, people)
    assert (counts['matched'], counts['masks'], counts['clean_masks']) == (1, 1, 1)
    assert (counts['own_hits'], counts['own_probes'], counts['joints']) == (4, 5, 1)
    assert reasons == {'person_missing': 1}

    crop[7, 8] = True
    details['overlap_mask'] = base64.b64encode(np.packbits(crop)).decode()
    counts, _, _ = score(photo, people)
    assert counts['partner_hits'] == 1 and counts['clean_masks'] == 0
    details.update(reason='missing_prompt_support', status='rejected')
    del details['overlap_mask']
    with Image.new('RGB', tuple(photo['size'])) as image, capture_masks(image) as capture:
        raw = np.zeros((image.height, image.width), dtype=bool)
        raw[10:20, 10:20] = crop
        capture.save(details, 'cleaned', raw)
        counts, missing, candidates = score(photo, people, capture)
        assert counts['own_hits'] == counts['masks'] == 0
        assert missing == {'person_missing': 1, 'missing_prompt_support': 1}
        assert candidates[0]['own_hits'] == 4 and candidates[0]['partner_hits'] == 1
        save_overlay(image, people, capture, tmp_path / 'overlay.png')
    with Image.open(tmp_path / 'overlay.png') as overlay:
        assert overlay.height > photo['size'][1]


def test_fit_all_excludes_leakage_and_extra_detections(monkeypatch):
    from benchmarks import benchmark_dance_contact as benchmark

    clean = dict(clean_masks=6, matched=10, correct_joints=37, own_hits=36,
                 partner_hits=0, background_hits=0, extra=0)
    measured = {'clean': {'totals': clean}, 'more-edits': {'totals': clean}}
    for issue in ('partner_hits', 'background_hits', 'extra'):
        measured[issue] = {'totals': {**clean, 'clean_masks': 7, issue: 1}}
    photos = {'tuning-photo': {'split': 'tune'}, 'reserved-photo': {'split': 'holdout'}}

    def run(config, split, profiles, selected, overlays):
        assert split == 'all' and selected == photos
        return {name: measured[name] for name in profiles}

    monkeypatch.setattr(benchmark, 'run_split', run)
    profiles = {name: {} for name in measured}
    profiles['more-edits'] = {'pose.detector.nms_threshold': 0.9}
    result = benchmark.fit_all(None, profiles, photos, None)
    assert result['winner'] == 'clean' and result['all_photos'] == measured
    assert benchmark.fit_all(None, {'extra': {}}, photos, None)['winner'] is None


def test_transfer_selection_uses_new_tuning_masks_and_checks_original_ties():
    from benchmarks.benchmark_dance_transfer import select_profile

    stock = dict(clean_masks=7, matched=12, correct_joints=42, own_hits=54,
                 partner_hits=1, background_hits=0, extra=0)
    profiles = {'stock': {}, 'simple': {'a': 1}, 'better-original': {'a': 1, 'b': 2}, 'leaks': {'a': 2}}
    tune = {name: {'totals': {**stock, 'clean_masks': 8}} for name in profiles}
    tune['stock'] = {'totals': stock}
    tune['leaks'] = {'totals': {**stock, 'clean_masks': 10, 'partner_hits': 2}}
    original = {name: {'totals': {**stock, 'partner_hits': 0}} for name in profiles}
    original['better-original']['totals']['clean_masks'] += 1
    assert select_profile(profiles, tune, original) == 'better-original'
