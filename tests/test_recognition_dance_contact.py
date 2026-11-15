import base64
import json

import numpy as np
from PIL import Image

from benchmarks.benchmark_dance_contact import DATA, assign_people, correct_joints, read_profiles, score
from stash_ai.recognition.pose_mask_capture import capture_masks


def test_real_fixtures_match_annotations():
    photos = json.loads((DATA / 'labels.json').read_text())
    assert {f.stem for f in DATA.glob('*.jpg')} == photos.keys()
    assert sorted(len(p['people']) for p in photos.values()) == [1] * 2 + [2] * 13 + [3] * 2
    for name, photo in photos.items():
        with Image.open(DATA / f'{name}.jpg') as image:
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


def test_public_profiles(tmp_path):
    profiles = read_profiles(DATA / 'profiles.yml')
    assert profiles['stock'] == {}
    assert profiles['single-prompt']['pose.refinement.max_prompt_points'] == 1
    sweep = tmp_path / 'sweep.yml'
    sweep.write_text('pose: {stock: {}, nms: {pose.detector.nms_threshold: 0.9}}\nmask: {stock: {}}')
    assert read_profiles(sweep) == {'stock': {}, 'nms/stock': {'pose.detector.nms_threshold': 0.9}}


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


def test_one_profile_for_all_photos(monkeypatch, tmp_path):
    from benchmarks import benchmark_dance_contact as benchmark

    profiles = {name: {} for name in ('many', 'balanced', 'leaks', 'background', 'extra')}
    annotations = {'hard': {}, 'easy1': {}, 'easy2': {}}
    distributions = {'many': [0, 3, 3], 'balanced': [1, 2, 2],
                     'leaks': [3, 3, 3], 'background': [3, 3, 3], 'extra': [3, 3, 3]}
    seen = []

    def evaluate(config, edits, photos, directory):
        assert photos is annotations
        name = list(profiles)[len(seen)]
        seen.append(name)
        frames = {n: {'counts': {'people': 3, 'clean_masks': count}}
                  for n, count in zip(photos, distributions[name], strict=True)}
        totals = dict(clean_masks=sum(distributions[name]), matched=9, correct_joints=20, own_hits=27,
                      partner_hits=int(name == 'leaks'), background_hits=int(name == 'background'),
                      extra=int(name == 'extra'))
        return dict(frames=frames, totals=totals)

    monkeypatch.setattr(benchmark, 'evaluate', evaluate)
    report = benchmark.run(None, profiles, annotations, tmp_path)
    assert seen == list(profiles)
    assert report['winner'] == 'balanced'
    assert report['results']['balanced']['coverage'] == dict(photos=3, with_clean_mask=3, fully_clean=0)
    assert report['results']['many']['coverage'] == dict(photos=3, with_clean_mask=2, fully_clean=2)
    assert benchmark.select_profile({'leaks': report['results']['leaks']}, profiles) is None
