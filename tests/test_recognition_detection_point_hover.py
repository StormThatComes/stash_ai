"""Hover evidence belongs to its phase and uses real measurements, including rejected joints."""
from types import SimpleNamespace

import numpy as np
from PIL import Image
import pytest

from stash_ai.recognition import detection_phases as phases
from stash_ai.recognition.detection_point_details import final_people
from stash_ai.recognition.detection_point_hover import targets, face_targets, nearest, Target
from stash_ai.recognition.sapiensid_backend import SapiensIDFaces
from stash_ai.recognition.config import defaults
from tests.test_recognition_pose_refinement import BOX, pose_points, prediction


def filtered(image, raw, history):
    data = defaults()
    settings = data['sapiensid']['pose']
    settings.update(backend='pmpose', keypoint_score_threshold=.5,
                    pmpose_presence_threshold=.5, pmpose_visibility_threshold=.5)
    model = SimpleNamespace(config=SimpleNamespace(data=data))
    people = SapiensIDFaces._people(model, [BOX], [.99], *raw)
    phases.output_people(image, people, prediction=raw, settings=settings)
    return dict(people=people, faces=[])


def test_each_phase_keeps_its_measurements_when_later_arrays_change():
    raw = prediction([pose_points()])
    raw[2]['presence'][:] = .81
    raw[2]['visibility'][:] = .62
    with Image.new('RGB', (160, 100)) as image, phases.capture(image, backend='pmpose') as history:
        phases.pose(image, 'First pass', [BOX], raw)
        before = history.phases[-1]
        raw[1][:] = .73
        raw[2]['visibility'][:] = .41
        phases.pose(image, 'Second pass', [BOX], raw)
        after = history.phases[-1]
        raw[1][:] = 0
        raw[2]['visibility'][:] = 0
    first = targets(before.people, before.title, (160, 100))[0].text
    second = targets(after.people, after.title, (160, 100))[0].text
    assert 'Score (expected OKS): 0.9500' in first and 'Visibility: 0.6200' in first
    assert 'Score (expected OKS): 0.7300' in second and 'Visibility: 0.4100' in second
    assert 'Presence: 0.8100' in first and 'Presence: 0.8100' in second


def test_filtered_points_remain_hoverable_with_exact_gate_and_original_values():
    raw = prediction([pose_points()])
    raw[1][0, 0] = .5  # Equality fails the strict output cutoff.
    raw[2]['presence'][0, 1] = .4
    raw[2]['visibility'][0, 2] = .3
    raw[0][0, 3, 0] = BOX[2]  # Right edge is exclusive.
    with Image.new('RGB', (160, 100)) as image, phases.capture(image, backend='pmpose') as history:
        phases.pose(image, 'Before output', [BOX], raw)
        record = filtered(image, raw, history)
        output = history.phases[-1]
        hits = targets(output.people, output.title, image.size)
        final = targets(final_people(record, history), 'Final result', image.size)
        with history.render(image, len(history.phases)-1) as rendered:
            assert rendered.getpixel((35, 16)) == (210, 210, 210)  # Hollow filtered nose.
    assert len(hits) == len(final) == 17
    assert all(hit.rejected for hit in hits[:4])
    assert 'keypoint_score_threshold: 0.5000 <= 0.5' in hits[0].text
    assert 'presence_threshold: 0.4000 <= 0.5' in hits[1].text
    assert 'visibility_threshold: 0.3000 <= 0.5' in hits[2].text
    assert 'Outside output box' in hits[3].text
    assert 'Score (expected OKS): 0.9500' in hits[4].text  # Never the previous synthetic 1.0.
    assert 'keypoint_score_threshold: 0.5000 <= 0.5' in final[0].text
    assert record['people'][0]['keypoints'][0] == [-1, -1]  # Overlay never restores model output.


def test_generic_probabilities_are_marked_as_proxies_not_measured_visibility():
    raw = prediction([pose_points()])
    with Image.new('RGB', (160, 100)) as image, phases.capture(image, backend='vitpose_plus_large') as history:
        phases.pose(image, 'Mask prompts', [BOX], raw)
    text = targets(history.people, 'Mask prompts', (160, 100))[0].text
    assert 'Score (heatmap score): 0.9500' in text
    assert 'Visibility: not measured (confidence proxy: 0.9500)' in text


def test_missing_values_and_changed_final_geometry_never_reuse_stale_evidence():
    raw = prediction([pose_points()])
    raw[1][0, 0] = .1
    with Image.new('RGB', (160, 100)) as image, phases.capture(image, backend='pmpose') as history:
        record = filtered(image, raw, history)
        record['people'][0]['bounds'] = [10, 10, 80, 90]  # A later accepted crowded-box retry.
        assert not any(t.rejected for t in targets(final_people(record, history), 'Final', image.size))
    person = dict(bounds=BOX, keypoints=pose_points().tolist())
    text = targets(final_people(dict(people=[person])), 'Final', (160, 100))[0].text
    assert 'Score (unavailable): N/A' in text and 'Visibility: not measured' in text


@pytest.mark.parametrize('zoom', [.5, 1., 4.])
def test_hit_radius_is_in_screen_pixels_and_collocated_people_are_not_hidden(zoom):
    points = (Target(20, 30, 'P0 nose'), Target(20, 30, 'P1 nose'), Target(80, 30, 'P2 nose'))
    assert nearest(points, 20+7/zoom, 30, zoom) == points[:2]
    assert not nearest(points, 20+9/zoom, 30, zoom)
    assert nearest(points, 80, 30, zoom) == points[2:]


def test_face_callout_hits_use_display_offsets_without_inventing_landmark_scores():
    record = dict(faces=[dict(face_index=0, score=.9, landmarks=[[20, 30]], display_landmarks=[[220, 30]])])
    [point] = face_targets(record)
    assert point.x == 220 and 'Position: (20.00, 30.00)' in point.text
    assert 'Point score: N/A' in point.text and 'Face detection score: 0.9000' in point.text
