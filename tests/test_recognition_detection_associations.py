"""Audit explanations follow actual reciprocal matching, including collisions."""
from types import SimpleNamespace

from stash_ai.recognition.detection_associations import explain_associations, face_label
from stash_ai.recognition.sapiensid_backend import associate


def test_competitor_with_an_alternative_still_blocks_reciprocal_match():
    faces = [SimpleNamespace(box=(20, 20, 20, 20), score=.99),
             SimpleNamespace(box=(30, 20, 20, 20), score=.95)]
    people = [dict(bounds=[0, 0, 100, 100], keypoints=[[35, 30]]),
              dict(bounds=[30, 0, 100, 100], keypoints=[[45, 30]])]
    matches = associate(faces, people)
    assert matches == [None, None]
    details = explain_associations(faces, people, matches)
    assert details == [dict(reason='competing_faces', candidates=[0], competing_faces=[1]),
                       dict(reason='multiple_person_candidates', candidates=[0, 1], competing_faces=[0])]
    assert face_label(0, faces[0], None, details[0]).endswith('competing faces: F1')
    assert face_label(1, faces[1], None, details[1]).endswith('multiple poses: P0, P1')


def test_exact_boundary_tolerance_and_matched_person_zero_are_preserved():
    face = SimpleNamespace(box=(20, 20, 20, 20), score=.99)
    person = dict(bounds=[21, 21, 100, 100], keypoints=[[30, 30]])
    assert associate([face], [person]) == [0]
    details = explain_associations([face], [person], [0])[0]
    assert details == dict(reason=None, candidates=[0], competing_faces=[])
    assert face_label(0, face, 0, details) == 'F0 0.990 -> P0'
    person['bounds'][0] = 21.01
    assert associate([face], [person]) == [None]
    assert explain_associations([face], [person], [None])[0]['reason'] == 'face_outside_person_box'


def test_crowded_labels_bound_id_lists_and_preserve_complete_json_details():
    face = SimpleNamespace(box=(20, 20, 20, 20), score=.99)
    person = dict(bounds=[0, 0, 100, 100], keypoints=[[30, 30]])
    details = explain_associations([face], [person] * 20, [None])[0]
    assert details['candidates'] == list(range(20))
    assert face_label(0, face, None, details).endswith('multiple poses: P0, P1, P2 (+17)')
    details = explain_associations([face] * 20, [person], [None] * 20)[0]
    assert details['competing_faces'] == list(range(1, 20))
    assert face_label(0, face, None, details).endswith('competing faces: F1, F2, F3 (+16)')
