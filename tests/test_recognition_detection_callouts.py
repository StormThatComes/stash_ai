"""Displaced face geometry must not obscure pose or fabricate measurements."""
from copy import deepcopy
from types import SimpleNamespace

from PIL import Image
import pytest

from stash_ai.recognition.detection_callouts import draw_leaders, layout_faces


def face(box, points=None):
    x, y, w, h = box
    return SimpleNamespace(box=box, landmarks=points if points is not None else
                           [[x + w / 2, y + h / 2]], score=.99)


@pytest.mark.parametrize('box', [[0, 0, 30, 30], [269, 0, 30, 30],
                                 [0, 269, 30, 30], [269, 269, 30, 30],
                                 [120.5, 120.5, 30, 30]])
def test_edge_faces_translate_without_clipping_or_scaling(box):
    detected = face(box)
    before = deepcopy(detected)
    with Image.new('RGB', (300, 300)) as image:
        layouts, size = layout_faces(image, [detected], [], ['F0'])
    display = layouts[0]
    dx, dy = display['offset']
    assert (dx, dy) != (0, 0)
    assert size == (300, 300)
    x, y, w, h = box
    assert display['bounds'] == [x + dx, y + dy, x + w + dx, y + h + dy]
    assert display['landmarks'] == [[px + dx, py + dy] for px, py in detected.landmarks]
    assert all(0 <= px < size[0] and 0 <= py < size[1] for px, py in display['landmarks'])
    assert detected == before


def test_candidate_avoids_other_faces_and_pose_points():
    faces = [face([100, 100, 30, 30]), face([140, 100, 30, 30])]
    # Face 1 blocks the right; a measured pose point blocks the left.
    people = [dict(keypoints=[[65, 115]])]
    with Image.new('RGB', (300, 300)) as image:
        layouts, size = layout_faces(image, faces, people, ['F0', 'F1'])
    assert layouts[0]['offset'][1] != 0
    assert size == (300, 300)
    a, b = [item['bounds'] for item in layouts]
    assert a[2] < b[0] or b[2] < a[0] or a[3] < b[1] or b[3] < a[1]


def test_overflow_preserves_scale_and_invalid_source_landmarks():
    detected = face([-5, -5, 150, 150],
                    [[20, 20], [-1, -1], [100, 20], [20, float('nan')], [float('inf'), 20]])
    with Image.new('RGB', (100, 100)) as image:
        layouts, size = layout_faces(image, [detected, detected], [], ['F0', 'F1'])
    assert size[0] > 100
    for display in layouts:
        dx, dy = display['offset']
        assert display['bounds'] == [dx, dy, 99 + dx, 99 + dy]
        assert display['landmarks'] == [[20 + dx, 20 + dy]] + [[-1, -1]] * 4
    assert layouts[1]['bounds'][0] > layouts[0]['bounds'][2]


def test_leaders_are_faint_dashed_and_end_at_translated_point():
    detected = face([5, 5, 20, 20], [[10, 20], [-1, -1]])
    layout = dict(offset=[56, 0], landmarks=[[66, 20], [55, -1]])
    with Image.new('RGB', (100, 50)) as image:
        draw_leaders(image, [detected], [layout], (50, 50))
        assert 0 < image.getpixel((17, 20))[0] < 120
        assert image.getpixel((21, 20)) == (0, 0, 0)
        assert image.getpixel((65, 20)) == (0, 0, 0)  # final dash gap
        assert image.getpixel((67, 20)) == (0, 0, 0)
        assert image.getpixel((0, 0)) == (0, 0, 0)  # missing landmark ignored
