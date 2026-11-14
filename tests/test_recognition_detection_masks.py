"""Real refinement diagnostics, source-coordinate rendering and bounded lifetime."""
from contextlib import contextmanager
import json
from types import SimpleNamespace

import numpy as np
from PIL import Image
import pytest

from stash_ai.recognition import detection_audit, detection_masks, pose_mask_capture
from stash_ai.recognition.cli import parser
from stash_ai.recognition.config import defaults
from stash_ai.recognition.detection_masks import MaskOptions
from stash_ai.recognition.pose_refinement import RefinedPMPose
from stash_ai.recognition.pose_refinement_config import defaults as refinement_defaults
from stash_ai.recognition.sapiensid_backend import SapiensIDFaces
from tests.test_recognition_detection_audit import Detector, setup
from tests.test_recognition_detection_sweep import table
from tests.test_recognition_pose_refinement import BOX, Pose, Segmenter


class SpeckledSAM(Segmenter):
    def predict(self, points, negative=()):
        mask, score = super().predict(points)
        mask[2, 2] = True  # Raw-only speckle; never passed to PMPose.
        mask[64:70, 43:49] = False  # Hole must survive both capture and rendering.
        return mask, score


class RefinedDetector(Detector):
    def __init__(self, *, boxes=None, drift=False, score=None, mode='all'):
        super().__init__()
        data = defaults()
        data['sapiensid']['pose']['backend'] = 'pmpose'
        data['sapiensid']['pose']['refinement']['enabled'] = True
        self.config = SimpleNamespace(data=data)
        self.boxes = boxes or [BOX]
        self.pose = RefinedPMPose(Pose(drift=drift), SpeckledSAM(), refinement_defaults())
        self.pose.settings['mode'] = mode
        if score is not None:
            predict = self.pose.segmenter.predict
            self.pose.segmenter.predict = lambda points, negative=(): (predict(points, negative)[0], score)

    def detect(self, image, *, force_pose=False):
        assert force_pose
        self.calls += 1
        prediction = self.pose.predict(image, self.boxes)
        self.frame_people = SapiensIDFaces._people(self, self.boxes, [.98]*len(self.boxes), *prediction)
        self.face_matches = []
        return []


@contextmanager
def render(detector=None, options=None):
    detector = detector or RefinedDetector()
    cue = SimpleNamespace(start=1, end=2, rect=[11, 13, 160, 100])
    with Image.new('RGB', (160, 100), (40, 60, 80)) as image:
        before = image.tobytes()
        with detection_audit.render_frame(image, detector, '7', 3, cue, mask_options=options) as result:
            yield result
        assert image.tobytes() == before
        assert pose_mask_capture.mask_observer(image, 0) is None


def test_overlay_keeps_holes_pose_colors_and_coordinates_without_extra_inference():
    detector = RefinedDetector()
    with render(detector) as (png, record):
        assert png.getpixel((62, 60)) == (62, 45, 92)  # 25% purple fill.
        assert png.getpixel((46, 66)) == (40, 60, 80)  # Unfilled hole.
        assert png.getpixel((2, 2)) == (40, 60, 80)  # Removed raw speckle.
        assert png.getpixel((75, 55)) == (128, 0, 128)  # Pose drawn above mask.
        assert (np.asarray(png)[10:27, 25:46] == 255).all(axis=2).any()  # Prompt rings under point labels.
        assert record['source_dimensions'] == [160, 100]
        assert record['source_rect'] == [11, 13, 160, 100]
        assert record['mask_visualization']['people'][0] == dict(
            person_index=0, available_stages=['raw', 'cleaned'], displayed_stage='quality', quality_mask_available=True, second_pass_used=True)
        assert record['people'][0]['pose_refinement']['box_expanded']
        json.dumps(record, allow_nan=False)  # No arrays/bitmaps leak into the manifest.
    assert detector.calls == detector.releases == 1
    assert detector.pose.segmenter.frames == detector.pose.segmenter.resets == 1
    assert detector.pose.segmenter.prompts == 1
    assert detector.pose.pose.calls == [('first', 1), ('masked', 1)]


def test_contours_do_not_fill_and_off_does_not_capture_or_change_inference(monkeypatch):
    detector = RefinedDetector()
    with render(detector, MaskOptions(view='contour')) as (png, _):
        assert png.getpixel((62, 60)) == (40, 60, 80)
        assert png.getpixel((79, 60)) == (128, 0, 128)
    def forbidden(*args):
        raise AssertionError('Mask capture must stay inactive')
    monkeypatch.setattr(pose_mask_capture.FrameMasks, 'save', forbidden)
    with render(RefinedDetector(), MaskOptions(view='off')) as (png, record):
        assert png.size == (160, 100)
        assert png.getpixel((62, 60)) == (40, 60, 80)
        from stash_ai.recognition.body_collection import public_person
        assert record['people'][0]['pose_refinement'] == public_person(detector.frame_people[0])['pose_refinement']
        assert record['people'][0]['keypoints'] == detector.frame_people[0]['keypoints']
        assert record['people'][0]['pose_refinement']['status'] == 'refined'
    with Image.new('RGB', (160, 100)) as image:
        RefinedDetector().pose.predict(image, [BOX])  # Ordinary recognition also does not capture.


def test_rejected_contours_have_visible_dashes_along_diagonal_edges():
    y, x = np.indices((40, 40))
    mask = x+y <= 30
    with Image.new('RGB', (40, 40), (40, 60, 80)) as image:
        detection_masks.paint_mask(image, mask, (128, 0, 128), opacity=0, rejected=True)
        edge = {image.getpixel((i, 30-i)) for i in range(3, 28)}
        assert edge == {(40, 60, 80), (128, 0, 128)}


@pytest.mark.parametrize('kwargs,stages,used,reason', [
    ({'score': .1}, ['raw'], False, 'low_mask_score'),
    ({'drift': True}, ['raw', 'cleaned'], True, 'second_pass_core_drift'),
    ({'mode': 'overlap'}, [], False, 'no_overlap'),
])
def test_rejections_and_skips_are_explicit_and_can_be_hidden(kwargs, stages, used, reason):
    with render(RefinedDetector(**kwargs), MaskOptions(view='panels')) as (_, record):
        details = record['mask_visualization']
        assert details['people'][0]['available_stages'] == stages
        assert details['people'][0]['second_pass_used'] is used
        assert record['people'][0]['pose_refinement']['reason'] == reason
        assert [p['available'] for p in details['panels']] == ['raw' in stages, 'cleaned' in stages]
    with render(RefinedDetector(**kwargs), MaskOptions(view='panels', show_rejected=False)) as (png, record):
        assert png.getpixel((62, 60)) == (40, 60, 80)
        shown = record['mask_visualization']['people'][0]
        assert shown['quality_mask_available'] == (reason == 'second_pass_core_drift')
        assert shown['displayed_stage'] == ('quality' if shown['quality_mask_available'] else None)
        assert bool(record['mask_visualization']['panels']) == shown['quality_mask_available']
        assert record['people'][0]['pose_refinement']['reason'] == reason


def test_person_panels_distinguish_raw_and_cleaned_masks_without_moving_main_geometry():
    with render(RefinedDetector(), MaskOptions(view='panels')) as (png, record):
        raw, cleaned = record['mask_visualization']['panels']
        rx, ry = raw['offset']
        cx, cy = cleaned['offset']
        assert rx == 0 and cx == 160 and ry == cy and ry > 100
        assert png.getpixel((rx+2, ry+2)) == (128, 0, 128)
        assert png.getpixel((cx+2, cy+2)) == (40, 60, 80)
        assert png.getpixel((cx+46, cy+66)) == (40, 60, 80)
        assert 'used by PMPose' in ''.join(cleaned['header_lines'])
        assert record['people'][0]['bounds'] == [20, 10, 80, 90]
        assert record['output_dimensions'] == list(png.size)


def test_each_person_mask_uses_pose_palette_and_is_isolated_in_detail_panel():
    detector = RefinedDetector(boxes=[BOX, [100, 10, 140, 90]])
    with render(detector, MaskOptions(view='panels')) as (png, record):
        panels = record['mask_visualization']['panels']
        px, py = panels[0]['offset']
        assert png.getpixel((147, 65)) == (30, 77, 60)  # Green second mask.
        assert png.getpixel((px+147, py+65)) == (40, 60, 80)  # P0 panel excludes P1.
        assert [p['person_index'] for p in panels] == [0, 0, 1, 1]


@pytest.mark.parametrize('match', [0, None])
def test_face_colors_and_translated_geometry_remain_above_masks(match):
    from tests.helpers.recognition import FACE
    detector = RefinedDetector()
    original_detect = detector.detect
    def detect(image, **kwargs):
        original_detect(image, **kwargs)
        detector.face_matches = [match]
        return [FACE]
    detector.detect = detect
    with render(detector) as (png, record):
        face = record['faces'][0]
        assert face['box_color'] == ([255, 220, 0] if match is not None else [255, 0, 0])
        assert face['annotation_color'] == [255, 220, 0]
        assert face['box'] == list(FACE.box)
        x, y = face['display_landmarks'][0]
        assert png.getpixel((int(x), int(y))) == (255, 220, 0)


def test_captured_masks_do_not_change_results_and_are_released_on_failure():
    with Image.new('RGB', (160, 100)) as image:
        model = RefinedDetector().pose
        expected = model.predict(image, [BOX])
        with pytest.raises(RuntimeError, match='render failed'):
            with pose_mask_capture.capture_masks(image) as capture:
                actual = model.predict(image, [BOX])
                np.testing.assert_array_equal(actual[0], expected[0])
                np.testing.assert_array_equal(actual[1], expected[1])
                for key in actual[2]:
                    np.testing.assert_array_equal(actual[2][key], expected[2][key])
                assert actual[3] == expected[3]
                raw, cleaned = capture.mask(actual[3][0], 'raw'), capture.mask(actual[3][0], 'cleaned')
                assert raw[2, 2] and not cleaned[2, 2]
                assert not cleaned[66, 46]
                raise RuntimeError('render failed')
        assert capture.people == {} and capture.bytes == 0 and capture.image is None
        assert pose_mask_capture.mask_observer(image, 0) is None


def test_capture_is_scoped_to_frame_context_and_disabled_nested_capture():
    with Image.new('RGB', (160, 100)) as image, image.copy() as other:
        with pose_mask_capture.capture_masks(image) as outer:
            assert pose_mask_capture.mask_observer(other, 0) is None
            with pose_mask_capture.capture_masks(image, enabled=False):
                assert pose_mask_capture.mask_observer(image, 0) is None
            assert pose_mask_capture.mask_observer(image, 0) is not None
        assert outer.people == {}


@pytest.mark.parametrize('limit', ['bytes', 'pixels'])
def test_diagnostic_limits_fail_clearly_and_release_frame_resources(monkeypatch, limit):
    if limit == 'bytes':
        monkeypatch.setattr(pose_mask_capture, 'MAX_BYTES', 1)
    else:
        monkeypatch.setattr(detection_masks, 'MAX_PIXELS', 1)
    detector = RefinedDetector()
    with pytest.raises(ValueError, match='--mask-view'):
        with render(detector, MaskOptions(view='panels')):
            pass
    assert detector.releases == 1 and detector.pose.segmenter.resets == 1


@pytest.mark.parametrize('value', ['nan', 'inf', '-0.1', '1.1'])
def test_cli_rejects_invalid_opacity(value):
    with pytest.raises(SystemExit):
        parser().parse_args(['audit-detection', '--scene-id', '1', '--output-dir', 'out', '--mask-opacity', value])


@pytest.mark.parametrize('view', ['overlay', 'contour', 'panels', 'off'])
def test_cli_accepts_mask_display_options(view):
    args = parser().parse_args(['audit-detection', '--scene-id', '1', '--output-dir', 'out',
                               '--mask-view', view, '--mask-opacity', '0.4', '--hide-rejected-masks'])
    assert MaskOptions.from_args(args) == MaskOptions(view=view, opacity=.4, show_rejected=False)


def test_sweep_reuses_mask_renderer_with_correct_panel_offsets_and_no_bitmap_json(tmp_path, monkeypatch):
    case = setup(tmp_path, monkeypatch)
    case.args.mask_view = 'panels'
    case.args.mask_opacity = .5
    table(case, '- {parameter: sapiensid.pose.refinement.min_mask_score, value: 0.99}')
    detectors = []
    def factory(cfg, paths):
        detector = RefinedDetector()
        detector.pose.settings['min_mask_score'] = cfg.data['sapiensid']['pose']['refinement']['min_mask_score']
        detectors.append(detector)
        return detector
    case.services.assets = lambda *args, **kwargs: {}
    assert detection_audit.execute(case.args, case.cfg, services=case.services, detector_factory=factory) == 0
    manifest = json.loads((case.destination / 'manifest.json').read_text())
    assert manifest['annotations']['sam_masks']['view'] == 'panels'
    assert manifest['annotations']['sam_masks']['opacity'] == .5
    frames = manifest['scene']['frames']
    assert frames[0]['people'][0]['pose_refinement']['status'] == 'refined'
    assert frames[1]['people'][0]['pose_refinement']['reason'] == 'low_mask_score'
    with Image.open(case.destination / manifest['scene']['file']) as png:
        for frame in frames:
            px, py = frame['mask_visualization']['panels'][0]['offset']
            sx, sy = frame['source_offset']
            # Raw speckle appears in the appropriate trial/person panel.
            assert png.getpixel((sx+px+2, sy+py+2)) == (128, 0, 128)
    assert all(d.calls == d.releases == d.pose.segmenter.frames == 1 and d.closed for d in detectors)
    assert not case.stash.writes and not list(tmp_path.glob('*.sqlite3*'))
    assert len(list(case.destination.iterdir())) == 2


def test_retry_masks_stay_with_the_selected_record_and_conflicts_are_hidden():
    detector = RefinedDetector()
    with Image.new('RGB', (160,100)) as image, pose_mask_capture.capture_masks(image) as capture:
        # Two attempts for P0, either of which can become the final detection.
        first = detector.pose.predict(image, [BOX])[3][0]
        detector.pose.settings['min_mask_score'] = 1
        retry = detector.pose.predict(image, [BOX])[3][0]
        assert detection_masks._stage(capture, first) == 'cleaned'
        assert detection_masks._stage(capture, retry) == 'raw'
        assert capture.mask(retry, 'cleaned') is None
    detect = detector.detect
    def conflict(image, **kwargs):
        faces = detect(image, **kwargs)
        record = detector.frame_people[0]['pose_refinement']
        record.update(mask_conflict=True, reason='shared_mask_pixels')
        record.pop('overlap_mask', None)
        return faces
    detector.pose.settings['min_mask_score'] = .5
    detector.detect = conflict
    with render(detector, MaskOptions(view='panels', show_rejected=False)) as (_, record):
        details = record['mask_visualization']
        assert not details['people'][0]['quality_mask_available']
        assert details['people'][0]['displayed_stage'] is None and details['panels'] == []
