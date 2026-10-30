"""Complete VTT coverage and detector-accurate, state-free visual exports."""
from contextlib import contextmanager
from dataclasses import replace
import json
import os
from types import SimpleNamespace

import numpy as np
from PIL import Image
import pytest

from stash_ai.recognition import detection_audit
from stash_ai.recognition.cli import main, parser
from stash_ai.recognition.config import plain
from stash_ai.recognition.media import SpriteInput
from stash_ai.recognition.person_audit import BOX_COLOR, FACE_COLORS, POSE_COLOR
from tests.helpers.recognition import FACE, Stash, textured_image
from tests.test_recognition_alignment_audit import config


PERSON = dict(bounds=[4, 4, 156, 156], score=.98,
              keypoints=[[80., 84.], [32., 40.]] + [[-1., -1.]] * 15)


class Detector:
    def __init__(self, *, fail_at=None):
        self.calls = self.releases = 0
        self.closed = False
        self.fail_at = fail_at

    @contextmanager
    def prepared_frame(self, image):
        before = image.tobytes()
        try:
            yield
        finally:
            assert image.tobytes() == before
            self.releases += 1

    def detect(self, image, *, force_pose=False):
        assert force_pose
        self.calls += 1
        if self.calls == self.fail_at:
            raise RuntimeError('synthetic detection failure')
        # Face+person, faceless person, then a completely empty cue.
        self.frame_people = [PERSON] if self.calls < 3 else []
        self.face_matches = [0] if self.calls == 1 else []
        return [FACE] if self.calls == 1 else []

    def close(self):
        self.closed = True


def setup(tmp_path, monkeypatch, *, fail_at=None):
    monkeypatch.setenv('STASH_URL', 'http://stash/')
    cfg = config(tmp_path)
    data = plain(cfg.data)
    data['execution']['max_cues'] = 1
    # This face fails recognition's minimum size gate, but must be drawn.
    data['detection']['min_face_pixels'] = 150
    cfg = replace(cfg, data=data)
    destination = tmp_path / 'audit'
    args = parser().parse_args(['audit-detection', '--scene-id', '1', '--output-dir', str(destination),
                                '--progress-style', 'off'])
    detector, stash, requested, sources = Detector(fail_at=fail_at), Stash(), [], []

    def assets(current, *, keys, write):
        assert current is cfg
        requested.extend(keys)
        return {'yunet': 'fake-detector'}

    def source(client, scene, current):
        assert current.data['execution']['max_cues'] == 0
        assert current.data['execution']['media_cache_bytes'] == 0
        result = SpriteInput(client, scene, current)
        sources.append(result)
        return result

    services = SimpleNamespace(client=lambda **_: stash, assets=assets, source=source)
    return SimpleNamespace(cfg=cfg, args=args, destination=destination, detector=detector,
                           stash=stash, requested=requested, sources=sources, services=services)


def run(case):
    return detection_audit.execute(case.args, case.cfg, services=case.services,
                                  detector_factory=lambda *_: case.detector)


@pytest.mark.parametrize('args', [[], ['--scene-id', '1'], ['--output-dir', 'out'],
                                ['--scene-id', '0', '--output-dir', 'out']])
def test_parser_requires_one_positive_scene_and_output(args):
    with pytest.raises(SystemExit):
        parser().parse_args(['audit-detection', *args])


def test_every_original_cue_is_exported_with_both_detectors_and_no_state_writes(tmp_path, monkeypatch):
    case = setup(tmp_path, monkeypatch)
    assert run(case) == 0
    assert case.requested == ['yunet']
    assert case.detector.closed and case.stash.closed
    assert case.detector.calls == case.detector.releases == 3
    assert case.stash.writes == []
    assert case.stash.reads == [('scene', '1')]
    assert all('/stream' not in url for url in case.stash.media_calls)
    assert not list(tmp_path.glob('*.sqlite3*'))
    assert case.cfg.data['execution']['max_cues'] == 1
    assert case.cfg.data['execution']['media_cache_bytes'] > 0
    assert not case.cfg.data['sapiensid']['enabled']
    assert not case.cfg.data['sapiensid']['collection']['enabled']
    manifest = json.loads((case.destination / 'manifest.json').read_text())
    assert manifest['kind'] == 'recognition_detection_audit'
    assert manifest['execution'] == dict(max_cues=0, force_pose=True, embeddings=False)
    assert manifest['configuration']['configured_max_cues'] == 1
    assert manifest['configuration']['detection']['min_face_pixels'] == 150
    scene = manifest['scene']
    assert scene['counts'] == dict(cues_exported=3, faces_detected=1, people_detected=2, cues_without_detections=1)
    assert scene['vtt_coverage'] == dict(total_cues=3, selected_cues=3, complete_sampling=True)
    assert len(scene['vtt_source_fingerprint']) == 64
    assert [f['cue_index'] for f in scene['frames']] == [0, 1, 2]
    assert [f['start_seconds'] for f in scene['frames']] == [0., 1., 2.]
    assert [f['end_seconds'] for f in scene['frames']] == [1., 2., 3.]
    assert len(list(case.destination.glob('*.png'))) == 3
    first, faceless, empty = scene['frames']
    assert first['faces'][0]['person_index'] == 0
    assert first['faces'][0]['landmarks'] == [list(p) for p in FACE.landmarks]
    assert first['people'][0]['drawn_indices'] == [0, 1]
    assert not faceless['faces'] and len(faceless['people']) == 1
    assert not empty['faces'] and not empty['people']
    with Image.open(case.destination / first['file']) as image:
        assert image.size == (160, 160)
        assert image.getpixel((32, 40)) == POSE_COLOR
        assert image.getpixel((49, 60)) == FACE_COLORS[0]
        assert image.getpixel((80, 84)) == FACE_COLORS[2]  # face drawn above pose
        assert image.getpixel((4, 120)) == POSE_COLOR
        assert image.getpixel((16, 120)) == BOX_COLOR
    with Image.open(case.destination / empty['file']) as image, textured_image() as original:
        assert image.tobytes() == original.tobytes()
    if os.name == 'posix':
        assert case.destination.stat().st_mode & 0o777 == 0o700
        assert all(path.stat().st_mode & 0o777 == 0o600 for path in case.destination.iterdir())


def test_cli_dispatch_loads_detectors_without_identity_encoder(tmp_path, monkeypatch):
    from stash_ai.recognition import sapiensid_backend
    case = setup(tmp_path, monkeypatch)
    captured = []
    monkeypatch.setattr(sapiensid_backend, 'SapiensIDFaces',
        lambda cfg, paths, **kwargs: captured.append((cfg, kwargs)) or case.detector)
    # CLI reloads YAML and applies the requested device; source still sees all cues.
    case.services.assets = lambda cfg, **kwargs: {}
    assert main(['audit-detection', '--scene-id', '1', '--face-config', str(tmp_path / 'face.yml'),
                 '--device', 'cpu', '--progress-style', 'off', '--output-dir', str(case.destination)],
                services=case.services) == 0
    assert captured[0][1] == {'load_identity': False}
    assert captured[0][0].data['models']['device'] == 'cpu'


def test_existing_output_is_refused_before_loading_or_network(tmp_path, monkeypatch):
    case = setup(tmp_path, monkeypatch)
    case.destination.mkdir()
    sentinel = case.destination / 'keep.txt'
    sentinel.write_text('keep')
    with pytest.raises(FileExistsError):
        run(case)
    assert sentinel.read_text() == 'keep'
    assert case.requested == [] and case.stash.reads == []
    assert case.detector.calls == 0


def test_failure_closes_resources_and_does_not_claim_complete_export(tmp_path, monkeypatch):
    case = setup(tmp_path, monkeypatch, fail_at=2)
    with pytest.raises(RuntimeError, match='synthetic detection failure'):
        run(case)
    assert case.detector.closed and case.stash.closed
    assert case.detector.releases == 2
    assert case.sources[0].cache == {}
    assert len(list(case.destination.glob('*.png'))) == 1
    assert not (case.destination / 'manifest.json').exists()
    assert case.stash.writes == []


def test_full_cue_crop_contains_all_faces_and_people_in_source_coordinates(tmp_path, monkeypatch):
    from io import BytesIO
    from tests.helpers.recognition import MediaClient
    from stash_ai.recognition.models import checked_detection
    case = setup(tmp_path, monkeypatch)
    client = MediaClient(count=1)
    client.data[client.url + 'vtt/1.vtt'] = (
        b'WEBVTT\n\n00:00:02.000 --> 00:00:03.000\nsprite.jpg#xywh=17,23,320,160\n')
    with Image.new('RGB', (350, 200), (10, 20, 30)) as sheet:
        output = BytesIO()
        sheet.save(output, 'PNG')
        client.data[client.url + 'sprite/1.jpg'] = output.getvalue()
    shifted = list(FACE.row)
    for index in (0, 4, 6, 8, 10, 12):
        shifted[index] += 160
    other_face = checked_detection(shifted)
    other_person = dict(bounds=[164, 4, 316, 156], score=.97,
                        keypoints=[[240, 84], [192, 40]] + [[-1, -1]] * 15)

    def detect(image, *, force_pose):
        assert force_pose and image.size == (320, 160)
        case.detector.frame_people = [PERSON, other_person]
        # Deliberately reverse face order; associations must follow returned order.
        case.detector.face_matches = [1, 0]
        return [other_face, FACE]

    case.detector.detect = detect
    case.destination.mkdir()
    with SpriteInput(client, client.scene, case.cfg) as source:
        result = detection_audit.audit_scene(source, case.detector, '1', case.destination)
    frame = result['frames'][0]
    assert frame['source_rect'] == [17, 23, 320, 160]
    assert frame['source_dimensions'] == [320, 160]
    assert [f['person_index'] for f in frame['faces']] == [1, 0]
    assert len(frame['people']) == len(frame['faces']) == 2
    with Image.open(case.destination / frame['file']) as image:
        assert image.size == (320, 160)
        assert image.getpixel((49, 60)) == image.getpixel((209, 60)) == FACE_COLORS[0]
        assert image.getpixel((32, 40)) == image.getpixel((192, 40)) == POSE_COLOR
        assert image.getpixel((319, 159)) == (10, 20, 30)


@pytest.mark.parametrize('empty', [True, False])
def test_incomplete_or_empty_source_fails_before_models_and_output(tmp_path, monkeypatch, empty):
    case = setup(tmp_path, monkeypatch)
    source = SimpleNamespace(cues=[] if empty else [object()],
                             coverage=dict(complete_sampling=False, total_cues=3), closed=False)
    source.close = lambda: setattr(source, 'closed', True)
    case.services.source = lambda *_: source
    with pytest.raises(ValueError, match='complete VTT'):
        run(case)
    assert source.closed and case.stash.closed
    assert case.requested == [] and not case.destination.exists()


def test_force_pose_uses_production_thresholds_without_changing_normal_scan(monkeypatch):
    from stash_ai.recognition import sapiensid_backend as backend
    from stash_ai.recognition.config import defaults

    class Tensor:
        def __init__(self, data): self.data = np.array(data, dtype=np.float32)
        def detach(self): return self
        def cpu(self): return self
        def numpy(self): return self.data.copy()

    data = defaults()
    data['models']['device'] = 'cpu'
    data['sapiensid']['pose'].update(score_threshold=.6, keypoint_score_threshold=.7,
                                     nms_threshold=.42, max_people_per_frame=7)
    confidences = [[.9] * 17]
    confidences[0][1] = .1
    points = [[[20, 20]] * 17]
    points[0][2] = [105, 20]  # outside the person/image, also omitted
    result = SimpleNamespace(boxes=SimpleNamespace(xyxy=Tensor([[0, 0, 100, 100]]), conf=Tensor([.99])),
        keypoints=SimpleNamespace(xy=Tensor(points), conf=Tensor(confidences)))
    calls = []
    monkeypatch.setattr(backend, 'load_pose', lambda cfg: SimpleNamespace(
        predict=lambda **kwargs: calls.append(kwargs) or [result]))
    monkeypatch.setattr('stash_ai.recognition.sapiensid_inference.resolve_device', lambda _: 'cpu')
    monkeypatch.setattr(backend, 'load_encoder', lambda _: pytest.fail('Encoder must not load'))
    detector = backend.SapiensIDFaces(SimpleNamespace(data=data), {}, load_identity=False, load_detector=False)
    try:
        with Image.new('RGB', (100, 100)) as image:
            assert detector.detect(image, faces=[]) == []
            assert calls == []  # ordinary recognition retains its existing behavior
            assert detector.detect(image, faces=[], force_pose=True) == []
            assert len(detector.frame_people) == 1
            assert detector.frame_people[0]['keypoints'][0] == [20., 20.]
            assert detector.frame_people[0]['keypoints'][1:3] == [[-1., -1.], [-1., -1.]]
            assert calls[0]['conf'] == .6 and calls[0]['iou'] == .42
            assert calls[0]['max_det'] == 7 and calls[0]['nms'] is True
            assert calls[0]['device'] == 'cpu'
    finally:
        detector.close()
