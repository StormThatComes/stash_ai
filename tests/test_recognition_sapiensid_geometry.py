"""Crop-boundary regressions through the real SapiensID preparation paths."""
from types import SimpleNamespace

import numpy as np
import pytest

torch = pytest.importorskip('torch')
pytest.importorskip('torchvision')
from PIL import Image

from stash_ai.recognition import sapiensid_backend as backend
from stash_ai.recognition.config import defaults
from stash_ai.recognition.models import checked_detection
from stash_ai.recognition.person_pipeline import PreparedCollector
from stash_ai.recognition.sapiensid_inference import SapiensIDV1Encoder
from stash_ai.recognition.sapiensid_v1 import prepare_person_input
from tests.test_recognition_sapiensid_inference import PersonModel


@pytest.fixture
def detector(monkeypatch):
    data = defaults()
    data['models']['device'] = 'cpu'
    config = SimpleNamespace(data=data)
    confidence = torch.ones((1, 17))
    confidence[0, -1] = .1
    result = SimpleNamespace(
        boxes=SimpleNamespace(xyxy=torch.tensor([[10., 20., 91., 140.]]), conf=torch.tensor([.99])),
        keypoints=SimpleNamespace(xy=torch.full((1, 17, 2), 30.), conf=confidence))
    monkeypatch.setattr(backend, 'load_pose', lambda _: SimpleNamespace(
        predict=lambda **kw: [result] * (len(kw['source']) if isinstance(kw['source'],list) else 1)))
    model = backend.SapiensIDFaces(config, {}, load_identity=False, load_detector=False)
    model.encoder = SapiensIDV1Encoder(PersonModel(), None, device='cpu')
    try:
        yield model
    finally:
        model.close()


def face_with_point(point):
    # The 1px disagreement with the person box is within association tolerance.
    return checked_detection([9, 19, 40, 40, *point, 40, 30, 30, 40, 25, 50, 40, 50, .99])


@pytest.mark.parametrize('point', [
    (9.75, 30), (30, 19.75),  # Before the crop's left/top edge.
    (91, 30), (30, 140),      # The right/bottom edge is exclusive.
    (92, 30), (30, 141),     # Face detector landmark outliers.
    (91 - 1e-7, 30),         # Rounds to the excluded edge in model float32.
    (30, 140 - 1e-7),
])
def test_outside_face_point_is_missing_in_audit_batch_and_scan_preparation(detector, point):
    face = face_with_point(point)
    original = face.row
    pipeline = PreparedCollector(detector.config, detector)
    try:
        with Image.new('RGB', (160, 180), (80, 90, 100)) as image:
            detector.detect(image, faces=[face])
            assert detector.observation_quality(image, face)['identity_rejection'] is None
            audit = detector.audit_input(image, face)
            assert audit[1][0, 17].tolist() == [-1., -1.]
            # Other measured face points keep their exact square-pad positions.
            expected = (np.asarray(face.landmarks[1:], dtype=np.float32) - [10, 20] + [19, 0]) / 120
            np.testing.assert_allclose(audit[1][0, 18:], expected, atol=1e-7)
            bounds, points = detector.person_input(image, face)
            assert bounds == [10, 20, 91, 140]
            prepared, _, _ = pipeline.prepare(image, (bounds, points))
            assert all(torch.equal(a, b) for a, b in zip(audit, prepared))
            np.testing.assert_array_equal(detector.embed(image, face), detector.embed_inputs([prepared])[0])
            assert face.row == original
    finally:
        pipeline.close()


@pytest.mark.parametrize('point', [(10, 20), (90.75, 139.75), (30, 40)])
def test_valid_face_points_and_existing_missing_pose_points_are_preserved(detector, point):
    face = face_with_point(point)
    with Image.new('RGB', (160, 180), (80, 90, 100)) as image:
        detector.detect(image, faces=[face])
        _, points, _ = detector.audit_input(image, face)
        expected = (np.asarray(face.landmarks, dtype=np.float32) - [10, 20] + [19, 0]) / 120
        np.testing.assert_allclose(points[0, 17:], expected, atol=1e-7)
        assert (points[0, :17] == -1).all(dim=1).sum().item() == 1


def test_outside_point_does_not_relax_ambiguous_person_association(detector):
    face = face_with_point((9.75, 30))
    with Image.new('RGB', (160, 180)) as image:
        detector.detect(image, faces=[face, face])
        assert detector.observation_quality(image, face)['identity_rejection'] == 'person_face_association_unavailable'
        with pytest.raises(ValueError, match='unambiguous'):
            detector.person_input(image, face)


def test_scene_scan_completes_and_reuses_person_evidence_after_boundary_disagreement(detector, tmp_path):
    from dataclasses import replace
    from stash_ai.recognition.config import plain
    from stash_ai.recognition.database import FaceDatabase
    from stash_ai.recognition.media import SpriteInput
    from stash_ai.recognition.person_collection import PersonCollector
    from stash_ai.recognition.scanning import scan_scene
    from tests.helpers.recognition import Backend, MediaClient
    from tests.test_recognition_scanning import config

    settings = config(tmp_path)
    data = plain(settings.data)
    data['sapiensid']['enabled'] = True
    data['execution']['checkpoint_cues'] = 3
    settings = replace(settings, data=data)
    face = face_with_point((9.75, 30))
    faces = Backend()
    faces.detect = lambda image: [face]
    collector = PersonCollector(settings, {}, factory=lambda *args: detector)
    client = MediaClient('7')
    try:
        with FaceDatabase(settings.database) as state, SpriteInput(client, client.scene, settings) as source:
            scan, _ = scan_scene(state, 'scope', client.scene, settings, 'ns', source,
                                 lambda: faces, on_cue=collector.collect_frame)
            collector.flush()
            assert state.connection.execute('SELECT status FROM scans WHERE id=?', (scan,)).fetchone()[0] == 'complete'
            evidence = state.connection.execute('SELECT status,embedding FROM person_evidence').fetchall()
            assert len(evidence) == 3
            assert all(row['status'] == 'ready' and len(row['embedding']) == 4096 * 4 for row in evidence)
            source.frames_indices = lambda *args: pytest.fail('completed evidence must be reused')
            collector.enrich(state.connection, scan, source)
    finally:
        collector.close()


@pytest.mark.parametrize('point,match', [
    ((-1, 20), 'inside the crop'),
    ((80, 20), 'inside the crop'),
    ((20, 145), 'inside the crop'),
    ((float('nan'), 20), 'finite pixel'),
    ((float('inf'), 20), 'finite pixel'),
])
def test_direct_encoder_input_still_rejects_invalid_geometry(point, match):
    with Image.new('RGB', (80, 145)) as image:
        with pytest.raises(ValueError, match=match):
            prepare_person_input(image, [point] + [[20, 20]] * 21)


def test_confident_pose_points_on_zero_edges_survive_and_unusable_points_do_not(detector):
    confidence=torch.ones((1,17));confidence[0,4]=0
    points=torch.full((1,17,2),20.)
    points[0,:6]=torch.tensor([[0,10],[10,0],[0,0],[-1,10],[0,10],[160,10]])
    result=SimpleNamespace(boxes=SimpleNamespace(xyxy=torch.tensor([[0.,0.,160.,180.]]),conf=torch.tensor([.99])),
                           keypoints=SimpleNamespace(xy=points,conf=confidence))
    detector.pose=SimpleNamespace(predict=lambda **kw:[result])
    with Image.new('RGB',(160,180)) as image:
        face=checked_detection([0,0,40,40,20,20,25,20,20,25,15,30,25,30,.99])
        detector.detect(image,faces=[face])
        retained=detector.frame_people[0]['keypoints']
        assert retained[:3]==[[0.,10.],[10.,0.],[0.,0.]]
        assert retained[3:6]==[[-1.,-1.]]*3
        assert detector.observation_quality(image,face)['quality']['keypoints']==14
