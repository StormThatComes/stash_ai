"""Contact decisions use independent measured head features in original VTT coordinates."""
import json
from types import SimpleNamespace

import numpy as np
from PIL import Image
import pytest

from stash_ai.recognition import contact_faces as contact
from stash_ai.recognition import rotated_faces
from stash_ai.recognition.database import FaceDatabase, encoded
from stash_ai.recognition.models import checked_detection
from tests.test_recognition_body_identity import configuration, scene
from tests.test_recognition_landmark_repair import project


def face(cx):
    return checked_detection([cx-21, 35, 42, 40, cx-7, 49, cx+7, 49,
                              cx, 56, cx-5, 64, cx+5, 64, .95])


LEFT, RIGHT = face(60), face(94)
MID = face(77)
# Two eyes belong to different people. A high score does not resolve this.
MIXED = checked_detection([*MID.box, *RIGHT.landmarks[0], *LEFT.landmarks[1],
                           *MID.row[8:14], .99])
CUE = SimpleNamespace(start=0., rect=(0,0,160,140), image='https://example.test/cue.jpg')


def pose(face):
    return dict(bounds=[0,0,160,140], keypoints=[list(face.landmarks[i]) for i in contact.HEAD]
                + [[-1,-1]]*14)


def setup_scene(db, config, faces, links, *, strict=False):
    config.data['models']['detector'] = 'retinaface_r50'
    config.data['detection'].update(pose_guided_rotation=False, landmarks_from_facebox=strict,
                                    min_face_pixels=8)
    item, bodies = scene(db, config, vectors=[[1,0],[0,1]], times=[0,0])
    for i, f in enumerate(faces):
        missing = not f.landmarks_available
        db.execute('INSERT INTO observations VALUES (?,?,?,?,?,?,?,?,?,?,?,?)',
                   (str(i),item['scan'],0,0.,CUE.image,encoded(CUE.rect),encoded(f.row),'{}',
                    'face_landmarks_invalid' if missing else None,2,None if missing else b'12345678',None))
    for address, target, link in zip(bodies, (LEFT,RIGHT), links):
        db.execute('UPDATE body_observations SET geometry=?,face=? WHERE id=?',
                   (encoded(pose(target)), link, address))
    db.commit()
    return item['scan'], bodies


def run(db, scan, cfg, backend):
    source = SimpleNamespace(cues=[CUE], frames_indices=lambda ids: (
        (i,CUE,Image.new('RGB',CUE.rect[2:])) for i in ids))
    rotated_faces.recover(db,scan,source,cfg,lambda:backend,None,lambda:None)


class Backend:
    dimensions = 2

    def __init__(self):
        self.calls = []
        self.boxes = [rotated_faces.head_crop(pose(f), CUE.rect[2:])[0] for f in (LEFT,RIGHT)]

    def contact_candidates(self, image, boxes=None):
        self.calls.append(image.size)
        if boxes is not None:
            measured = {f.box:f for f in (LEFT,MIXED,RIGHT)}
            return [measured[f.box] for f in boxes]
        # The nose of each owner is centred in its crop. Images have equal
        # size: the test substitutes this method when the target order varies.
        raise AssertionError('unexpected crop')

    def detect(self,image):
        raise AssertionError('expected a pose-conditioned measurement')

    def embed(self,image,detection):
        return np.array([1.,0.])

    def close(self):
        pass


def test_independent_heads_and_profile_abstention():
    people = [pose(LEFT), pose(RIGHT)]
    def measure(f):
        measured = contact.measured_people(people,CUE.rect[2:])
        return contact.ownership(f,measured,contact.nearby_heads(f,measured))
    assert measure(LEFT) == (0, ())
    assert measure(RIGHT) == (1, ())
    assert measure(MIXED) == (None, (0,1))
    assert measure(MID) == (None, ())
    people[1]['pose_refinement'] = dict(face_prompt=(2,RIGHT.landmarks[0]))
    assert measure(MIXED)[1] == ()


@pytest.mark.parametrize('strict',[False,True])
def test_two_good_faces_survive_and_mixed_face_is_inspectable(tmp_path, strict):
    cfg = configuration(tmp_path)
    faces = [LEFT,MIXED,RIGHT]
    if strict:
        faces = [f.without_landmarks() for f in faces]
    with FaceDatabase(cfg.database) as state:
        db = state.connection
        scan, bodies = setup_scene(db,cfg,faces,[None,None] if strict else ['0','2'],strict=strict)
        backend = Backend()
        run(db,scan,cfg,backend)
        rows = db.execute('SELECT id,rejected,embedding,geometry,quality FROM observations ORDER BY id').fetchall()
        assert [r['rejected'] for r in rows] == [None,contact.MIXED,None]
        assert rows[1]['embedding'] is None
        assert set(json.loads(rows[1]['quality'])['conflicting_pose_heads']) == set(bodies)
        assert [json.loads(r['geometry']) for r in rows] == [list(f.row) for f in (LEFT,MIXED,RIGHT)]
        for b, i in zip(bodies,('0','2')):
            assert db.execute('SELECT face FROM body_observations WHERE id=?',(b,)).fetchone()[0] == i
        assert len(backend.calls) == int(strict)
        run(db,scan,cfg,backend)
        assert len(backend.calls) == int(strict)


def test_one_mixed_box_recovers_two_heads_even_when_both_appear_in_each_crop(tmp_path):
    cfg = configuration(tmp_path)
    with FaceDatabase(cfg.database) as state:
        db = state.connection
        scan, bodies = setup_scene(db,cfg,[MIXED],['0',None])
        backend = Backend()
        ordered = sorted(zip(bodies,backend.boxes))
        def detect(image):
            box = ordered[len(backend.calls)][1]
            backend.calls.append(box)
            return [project(f,box,0) for f in (RIGHT,LEFT)]
        backend.contact_candidates = detect
        run(db,scan,cfg,backend)
        for b, f in zip(bodies,(LEFT,RIGHT)):
            row = db.execute('SELECT o.geometry,o.rejected,o.embedding FROM observations o '
                             'JOIN body_observations b ON b.face=o.id WHERE b.id=?',(b,)).fetchone()
            assert json.loads(row['geometry']) == list(f.row)
            assert row['rejected'] is None and row['embedding']
        assert db.execute("SELECT rejected FROM observations WHERE id='0'").fetchone()[0] == contact.MIXED
        run(db,scan,cfg,backend)
        assert len(backend.calls) == 2


@pytest.mark.parametrize('third',[False,True])
def test_midpoint_needs_independent_pixel_confirmation(tmp_path, third):
    cfg = configuration(tmp_path)
    with FaceDatabase(cfg.database) as state:
        db = state.connection
        scan, _ = setup_scene(db,cfg,[LEFT,MID,RIGHT],['0','2'])
        backend = Backend()
        def verify(image):
            backend.calls.append(image.size)
            box = (35,15,119,95)
            assert image.size == (84,80)
            return [project(f,box,0) for f in ((LEFT, MID, RIGHT) if third else (RIGHT,LEFT))]
        backend.contact_candidates = verify
        run(db,scan,cfg,backend)
        reason,vector = db.execute("SELECT rejected,embedding FROM observations WHERE id='1'").fetchone()
        assert reason == (None if third else contact.BETWEEN)
        assert bool(vector) == third
        assert len(backend.calls) == 1


def test_no_inference_for_two_confirmed_people(tmp_path):
    cfg = configuration(tmp_path)
    with FaceDatabase(cfg.database) as state:
        db = state.connection
        scan, _ = setup_scene(db,cfg,[LEFT,RIGHT],['0','1'])
        source = SimpleNamespace(cues=[CUE])
        rotated_faces.recover(db,scan,source,cfg,lambda:pytest.fail('loaded detector'),None,lambda:None)


def test_retry_failure_rolls_back_rejection_and_links(tmp_path):
    cfg = configuration(tmp_path)
    with FaceDatabase(cfg.database) as state:
        db = state.connection
        scan, _ = setup_scene(db,cfg,[MIXED],['0',None])
        backend = Backend()
        def fail(image):
            raise RuntimeError('inference failed')
        backend.contact_candidates = fail
        with pytest.raises(RuntimeError,match='inference failed'):
            run(db,scan,cfg,backend)
        assert db.execute("SELECT rejected FROM observations WHERE id='0'").fetchone()[0] is None
        assert db.execute("SELECT COUNT(*) FROM body_observations WHERE face='0'").fetchone()[0] == 1


def test_two_detections_claiming_same_head_abstain():
    box, turn = rotated_faces.head_crop(pose(LEFT), CUE.rect[2:])
    def detect(image):
        return [project(LEFT,box,0)]*2
    with Image.new('RGB',CUE.rect[2:]) as image:
        assert rotated_faces.detect_head(SimpleNamespace(detect=detect),image,box,turn,
            [pose(LEFT),pose(RIGHT)],0,[],contact=True) is None


def test_crop_only_verifier_uses_crop_pixels_and_never_restores_discarded_points(tmp_path):
    from stash_ai.recognition.models import YuNetDetector
    from stash_ai.recognition.retinaface import RetinaFaceDetector
    config = configuration(tmp_path)
    config.data['models']['detector'] = 'retinaface_r50'
    config.data['detection'].update(landmarks_from_facebox=True, pose_guided_rotation=False)
    detector = object.__new__(RetinaFaceDetector)
    detector.config = config
    adapter = object.__new__(YuNetDetector)
    adapter.config, adapter.ort_detector = config, detector
    crop = (28,25,92,85)
    detector._raw_many = lambda crops: [[project(LEFT,crop,0)] for _ in crops]
    with Image.new('RGB',CUE.rect[2:]) as image:
        assert adapter.contact_candidates(image, [LEFT.without_landmarks()]) == [LEFT]
        detector._raw_many = lambda crops: [[] for _ in crops]
        assert adapter.contact_candidates(image, [LEFT]) == [LEFT.without_landmarks()]


def test_distinct_confirmed_close_head_can_overlap_accepted_neighbour():
    left, right = face(66), face(79)
    people = [pose(left),pose(right)]
    box,turn = rotated_faces.head_crop(people[1],CUE.rect[2:])
    def detect(image):
        return [project(left,box,0), project(right,box,0)]
    assert rotated_faces.duplicate_face(left,right)
    with Image.new('RGB',CUE.rect[2:]) as image:
        result = rotated_faces.detect_head(SimpleNamespace(detect=detect),image,box,turn,people,1,[left],contact=True)
        assert result[0] == right


def test_contact_scan_cache_isolated_by_body_and_detector(tmp_path,monkeypatch):
    from stash_ai.recognition.scanning import scan_identity
    config = configuration(tmp_path)
    for backend, collect, invalidate in [('retinaface_r50', True, True), ('scrfd_10g',True,False),
                                         ('retinaface_r50',False,False)]:
        config.data['models']['detector'] = backend
        config.data['sapiensid']['collection']['enabled'] = collect
        before = scan_identity(config,'recognizer')
        with monkeypatch.context() as patch:
            patch.setattr(contact,'CONTRACT','previous')
            assert (scan_identity(config,'recognizer') != before) == invalidate
