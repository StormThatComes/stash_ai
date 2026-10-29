"""SapiensID v1 person embeddings anchored to real detected face observations."""
from collections import Counter
import math
from threading import get_ident

from .models import YuNetDetector
from .sapiensid_v1 import EMBEDDING_DIMENSIONS, PREPROCESSING
from .sapiensid_loading import UPSTREAM_COMMIT, load_encoder, load_pose
from .sapiensid_registry import POSE_SHA256, LEGACY_POSE_SHA256, POSE_INFERENCE, MODELS

CONTRACT = 'sapiensid-v1-coco17-pose-face-anchor-square384-v3'


def input_contract(config):
    checksum = config.data['sapiensid']['pose']['sha256'] or POSE_SHA256
    if checksum == LEGACY_POSE_SHA256:
        return 'sapiensid-v1-yolov8-pose-face-anchor-square384-v2'
    return CONTRACT


def identity_contract(config):
    settings = config.data['sapiensid']
    return {'contract': input_contract(config), 'source_revision': UPSTREAM_COMMIT,
            'model_variant': settings['model']['variant'],
            'model_sha256': MODELS[settings['model']['variant']]['sha256'],
            'model_yaml_sha256': settings['model']['model_yaml_sha256'],
            'pose_sha256': settings['pose']['sha256'] or POSE_SHA256,
            'pose_inference': POSE_INFERENCE, 'preprocessing': PREPROCESSING,
            'person_score_threshold': settings['pose']['score_threshold'],
            'keypoint_score_threshold': settings['pose']['keypoint_score_threshold'],
            'person_nms_threshold': settings['pose']['nms_threshold'],
            'max_people_per_frame': settings['pose']['max_people_per_frame']}


def association_choices(faces, people):
    """Require unique face -> person and person -> face head-region membership.

    Person boxes alone overlap too often to establish ownership. At least one
    measured YOLO nose/eye must fall inside the face box. Ambiguity is withheld;
    neither detector confidence nor ordering breaks a cross-person tie.
    """
    choices = []
    for face in faces:
        x, y, w, h = face.box
        indices = []
        for index, person in enumerate(people):
            left, top, right, bottom = person['bounds']
            # Small detector disagreement is allowed only with measured head
            # points; reciprocal uniqueness below still includes rejected faces.
            tolerance = .05 * min(w, h)
            if not (left-tolerance <= x and top-tolerance <= y and
                    right+tolerance >= x+w and bottom+tolerance >= y+h):
                continue
            head = person['keypoints'][:3]
            if any(px >= 0 and py >= 0 and x <= px <= x+w and y <= py <= y+h for px, py in head):
                indices.append(index)
        choices.append(indices)
    return choices


def associate(faces, people):
    choices = association_choices(faces, people)
    usage = Counter(index for indices in choices for index in indices)
    return [indices[0] if len(indices) == 1 and usage[indices[0]] == 1 else None for indices in choices]


def adjust_pose(points):
    """Match upstream image-left/right canonicalization; keep missing pairs."""
    points = [list(p) for p in points]
    for left, right in zip((1,3,5,7,9,11,13,15), (2,4,6,8,10,12,14,16)):
        a, b = points[left], points[right]
        av, bv = a != [-1., -1.], b != [-1., -1.]
        if (av and bv and a[0] < b[0]) or (av and not bv and a[0] < .5) or (bv and not av and b[0] > .5):
            points[left], points[right] = b, a
    # Upstream drops nose/eye outliers when both ears and shoulders are known.
    if all(points[i] != [-1., -1.] for i in (3,4,5,6)):
        lo, hi = min(points[i][0] for i in (3,4,5,6)), max(points[i][0] for i in (3,4,5,6))
        for i in (0,1,2):
            if points[i] != [-1., -1.] and not lo <= points[i][0] <= hi:
                points[i] = [-1., -1.]
    return points


class SapiensIDFaces(YuNetDetector):
    dimensions = EMBEDDING_DIMENSIONS

    def __init__(self, config, paths, *, load_identity=True, load_detector=True):
        self.encoder = self.pose = None
        self._person_frame = None
        self.owner = get_ident()
        try:
            self.config=config
            self.provenance={}
            self.detector=self.ort_detector=None
            self.has_detector=load_detector
            if load_detector:
                super().__init__(config, paths)
            from .sapiensid_inference import resolve_device
            if load_identity:
                self.encoder = load_encoder(config)
            self.device = str(self.encoder.device if self.encoder else resolve_device(config.data['models']['device']))
            self.pose = load_pose(config)
            self.provenance['recognizer'] = identity_contract(config)
        except BaseException:
            self.close()
            raise

    def _owner(self):
        if get_ident() != self.owner:
            raise RuntimeError('SapiensID inference must remain on its owning thread.')

    def detect(self, image, faces=None):
        self._owner()
        if faces is None and not self.has_detector:
            raise ValueError('Supplemental collection requires retained face detections.')
        faces = super().detect(image) if faces is None else faces
        people = []
        if faces or self.config.data['sapiensid']['collection']['enabled']:
            settings = self.config.data['sapiensid']
            # Explicitly select the one-to-many head: iou remains effective on
            # YOLO26 instead of being silently ignored by NMS-free inference.
            results = self.pose.predict(source=image, device=self.device, verbose=False, nms=True,
                                        conf=settings['pose']['score_threshold'],
                                        iou=settings['pose']['nms_threshold'],
                                        max_det=settings['pose']['max_people_per_frame'])
            if len(results) != 1:
                raise ValueError('SapiensID pose detector returned an invalid frame count.')
            result = results[0]
            import numpy as np
            boxes = result.boxes.xyxy.detach().cpu().numpy()
            scores = result.boxes.conf.detach().cpu().numpy()
            points = result.keypoints.xy.detach().cpu().numpy()
            confidences = result.keypoints.conf.detach().cpu().numpy()
            if (boxes.shape != (len(scores), 4) or points.shape != (len(scores), 17, 2) or
                    confidences.shape != (len(scores), 17) or any(not np.isfinite(a).all() for a in (boxes,scores,points,confidences)) or
                    np.any((scores < 0) | (scores > 1)) or np.any((confidences < 0) | (confidences > 1))):
                raise ValueError('SapiensID pose detector returned malformed/non-finite geometry.')
            if len(boxes) > settings['pose']['max_people_per_frame']:
                raise ValueError('SapiensID pose detector exceeded the configured result count.')
            for box, score, kps, confidence in zip(boxes, scores, points, confidences):
                if score <= settings['pose']['score_threshold']:
                    continue
                left, top = max(0, math.floor(box[0])), max(0, math.floor(box[1]))
                right, bottom = min(image.width, math.ceil(box[2])), min(image.height, math.ceil(box[3]))
                if left >= right or top >= bottom:
                    raise ValueError('SapiensID pose detector returned an empty person box.')
                usable = (confidence > settings['pose']['keypoint_score_threshold']) & (kps > 0).all(axis=1)
                usable &= (kps[:,0] >= left) & (kps[:,0] < right) & (kps[:,1] >= top) & (kps[:,1] < bottom)
                kps[~usable] = -1
                people.append({'bounds': [left,top,right,bottom], 'keypoints': kps.tolist(), 'score': float(score)})
        matches = associate(faces, people)
        self.frame_people, self.face_matches = people, matches
        self._person_frame = image, {tuple(face.row): people[index] if index is not None else None
                                     for face, index in zip(faces, matches)}
        return faces

    def _person(self, image, face):
        if self._person_frame is None or self._person_frame[0] is not image:
            self.detect(image)
        return self._person_frame[1].get(tuple(face.row))

    def observation_quality(self, image, face):
        self._owner()
        person = self._person(image, face)
        return {'identity_input': input_contract(self.config), 'person': person,
                'identity_rejection': None if person else 'person_face_association_unavailable'}

    def _person_geometry(self, image, face):
        import numpy as np
        self._owner()
        person = self._person(image, face)
        if person is None:
            raise ValueError('SapiensID requires an unambiguous measured face/person association.')
        return self.person_geometry(image, person, face)

    def person_geometry(self, image, person, face=None):
        import numpy as np
        self._owner()
        left, top, right, bottom = person['bounds']
        side = max(right-left, bottom-top)
        padx, pady = (side-(right-left))//2, (side-(bottom-top))//2
        # Canonicalization uses the normalized square coordinate system.
        normalized = [[(x-left+padx)/side, (y-top+pady)/side] if x >= 0 else [-1.,-1.]
                      for x,y in person['keypoints']]
        body = adjust_pose(normalized)
        points = [[x*side-padx, y*side-pady] if x >= 0 else [-1.,-1.] for x,y in body]
        points.extend([[x-left, y-top] for x,y in face.landmarks] if face is not None else [[-1.,-1.]]*5)
        # Face/person detectors can disagree at the crop boundary even after a
        # unique association. Omit out-of-crop measurements, never clamp them
        # into fabricated border landmarks. Check in the same float32 domain
        # as prepare_person_input: rounding can put a point on the excluded
        # right/bottom edge. All inference and audit paths share this geometry.
        points = np.asarray(points, dtype=np.float32)
        if points.shape != (22, 2) or not np.isfinite(points).all():
            raise ValueError('SapiensID requires 22 finite pixel landmark pairs.')
        outside = ((points < 0) | (points >= [right-left, bottom-top])).any(axis=1)
        points[outside] = -1
        return person['bounds'], points.tolist()

    def person_input(self, image, face):
        return self._person_geometry(image,face)

    def embed_inputs(self, prepared):
        self._owner()
        return self.encoder.embed_prepared(prepared)

    def embed(self, image, face):
        return self.embed_faces(image,[face])[0]

    def embed_faces(self, image, faces):
        from contextlib import ExitStack
        self._owner()
        if self.encoder is None:
            raise RuntimeError('The alignment audit does not load a recognition encoder.')
        if not 1<=len(faces)<=self.encoder.max_batch_size:
            raise ValueError('SapiensID face batch exceeds the configured limit.')
        with ExitStack() as stack:
            persons=[]
            for face in faces:
                bounds,points=self._person_geometry(image,face)
                persons.append((stack.enter_context(image.crop(bounds)),points))
            return self.encoder.embed_person_batch(persons)

    def audit_input(self, image, face):
        from .sapiensid_v1 import prepare_person_input
        bounds, points = self._person_geometry(image, face)
        with image.crop(bounds) as crop:
            return prepare_person_input(crop, points)


    def release_frame(self, previous):
        super().release_frame(previous)
        self._person_frame = None

    def close(self):
        self._owner()
        self._person_frame = None
        self.pose = None
        try:
            if self.encoder is not None:
                self.encoder.close()
        finally:
            self.encoder = None
            super().close()
