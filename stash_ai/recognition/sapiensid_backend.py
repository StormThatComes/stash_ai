"""SapiensID v1 person embeddings anchored to real detected face observations."""
from collections import Counter
import math
from threading import get_ident

from .models import YuNetDetector
from .sapiensid_v1 import EMBEDDING_DIMENSIONS, PREPROCESSING
from .sapiensid_loading import UPSTREAM_COMMIT, load_encoder, load_pose
from .sapiensid_registry import POSE_SHA256, LEGACY_POSE_SHA256, POSE_INFERENCE, MODELS

CONTRACT = 'sapiensid-v1-coco17-pose-face-anchor-square384-v4'


def input_contract(config):
    if config.data['sapiensid']['pose'].get('backend', 'yolo') == 'vitpose_plus_large':
        return 'sapiensid-v1-vitpose-plus-large-coco17-face-anchor-square384-v1'
    checksum = config.data['sapiensid']['pose']['sha256'] or POSE_SHA256
    if checksum == LEGACY_POSE_SHA256:
        return 'sapiensid-v1-yolov8-pose-face-anchor-square384-v3'
    return CONTRACT


def identity_contract(config):
    settings = config.data['sapiensid']
    result = {'contract': input_contract(config), 'source_revision': UPSTREAM_COMMIT,
            'model_variant': settings['model']['variant'],
            'model_sha256': MODELS[settings['model']['variant']]['sha256'],
            'model_yaml_sha256': settings['model']['model_yaml_sha256'],
            'pose_sha256': settings['pose']['sha256'] or POSE_SHA256,
            'pose_inference': POSE_INFERENCE, 'preprocessing': PREPROCESSING,
            'person_score_threshold': settings['pose']['score_threshold'],
            'keypoint_score_threshold': settings['pose']['keypoint_score_threshold'],
            'person_nms_threshold': settings['pose']['nms_threshold'],
            'max_people_per_frame': settings['pose']['max_people_per_frame']}
    if settings['pose'].get('backend', 'yolo') == 'vitpose_plus_large':
        from .vitpose_registry import contract as vitpose_contract
        result['pose_refiner'] = vitpose_contract()
    return result


def association_choices(faces, people):
    """Require unique face -> person and person -> face head-region membership.

    Person boxes alone overlap too often to establish ownership. At least one
    measured pose nose/eye must fall inside the face box. Ambiguity is withheld;
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


from .person_geometry import adjust_pose, body_points


class SapiensIDFaces(YuNetDetector):
    dimensions = EMBEDDING_DIMENSIONS

    def __init__(self, config, paths, *, load_identity=True, load_detector=True):
        self.encoder = self.pose = self.pose_refiner = None
        self._person_frame = None
        self._prepared_frames = {}
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
            if config.data['sapiensid']['pose'].get('backend', 'yolo') == 'vitpose_plus_large':
                from .vitpose_pose import load
                self.pose_refiner = load(config, self.device)
            self.provenance['recognizer'] = identity_contract(config)
        except BaseException:
            self.close()
            raise

    def _owner(self):
        if get_ident() != self.owner:
            raise RuntimeError('SapiensID inference must remain on its owning thread.')

    def detect(self, image, faces=None, *, force_pose=False, _defer=None):
        """Detect faces/people; visual audits can inspect pose even without faces."""
        self._owner()
        if faces is None and not self.has_detector:
            raise ValueError('Supplemental collection requires retained face detections.')
        faces = super().detect(image) if faces is None else faces
        prepared = self._prepared_frames.pop(id(image), None)
        if prepared is not None:
            original, expected_faces, people = prepared
            if original is not image or [tuple(f.row) for f in faces] != [tuple(f.row) for f in expected_faces]:
                raise ValueError('Prepared pose frame changed before consumption.')
            return self._activate(image, faces, people)
        people = []
        if force_pose or faces or self.config.data['sapiensid']['collection']['enabled']:
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
            if (scores.ndim != 1 or boxes.shape != (len(scores), 4)
                    or any(not np.isfinite(a).all() for a in (boxes, scores))
                    or np.any((scores < 0) | (scores > 1))):
                raise ValueError('SapiensID pose detector returned malformed/non-finite geometry.')
            if len(boxes) > settings['pose']['max_people_per_frame']:
                raise ValueError('SapiensID pose detector exceeded the configured result count.')
            selected, bounds = [], []
            for index, (box, score) in enumerate(zip(boxes, scores)):
                if score <= settings['pose']['score_threshold']:
                    continue
                left, top = max(0, math.floor(box[0])), max(0, math.floor(box[1]))
                right, bottom = min(image.width, math.ceil(box[2])), min(image.height, math.ceil(box[3]))
                if left >= right or top >= bottom:
                    raise ValueError('SapiensID pose detector returned an empty person box.')
                selected.append(index)
                bounds.append([left, top, right, bottom])
            if self.pose_refiner is not None:
                if _defer is not None:
                    _defer.append((image, faces, bounds, [float(scores[i]) for i in selected]))
                    return faces
                points, confidences = self.pose_refiner.predict(image, bounds)
            else:
                points = result.keypoints.xy.detach().cpu().numpy()
                confidences = result.keypoints.conf.detach().cpu().numpy()
                if (points.shape != (len(scores), 17, 2) or confidences.shape != (len(scores), 17)
                        or any(not np.isfinite(a).all() for a in (points, confidences))
                        or np.any((confidences < 0) | (confidences > 1))):
                    raise ValueError('SapiensID pose detector returned malformed COCO17 geometry.')
                points, confidences = points[selected], confidences[selected]
            people = self._people(bounds, [float(scores[i]) for i in selected], points, confidences)
        if _defer is not None:
            _defer.append((image, faces, [], []))
            return faces
        return self._activate(image, faces, people)

    def _people(self, bounds, scores, points, confidences):
        import numpy as np
        if (points.shape != (len(bounds),17,2) or confidences.shape != (len(bounds),17)
                or any(not np.isfinite(a).all() for a in (points,confidences))
                or np.any((confidences<0)|(confidences>1))):
            raise ValueError('SapiensID pose detector returned malformed/non-finite geometry.')
        people = []
        for box, score, raw, confidence in zip(bounds, scores, points, confidences):
            left, top, right, bottom = box
            kps = raw.copy()
            usable = (confidence > self.config.data['sapiensid']['pose']['keypoint_score_threshold']) & (kps >= 0).all(axis=1)
            usable &= (kps[:,0]>=left)&(kps[:,0]<right)&(kps[:,1]>=top)&(kps[:,1]<bottom)
            kps[~usable] = -1
            people.append(dict(bounds=box,keypoints=kps.tolist(),score=float(score)))
        return people

    def detect_many(self, frames):
        self._owner()
        if self._prepared_frames:
            raise RuntimeError('Consume prepared pose frames before preparing another batch.')
        if self.pose_refiner is None or not callable(getattr(self.pose_refiner, 'predict_many', None)):
            for image, faces in frames:
                self.detect(image, faces=faces)
                self._prepared_frames[id(image)] = image, faces, self.frame_people
            return
        deferred = []
        for image, faces in frames:
            self.detect(image, faces=faces, _defer=deferred)
        predictions = self.pose_refiner.predict_many([(image,bounds) for image,_,bounds,_ in deferred])
        if len(predictions) != len(deferred):
            raise ValueError('ViTPose returned an invalid frame count.')
        prepared = {}
        for (image,faces,bounds,scores), (points,confidence) in zip(deferred,predictions):
            prepared[id(image)] = image, faces, self._people(bounds,scores,points,confidence)
        self._prepared_frames = prepared

    def _activate(self, image, faces, people):
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
        from .body_collection import quality
        measured=quality(image,person,self.frame_people,self.config.data['sapiensid']['collection'])[0] if person else {}
        return {'identity_input': input_contract(self.config), 'person': person, 'quality': measured,
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
        points = body_points(person)
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
        self._prepared_frames.clear()
        self.pose = None
        self.pose_refiner = None
        try:
            if self.encoder is not None:
                self.encoder.close()
        finally:
            self.encoder = None
            super().close()
