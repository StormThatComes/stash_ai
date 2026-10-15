"""Visible expression/eye/mouth cues for portrait ranking, independent of identity."""
from dataclasses import asdict
import math

from .config import digest
from .models import ASSETS

WEIGHTS = ('neutral_expression', 'closed_mouth', 'open_mouth', 'open_eyes')
# Eye blink and gaze are separate from expression activity. _neutral is not an
# expression probability; derive a documented portrait heuristic instead.
ACTIVITY = (
    'browDownLeft', 'browDownRight', 'browInnerUp', 'browOuterUpLeft', 'browOuterUpRight',
    'cheekPuff', 'cheekSquintLeft', 'cheekSquintRight', 'eyeSquintLeft', 'eyeSquintRight',
    'eyeWideLeft', 'eyeWideRight', 'jawOpen', 'jawForward', 'jawLeft', 'jawRight',
    'mouthSmileLeft', 'mouthSmileRight', 'mouthFrownLeft', 'mouthFrownRight',
    'mouthPucker', 'mouthFunnel', 'mouthLeft', 'mouthRight', 'mouthStretchLeft', 'mouthStretchRight',
    'mouthPressLeft', 'mouthPressRight', 'mouthUpperUpLeft', 'mouthUpperUpRight',
    'mouthLowerDownLeft', 'mouthLowerDownRight', 'noseSneerLeft', 'noseSneerRight')
REQUIRED = (*ACTIVITY, 'eyeBlinkLeft', 'eyeBlinkRight')
CONTRACT = {'version': 2, 'models': {key: asdict(ASSETS[key]) for key in
            ('expression_mesh', 'expression_blendshapes')},
            'crop': 'square-1.5-max-face-side-eye-roll-aligned-RGB-zero-padding',
            'input': 'float32-NHWC-256-RGB-0..1', 'mesh': '478x3-crop-pixels-presence-logit>=0',
            'blendshapes': 'upstream-146-landmark-subset-original-image-pixels-to-52-scores',
            'association': 'five-mesh-anchors-mean-error<=0.35-interocular-distance',
            'neutral_activity': ACTIVITY, 'neutral': '1-max(activity)',
            'closed_mouth': '1-jawOpen', 'open_mouth': 'jawOpen', 'open_eyes': '1-max(eyeBlinkLeft,eyeBlinkRight)'}
CONTRACT_ID = digest(CONTRACT)


class ExpressionError(RuntimeError):
    pass


class ExpressionGeometryError(ValueError):
    """An observation-local face association failure, not a poisoned session."""


def required(policy):
    return any(policy['weights'][name] > 0 for name in WEIGHTS)


def valid_score(value):
    return type(value) in (int, float) and 0 <= value <= 1 and math.isfinite(value)


def measurement(blendshapes, *, crop=None, device='cpu', provenance=None):
    if not isinstance(blendshapes, dict) or any(not valid_score(blendshapes.get(key)) for key in REQUIRED):
        raise ValueError('Expression model returned incomplete/non-finite blendshape scores.')
    return {'neutral_expression': 1 - max(blendshapes[key] for key in ACTIVITY),
            'closed_mouth': 1 - blendshapes['jawOpen'], 'open_mouth': blendshapes['jawOpen'],
            'open_eyes': 1 - max(blendshapes['eyeBlinkLeft'], blendshapes['eyeBlinkRight']),
            'expression': {'contract': CONTRACT_ID, 'device': device, 'runtime': provenance or {},
                           'model_sha256': {key: ASSETS[key].sha256 for key in
                                            ('expression_mesh', 'expression_blendshapes')},
                           'blendshapes': {key: blendshapes[key] for key in REQUIRED}, 'crop': crop}}


def cached(quality):
    record = quality.get('expression')
    if not isinstance(record, dict) or record.get('contract') != CONTRACT_ID:
        return False
    try:
        expected = measurement(record.get('blendshapes'))
    except ValueError:
        return False
    return all(valid_score(quality.get(key)) and abs(quality[key] - expected[key]) < 1e-7 for key in WEIGHTS)


class LazyExpression:
    """One lazy task per run; zero weights require neither inference nor assets."""
    def __init__(self, config, paths, factory=None):
        self.config, self.paths, self.factory = config, paths, factory or FaceExpression
        self.model = None
        self._initialization_error = None
        self._inference_failures = 0

    def measure(self, image, face):
        from stash_ai.runtime.shutdown import require_idle
        require_idle()
        try:
            if self.model is None:
                if self._initialization_error is not None:
                    raise ValueError(self._initialization_error)
                try:
                    self.model = self.factory(self.config, self.paths)
                except Exception as error:
                    self._initialization_error = str(error)[:500]
                    raise
            value = self.model.measure(image, face)
            if not isinstance(value, dict) or not cached(value):
                raise ValueError('Expression model returned an invalid or incompatible measurement.')
            return value
        except ExpressionGeometryError as error:
            raise ExpressionError('Face expression measurement failed: ' + str(error)) from error
        except Exception as error:
            if self.model is not None:
                from stash_ai.runtime.cleanup import cleanup
                model, self.model = self.model, None
                self._inference_failures += 1
                if self._inference_failures >= 2:
                    self._initialization_error = 'Expression session recovery exhausted after repeated inference failures.'
                cleanup([('close failed expression model', model.close)])
            raise ExpressionError('Face expression measurement failed: ' + str(error)) from error

    def close(self):
        self._initialization_error = None
        self._inference_failures = 0
        if self.model is not None:
            model, self.model = self.model, None
            model.close()



def associated(landmarks, face):
    """Associate the mesh to the verified YuNet face, not a neighboring person."""
    import numpy as np
    points = np.asarray(landmarks)
    if points.shape != (478, 2) or not np.isfinite(points).all():
        raise ValueError('Expression model returned incompatible/non-finite landmarks.')
    expected = np.asarray(face.landmarks, dtype=float)
    expected[:2] = expected[:2][np.argsort(expected[:2, 0])]
    expected[3:] = expected[3:][np.argsort(expected[3:, 0])]
    scale = float(np.linalg.norm(expected[0] - expected[1]))
    if not math.isfinite(scale) or scale <= 1:
        raise ExpressionGeometryError('Expression target landmarks are invalid.')
    anchors = np.array([(points[33] + points[133]) / 2, (points[362] + points[263]) / 2,
                        points[1], points[61], points[291]])
    anchors[:2] = anchors[:2][np.argsort(anchors[:2, 0])]
    anchors[3:] = anchors[3:][np.argsort(anchors[3:, 0])]
    error = float(np.linalg.norm(anchors - expected, axis=1).mean() / scale)
    if error > .35:
        raise ExpressionGeometryError('Expression mesh cannot be associated with the verified identity.')


class FaceExpression:
    """Two pinned ONNX models using the shared explicit CPU/CUDA/auto policy."""
    def __init__(self, config, paths):
        from .expression_onnx import ExpressionBackend
        self.backend = ExpressionBackend(config, paths)
        self.device = self.backend.device

    def measure(self, image, face):
        shapes, crop = self.backend.measure(image, face)
        return measurement(shapes, crop=crop, device=self.device, provenance=self.backend.provenance)

    def close(self):
        self.backend.close()
