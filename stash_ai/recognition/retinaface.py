"""Pinned RetinaFace ResNet-50 box/five-landmark inference using the shared ORT runtime.

The adapter follows RetinaFace's prior-box contract: strides 8/16/32,
two sizes per cell, variances .1/.2 and BGR mean subtraction. No upstream
Python package or executable model code is downloaded.
"""
from functools import lru_cache

import numpy as np

from .models import ASSETS, checked_detection, verified_bytes
from .runtime import runtime, session


class RetinaFaceDetector:
    INPUT_SIZE = 640
    SIZES = ((8, (16, 32)), (16, (64, 128)), (32, (256, 512)))

    def __init__(self, config, paths, *, runtime_info=None):
        self.config = config
        self.session = None
        self.key = config.data['models']['detector']
        if self.key != 'retinaface_r50':
            raise ValueError('RetinaFace adapter requires retinaface_r50.')
        info = runtime_info or runtime(config)
        try:
            self.session, self.device = session(
                config, verified_bytes(paths[self.key], ASSETS[self.key]), runtime_info=info)
            inputs, outputs = self.session.get_inputs(), self.session.get_outputs()
            count = len(self.priors())
            if (len(inputs) != 1 or inputs[0].type != 'tensor(float)'
                    or len(inputs[0].shape) != 4 or inputs[0].shape[1] != 3
                    or isinstance(inputs[0].shape[0], int) and inputs[0].shape[0] != 1
                    or any(isinstance(n, int) and n != self.INPUT_SIZE for n in inputs[0].shape[2:])
                    or len(outputs) != 3
                    or any(o.type != 'tensor(float)' or len(o.shape) != 3
                           or isinstance(o.shape[0], int) and o.shape[0] != 1
                           or isinstance(o.shape[1], int) and o.shape[1] != count
                           or o.shape[2] != channels for o, channels in zip(outputs, (4, 2, 10)))):
                raise ValueError('RetinaFace ONNX contract requires a 640px input and box/score/five-landmark outputs.')
            self.input_name = inputs[0].name
            self.output_names = [o.name for o in outputs]
            asset = ASSETS[self.key]
            self.provenance = dict(detector=self.key, onnxruntime=info[0].__version__,
                providers=self.session.get_providers(),
                preprocessing='letterbox-top-left-bgr-mean-104-117-123-640x640',
                output='retinaface-priors-8-16-32-variance-0.1-0.2-five-kps',
                weights=dict(repo=asset.repo, revision=asset.revision,
                             filename=asset.filename, sha256=asset.sha256))
        except BaseException:
            self.close()
            raise

    @classmethod
    @lru_cache(maxsize=4)
    def priors(cls):
        # Pixel-space centers/sizes; order is level, row, column, anchor size.
        rows = []
        for stride, sizes in cls.SIZES:
            side = cls.INPUT_SIZE // stride
            y, x = np.mgrid[:side, :side]
            centers = (np.column_stack((x.ravel(), y.ravel())) + .5) * stride
            rows.append(np.column_stack((np.repeat(centers, len(sizes), axis=0),
                np.tile(np.column_stack((sizes, sizes)), (len(centers), 1)))))
        values = np.concatenate(rows).astype(np.float32)
        values.flags.writeable = False
        return values

    @classmethod
    def _blob(cls, bgr):
        import cv2
        if bgr.ndim != 3 or bgr.shape[2] != 3 or min(bgr.shape[:2]) < 1 or bgr.dtype != np.uint8:
            raise ValueError('RetinaFace requires a nonempty uint8 BGR source image.')
        height, width = bgr.shape[:2]
        scale = cls.INPUT_SIZE / max(width, height)
        w, h = max(1, int(width * scale)), max(1, int(height * scale))
        canvas = np.zeros((cls.INPUT_SIZE, cls.INPUT_SIZE, 3), dtype=np.uint8)
        canvas[:h, :w] = cv2.resize(bgr, (w, h), interpolation=cv2.INTER_LINEAR)
        values = canvas.astype(np.float32) - np.array([104, 117, 123], dtype=np.float32)
        return np.ascontiguousarray(values.transpose(2, 0, 1)[None]), w / width, h / height

    @classmethod
    def decode(cls, outputs, *, scale_x, scale_y, config):
        import cv2
        priors = cls.priors()
        if len(outputs) != 3 or any(np.shape(o) != (1, len(priors), channels)
                or not np.isfinite(o).all() for o, channels in zip(outputs, (4, 2, 10))):
            raise ValueError('RetinaFace returned malformed/non-finite box/score/landmark tensors.')
        if not all(np.isfinite(s) and s > 0 for s in (scale_x, scale_y)):
            raise ValueError('RetinaFace source scales must be positive and finite.')
        location, confidence, landmarks = (np.asarray(o)[0] for o in outputs)
        if np.any((confidence < 0) | (confidence > 1)):
            raise ValueError('RetinaFace returned invalid confidence probabilities.')
        policy = config.data['detection']
        threshold = policy.get('candidate_score_threshold')
        threshold = policy['score_threshold'] if threshold is None else threshold
        indices = np.flatnonzero(confidence[:, 1] > threshold)
        maximum = policy['max_faces_per_frame']
        top_k = max(256, min(4000, maximum * 20))
        indices = indices[np.argsort(-confidence[indices, 1], kind='stable')[:top_k]]
        if not len(indices):
            return []
        anchor, offsets = priors[indices], location[indices]
        scale = np.array([scale_x, scale_y], dtype=np.float32)
        with np.errstate(over='ignore', invalid='ignore'):
            center = anchor[:, :2] + offsets[:, :2] * .1 * anchor[:, 2:]
            size = anchor[:, 2:] * np.exp(offsets[:, 2:] * .2)
            boxes = np.column_stack(((center - size / 2) / scale, size / scale))
            points = (anchor[:, None, :2] + landmarks[indices].reshape(-1, 5, 2)
                      * .1 * anchor[:, None, 2:]) / scale
        if not np.isfinite(boxes).all() or not np.isfinite(points).all() or np.any(boxes[:, 2:] <= 0):
            raise ValueError('RetinaFace decoded invalid face geometry.')
        scores = confidence[indices, 1]
        selected = cv2.dnn.NMSBoxes(boxes.tolist(), scores.tolist(), threshold, policy['nms_threshold'])
        return [checked_detection([*boxes[i], *points[i].ravel(), scores[i]])
                for i in np.asarray(selected).reshape(-1)[:maximum]]

    def _raw(self, image):
        tensor, sx, sy = self._blob(image)
        outputs = self.session.run(self.output_names, {self.input_name: tensor})
        return self.decode(outputs, scale_x=sx, scale_y=sy, config=self.config)

    def _raw_many(self, images):
        # Use one input per run to bound activation memory. Preserve caller
        # order without retaining multiple 640px activations.
        return [self._raw(image) for image in images]

    def detect_many(self, images):
        from .face_landmark_repair import repair
        images = list(images)
        return repair(self, images, self._raw_many(images))

    def detect(self, image):
        return self.detect_many([image])[0]

    def close(self):
        self.session = None
