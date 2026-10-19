"""Pinned SCRFD 10G/2.5G five-keypoint detectors via ONNX Runtime.

This is the 9-output / three-stride / two-anchor SCRFD KPS contract. Input
preprocessing and anchor decoding follow InsightFace's official SCRFD adapter.
No InsightFace or PyTorch runtime is needed.
"""
import math

from .models import ASSETS, checked_detection, verified_bytes
from .runtime import runtime, session


class ScrfdDetector:
    STRIDES = (8, 16, 32)
    ANCHORS = 2
    INPUT_SIZE = 640

    def __init__(self, config, paths, *, runtime_info=None):
        self.config = config
        self.session = None
        key = config.data['models']['detector']
        if key not in ('scrfd_10g', 'scrfd_2_5g'):
            raise ValueError('SCRFD adapter requires a five-keypoint SCRFD detector.')
        self.key = key
        info = runtime_info or runtime(config)
        try:
            self.session, self.device = session(
                config, verified_bytes(paths[key], ASSETS[key]), runtime_info=info)
            inputs, outputs = self.session.get_inputs(), self.session.get_outputs()
            if (len(inputs) != 1 or inputs[0].type != 'tensor(float)'
                    or len(inputs[0].shape) != 4 or inputs[0].shape[0] not in (1, None)
                    or inputs[0].shape[1] != 3
                    or any(isinstance(dim, int) and dim != self.INPUT_SIZE
                           for dim in inputs[0].shape[2:])
                    or len(outputs) != 9
                    or any(output.type != 'tensor(float)' or len(output.shape) not in (2, 3)
                           for output in outputs)):
                raise ValueError('SCRFD KPS ONNX input/output contract did not match the 9-output 640px model.')
            self.input_name = inputs[0].name
            self.output_names = [output.name for output in outputs]
            self.provenance = {
                'detector': key, 'onnxruntime': info[0].__version__,
                'providers': self.session.get_providers(),
                'preprocessing': 'letterbox-top-left-rgb-(pixel-127.5)/128-640x640',
                'output': 'scrfd-three-strides-8-16-32-two-anchors-five-kps',
                'weights': {'repo': ASSETS[key].repo, 'revision': ASSETS[key].revision,
                            'filename': ASSETS[key].filename, 'sha256': ASSETS[key].sha256},
            }
        except BaseException:
            self.close()
            raise

    @classmethod
    def _blob(cls, bgr):
        """Preserve aspect ratio, upscale small VTT images, pad right/bottom."""
        import cv2
        import numpy as np

        height, width = bgr.shape[:2]
        if height < 1 or width < 1 or bgr.shape[2:] != (3,) or bgr.dtype != np.uint8:
            raise ValueError('SCRFD requires a nonempty uint8 BGR source image.')
        scale = min(cls.INPUT_SIZE / width, cls.INPUT_SIZE / height)
        output_width = max(1, min(cls.INPUT_SIZE, int(width * scale)))
        output_height = max(1, min(cls.INPUT_SIZE, int(height * scale)))
        resized = cv2.resize(bgr, (output_width, output_height), interpolation=cv2.INTER_LINEAR)
        canvas = np.zeros((cls.INPUT_SIZE, cls.INPUT_SIZE, 3), dtype=np.uint8)
        canvas[:output_height, :output_width] = resized
        rgb = canvas[:, :, ::-1].astype(np.float32)
        tensor = np.ascontiguousarray(((rgb - 127.5) / 128.).transpose(2, 0, 1)[None])
        return tensor, output_width / width, output_height / height

    @classmethod
    def decode(cls, outputs, *, scale_x, scale_y, config):
        """Decode bounded score/bbox/kps tensors and perform per-image NMS."""
        import cv2
        import numpy as np

        if len(outputs) != 9:
            raise ValueError('SCRFD KPS inference requires exactly nine outputs.')
        detections = []
        score_threshold = config.data['detection']['score_threshold']
        for layer, stride in enumerate(cls.STRIDES):
            count = (cls.INPUT_SIZE // stride) ** 2 * cls.ANCHORS
            values = []
            for number, channels in ((layer, 1), (layer + 3, 4), (layer + 6, 10)):
                value = np.asarray(outputs[number])
                if value.shape == (1, count, channels):
                    value = value[0]
                if value.shape != (count, channels) or not np.isfinite(value).all():
                    raise ValueError(f'SCRFD stride {stride} returned malformed output {number}.')
                values.append(value)
            scores, distances, landmarks = values
            selected = np.flatnonzero(scores[:, 0] > score_threshold)
            if not len(selected):
                continue
            columns = cls.INPUT_SIZE // stride
            grid_x = (selected // cls.ANCHORS) % columns
            grid_y = (selected // cls.ANCHORS) // columns
            centers = np.column_stack((grid_x, grid_y)).astype(np.float32) * stride
            offsets = distances[selected] * stride
            points = landmarks[selected].reshape(-1, 5, 2) * stride + centers[:, None, :]
            left = (centers[:, 0] - offsets[:, 0]) / scale_x
            top = (centers[:, 1] - offsets[:, 1]) / scale_y
            right = (centers[:, 0] + offsets[:, 2]) / scale_x
            bottom = (centers[:, 1] + offsets[:, 3]) / scale_y
            points[:, :, 0] /= scale_x
            points[:, :, 1] /= scale_y
            for n, index in enumerate(selected):
                if right[n] <= left[n] or bottom[n] <= top[n]:
                    continue
                detections.append(checked_detection([
                    left[n], top[n], right[n] - left[n], bottom[n] - top[n],
                    *points[n].reshape(-1), scores[index, 0],
                ]))
        if not detections:
            return []
        # Bound NMS input while retaining a substantial multiple of the
        # configured final maximum (not a single anchor per candidate).
        maximum = config.data['detection']['max_faces_per_frame']
        top_k = max(256, min(4000, maximum * 20))
        detections.sort(key=lambda item: (-item.score, item.row))
        detections = detections[:top_k]
        boxes = [list(map(float, item.box)) for item in detections]
        scores = [float(item.score) for item in detections]
        selected = cv2.dnn.NMSBoxes(boxes, scores, score_threshold,
                                    config.data['detection']['nms_threshold'])
        return [detections[int(n)] for n in np.asarray(selected).reshape(-1)[:maximum]]

    def detect(self, bgr):
        tensor, scale_x, scale_y = self._blob(bgr)
        outputs = self.session.run(self.output_names, {self.input_name: tensor})
        return self.decode(outputs, scale_x=scale_x, scale_y=scale_y, config=self.config)

    def close(self):
        self.session = None
