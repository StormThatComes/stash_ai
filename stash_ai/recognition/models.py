"""Pinned face assets and CPU OpenCV adapters; independent of tag vocabularies."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path

from .config import digest


@dataclass(frozen=True)
class Asset:
    repo: str
    revision: str
    filename: str
    sha256: str
    size: int
    license: str
    dimensions: int = 0


ASSETS = {
    'yunet': Asset('opencv/face_detection_yunet', '3cc26e7f1014a5ee5d74a42acee58bafc9d0a310',
                   'face_detection_yunet_2023mar.onnx',
                   '8f2383e4dd3cfbb4553ea8718107fc0423210dc964f9f4280604804ed2552fa4', 232589, 'MIT'),
    'sface': Asset('opencv/face_recognition_sface', '3d7082438a6e4551e840c9b2bb60b71e8da4b524',
                   'face_recognition_sface_2021dec.onnx',
                   '0ba9fbfa01b5270c96627c4ef784da859931e02f04419c829e83484087c34e79', 38696353,
                   'Apache-2.0', 128),
}


def verify(path, asset):
    with Path(path).open('rb') as stream:
        if Path(path).stat().st_size != asset.size:
            raise ValueError(f'{asset.filename}: wrong asset size; existing files were not replaced.')
        if hashlib.file_digest(stream, 'sha256').hexdigest() != asset.sha256:
            raise ValueError(f'{asset.filename}: wrong checksum; existing files were not replaced.')


def ensure_assets(config, *, write=print):
    from stash_ai.backends.downloads import install_hf_asset
    result = {}
    for key in (config.data['models']['detector'], config.data['models']['recognizer']):
        asset = ASSETS[key]
        path = config.model_dir / key / asset.filename
        if not path.is_file():
            if not config.data['models']['download_missing']:
                raise FileNotFoundError(f'Missing face asset {path}; download_missing is disabled.')
            write(f'Recognition asset: downloading {key} at {asset.revision[:12]}.')
            install_hf_asset(asset.repo, asset.filename, asset.revision, path,
                             expected_sha256=asset.sha256, max_bytes=asset.size)
        verify(path, asset)
        result[key] = path
    return result


def namespace(config):
    from dataclasses import asdict
    return digest({'format': 1, 'alignment': 'opencv-five-landmarks-v1',
                   'assets': {key: asdict(ASSETS[key]) for key in
                              (config.data['models']['detector'], config.data['models']['recognizer'])}})


@dataclass(frozen=True)
class Detection:
    row: tuple[float, ...]

    @property
    def box(self):
        return self.row[:4]

    @property
    def landmarks(self):
        return tuple(zip(self.row[4:14:2], self.row[5:14:2]))

    @property
    def score(self):
        return self.row[14]


def checked_detection(row):
    import math
    values = tuple(float(v) for v in row)
    if len(values) != 15 or not all(math.isfinite(v) for v in values):
        raise ValueError('Face detector returned malformed/non-finite geometry.')
    if values[2] <= 0 or values[3] <= 0 or not 0 <= values[-1] <= 1:
        raise ValueError('Face detector returned invalid dimensions/confidence.')
    return Detection(values)


def normalized_embedding(value, dimensions):
    import numpy as np
    result = np.asarray(value, dtype=np.float32).reshape(-1)
    if result.shape != (dimensions,) or not np.isfinite(result).all():
        raise ValueError('Face recognizer returned incompatible/non-finite embedding.')
    norm = float(np.linalg.norm(result))
    if norm <= 1e-12:
        raise ValueError('Face recognizer returned a zero-norm embedding.')
    return (result / norm).copy()


class OpenCVFaces:
    """YuNet + SFace. Official wheels provide a tested CPU DNN backend."""
    def __init__(self, config, paths):
        if config.data['models']['device'] == 'cuda':
            raise ValueError('YuNet/SFace uses the supported CPU OpenCV backend; set device: cpu/auto.')
        import cv2
        self.cv = cv2
        self.config = config
        self.device = 'cpu'
        settings = config.data['detection']
        self.detector = cv2.FaceDetectorYN.create(str(paths['yunet']), '', (320, 320),
                                                 settings['score_threshold'], settings['nms_threshold'],
                                                 settings['max_faces_per_frame'])
        self.recognizer = cv2.FaceRecognizerSF.create(str(paths['sface']), '')
        self.dimensions = ASSETS['sface'].dimensions
        for key, path in paths.items():
            verify(path, ASSETS[key])

    def _bgr(self, image):
        import numpy as np
        with image.convert('RGB') as converted:
            return np.ascontiguousarray(np.asarray(converted)[:, :, ::-1])

    def detect(self, image):
        data = self._bgr(image)
        self.detector.setInputSize((image.width, image.height))
        _, rows = self.detector.detect(data)
        if rows is None:
            return []
        if len(rows) > self.config.data['detection']['max_faces_per_frame']:
            raise ValueError('Face detector exceeded configured result count.')
        return [checked_detection(row) for row in rows]

    def embed(self, image, detection):
        import numpy as np
        aligned = self.recognizer.alignCrop(self._bgr(image), np.asarray(detection.row, dtype=np.float32))
        return normalized_embedding(self.recognizer.feature(aligned), self.dimensions)

    def close(self):
        self.detector = self.recognizer = None


def load_backend(config, paths):
    return OpenCVFaces(config, paths)
