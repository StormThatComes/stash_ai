"""Pinned face assets and CPU/GPU adapters; independent of tag vocabularies."""
from __future__ import annotations

from dataclasses import dataclass
from contextlib import contextmanager
import hashlib
import os
import stat
from pathlib import Path

from .config import digest
from .progress import tracking


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
    'expression_mesh': Asset('thedemons/mirayu-models', 'dc44090caa7ce49c81139e4fd7c22b63f81943a3',
                             'mediapipe-face-landmarker/face_landmarks_detector.onnx',
                             'de2cda3bdaab8ba37d8f1236a7091c301c1cff4dd0460f5b43c03df042c6d9c0',
                             4921002, 'Apache-2.0'),
    'expression_blendshapes': Asset('thedemons/mirayu-models', 'dc44090caa7ce49c81139e4fd7c22b63f81943a3',
                                   'mediapipe-face-landmarker/face_blendshapes.onnx',
                                   '24ce8a9c501db3d2c6d32958d1b9d867d3b15de3de3a896cdc8d28641b30b608',
                                   1883471, 'Apache-2.0'),
    'faceocc': Asset('mertakin/FaceOcc', '03f229dc75fa14ae480cca9810f983912c2730ad',
                     'model.safetensors', '5d6d27a9cb221692425840a80eb4efc07e6245670b233e2eb618e83fd9b7caf6',
                     57377516, 'MIT'),
    'yunet': Asset('opencv/face_detection_yunet', '3cc26e7f1014a5ee5d74a42acee58bafc9d0a310',
                   'face_detection_yunet_2023mar.onnx',
                   '8f2383e4dd3cfbb4553ea8718107fc0423210dc964f9f4280604804ed2552fa4', 232589, 'MIT'),
    'sface': Asset('opencv/face_recognition_sface', '3d7082438a6e4551e840c9b2bb60b71e8da4b524',
                   'face_recognition_sface_2021dec.onnx',
                   '0ba9fbfa01b5270c96627c4ef784da859931e02f04419c829e83484087c34e79', 38696353,
                   'Apache-2.0', 128),
    'auraface': Asset('fal/AuraFace-v1', 'af6d057c9b0ec4071d4c49c80e3539258798b609',
                     'glintr100.onnx',
                     'a7933ea5330113b01c9b60351d8f4c33003f145d8470ac5f0e52ee2effe25c60',
                     260694151, 'Apache-2.0', 512),
}


def _asset_bytes(path, asset, *, materialize):
    """Hash a bounded descriptor read; materialized bytes are the loader input."""
    with Path(path).open('rb') as stream:
        before = os.fstat(stream.fileno())
        if not stat.S_ISREG(before.st_mode) or before.st_size != asset.size:
            raise ValueError(f'{asset.filename}: wrong asset size; existing files were not replaced.')
        if materialize:
            blob = stream.read(asset.size + 1)
            size, checksum = len(blob), hashlib.sha256(blob).hexdigest()
        else:
            remaining, size, checksum = asset.size + 1, 0, hashlib.sha256()
            while remaining:
                chunk = stream.read(min(1024 * 1024, remaining))
                if not chunk:
                    break
                size += len(chunk)
                remaining -= len(chunk)
                checksum.update(chunk)
            checksum = checksum.hexdigest()
            blob = None
        after = os.fstat(stream.fileno())
        if size != asset.size or after.st_size != asset.size:
            raise ValueError(f'{asset.filename}: wrong asset size; existing files were not replaced.')
        if checksum != asset.sha256:
            raise ValueError(f'{asset.filename}: wrong checksum; existing files were not replaced.')
        return blob


def verify(path, asset):
    _asset_bytes(path, asset, materialize=False)


def verified_bytes(path, asset):
    # Returning immutable bytes binds verification to the exact object consumed
    # by OpenCV/ORT/safetensors. No verified path is reopened by a loader.
    return _asset_bytes(path, asset, materialize=True)


def asset_selection(config):
    from .visibility import required
    selected = [config.data['models']['detector'], config.data['models']['recognizer']]
    identity_assets = set(selected)
    detection_visibility = config.data['detection']['min_visible_face_fraction'] > 0
    if detection_visibility or (config.data['portraits']['enabled'] and required(config.data['portraits'])):
        selected.append('faceocc')
    if detection_visibility:
        identity_assets.add('faceocc')
    from .expression import required as expression_required
    if config.data['portraits']['enabled'] and expression_required(config.data['portraits']):
        selected.extend(('expression_mesh', 'expression_blendshapes'))
    return selected, identity_assets


def ensure_assets(config, *, write=print, keys=None):
    if keys is None:
        selected, identity_assets = asset_selection(config)
    else:
        selected = list(dict.fromkeys(keys))
        identity_assets = set(selected).intersection(asset_selection(config)[1])
    if any(key not in ASSETS for key in selected):
        raise ValueError('Unknown recognition asset requested.')
    if any(key != 'faceocc' for key in selected):
        from .runtime import runtime
        runtime(config)  # Prepare a device only for assets whose consumers need ORT.
    from stash_ai.backends.downloads import install_hf_asset
    result = {}
    with tracking(
        selected,
        'Verify recognition assets',
        unit='asset',
    ) as tracked:
        for key in tracked:
            asset = ASSETS[key]
            path = config.model_dir / key / asset.filename
            result[key] = path
            try:
                if not path.is_file():
                    if not config.data['models']['download_missing']:
                        raise FileNotFoundError(f'Missing face asset {path}; download_missing is disabled.')
                    write(f'Recognition asset: downloading {key} at {asset.revision[:12]}.')
                    install_hf_asset(asset.repo, asset.filename, asset.revision, path,
                                     expected_sha256=asset.sha256, max_bytes=asset.size)
                verify(path, asset)
            except Exception as error:
                if key in identity_assets:
                    raise
                from stash_ai.stash_api.transport import safe_error
                # Keep the requested path so lazy initialization checks the pinned
                # asset again and reports an unavailable cue, never an unverified model.
                write(f'Portrait quality asset {key} unavailable: ' + safe_error(error)[:500])
    return result


def namespace(config):
    from dataclasses import asdict
    from .context import policy_digest
    alignment = ('arcface-112-similarity-rgb-127.5-v1' if config.data['models']['recognizer'] == 'auraface'
                 else 'opencv-five-landmarks-v1')
    return policy_digest(config, ('namespace', alignment, ASSETS[config.data['models']['detector']],
                                 ASSETS[config.data['models']['recognizer']]), lambda: {
                   'format': 1, 'alignment': alignment,
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
    raw = np.asarray(value)
    if raw.dtype.kind not in 'fiu':
        raise ValueError('Face recognizer must return real numeric embedding values.')
    result = raw.astype(np.float32).reshape(-1)
    if result.shape != (dimensions,) or not np.isfinite(result).all():
        raise ValueError('Face recognizer returned incompatible/non-finite embedding.')
    norm = float(np.linalg.norm(result.astype(np.float64)))
    if norm <= 1e-12:
        raise ValueError('Face recognizer returned a zero-norm embedding.')
    result = (result.astype(np.float64) / norm).astype(np.float32)
    if not np.isfinite(result).all() or abs(float(np.linalg.norm(result.astype(np.float64))) - 1.) > .001:
        raise ValueError("Face recognizer returned an invalid normalized embedding.")
    return result


class YuNetDetector:
    """Shared CPU/OpenCV or GPU/ORT detector with independent recognizers."""
    def __init__(self, config, paths, *, runtime_info=None):
        import cv2
        self.cv = cv2
        self.config = config
        self.device = 'cpu'
        settings = config.data['detection']
        # Identity construction owns only its detector. Optional quality
        # models verify their own bytes during lazy initialization.
        verify(paths['yunet'], ASSETS['yunet'])
        self.ort_detector = None
        if config.data['models']['device'] != 'cpu':
            if runtime_info is None:
                from .runtime import runtime
                runtime_info = runtime(config)
            from .onnx_models import OnnxYuNet
            self.ort_detector = OnnxYuNet(config, paths['yunet'], cv2, runtime_info)
            self.detector = None
            self.detector_device = self.ort_detector.device
        else:
            import numpy as np
            blob = verified_bytes(paths['yunet'], ASSETS['yunet'])
            self.detector = cv2.FaceDetectorYN.create('onnx', np.frombuffer(blob, dtype=np.uint8),
                                                     np.empty(0, dtype=np.uint8), (320, 320),
                                                     settings['score_threshold'], settings['nms_threshold'],
                                                     settings['max_faces_per_frame'])
            self.detector_device = 'cpu'
        self.provenance = {'opencv': cv2.__version__, 'detector_device': self.detector_device}
        if self.ort_detector is not None:
            self.provenance['detector'] = self.ort_detector.provenance

    def _bgr(self, image):
        prepared = getattr(self, '_prepared_frame', None)
        if prepared is not None and prepared[0] is image:
            return prepared[1]
        import numpy as np
        with image.convert('RGB') as converted:
            return np.ascontiguousarray(np.asarray(converted)[:, :, ::-1])

    def prepare_frame(self, image):
        previous = getattr(self, '_prepared_frame', None)
        self._prepared_frame = image, self._bgr(image)
        return previous

    def release_frame(self, previous):
        self._prepared_frame = previous

    @contextmanager
    def prepared_frame(self, image):
        previous = self.prepare_frame(image)
        try:
            yield
        finally:
            from stash_ai.runtime.cleanup import cleanup
            cleanup([('release prepared recognition frame', lambda: self.release_frame(previous))])

    def detect(self, image):
        data = self._bgr(image)
        if self.ort_detector is not None:
            return self.ort_detector.detect(data)
        self.detector.setInputSize((image.width, image.height))
        _, rows = self.detector.detect(data)
        if rows is None:
            return []
        if len(rows) > self.config.data['detection']['max_faces_per_frame']:
            raise ValueError('Face detector exceeded configured result count.')
        return [checked_detection(row) for row in rows]

    def close(self):
        self._prepared_frame = None
        self.detector = None
        if getattr(self, 'ort_detector', None) is not None:
            self.ort_detector.close()
            self.ort_detector = None


class OpenCVFaces(YuNetDetector):
    """YuNet/SFace: CPU OpenCV or shared ORT GPU neural inference."""
    def __init__(self, config, paths):
        from .runtime import runtime
        info = runtime(config)
        self.ort_recognizer = None
        try:
            super().__init__(config, paths, runtime_info=info)
            # alignCrop is kept identical on both devices; this OpenCV object
            # never runs neural feature() in GPU/auto mode.
            import numpy as np
            blob = verified_bytes(paths['sface'], ASSETS['sface'])
            self.recognizer = self.cv.FaceRecognizerSF.create('onnx', np.frombuffer(blob, dtype=np.uint8),
                                                            np.empty(0, dtype=np.uint8))
            if config.data['models']['device'] != 'cpu':
                from .onnx_models import OnnxSFace
                self.ort_recognizer = OnnxSFace(config, paths['sface'], info)
                self.device = self.ort_recognizer.device
                self.provenance['recognizer'] = self.ort_recognizer.provenance
            self.dimensions = ASSETS['sface'].dimensions
        except BaseException:
            self.close()
            raise

    def embed(self, image, detection):
        import numpy as np
        aligned = self.recognizer.alignCrop(self._bgr(image), np.asarray(detection.row, dtype=np.float32))
        if self.ort_recognizer is not None:
            return self.ort_recognizer.embed(aligned)
        return normalized_embedding(self.recognizer.feature(aligned), self.dimensions)

    def close(self):
        super().close()
        self.recognizer = None
        if getattr(self, 'ort_recognizer', None) is not None:
            self.ort_recognizer.close()
            self.ort_recognizer = None


def load_backend(config, paths):
    if config.data['models']['recognizer'] == 'auraface':
        from .aura import AuraFaces
        return AuraFaces(config, paths)
    return OpenCVFaces(config, paths)
