"""Pinned AuraFace recognition only; no InsightFace model packs or attributes."""
from .models import ASSETS, YuNetDetector, normalized_embedding, verified_bytes
from .runtime import runtime, session

CONTRACT = {'format': 1, 'alignment': 'five-point-similarity-arcface-112',
            'color': 'RGB', 'input': 'float32 NCHW 1x3x112x112',
            'normalization': '(pixel-127.5)/127.5', 'output': 'unit-float32-512',
            'license_source': 'fal/AuraFace-v1/LICENSE.md@af6d057c9b0ec4071d4c49c80e3539258798b609'}


def similarity_matrix(landmarks):
    """Least-squares 2D similarity (five points); no additional alignment package."""
    import numpy as np
    source = np.asarray(landmarks, dtype=np.float64)
    target = np.asarray([[38.2946, 51.6963], [73.5318, 51.5014], [56.0252, 71.7366],
                         [41.5493, 92.3655], [70.7299, 92.2041]], dtype=np.float64)
    if source.shape != (5, 2) or not np.isfinite(source).all():
        raise ValueError('AuraFace alignment requires five finite 2D landmarks.')
    mean_source, mean_target = source.mean(axis=0), target.mean(axis=0)
    centered = source - mean_source
    variance = float((centered * centered).sum() / 5)
    if variance <= 1e-8 or np.linalg.matrix_rank(centered) < 2:
        raise ValueError('AuraFace alignment landmarks are degenerate.')
    covariance = (target - mean_target).T @ centered / 5
    u, singular, vt = np.linalg.svd(covariance)
    signs = np.ones(2)
    if np.linalg.det(u) * np.linalg.det(vt) < 0:
        signs[-1] = -1
    rotation = (u * signs) @ vt
    scale = float(np.dot(singular, signs) / variance)
    matrix = np.column_stack((scale * rotation, mean_target - scale * rotation @ mean_source))
    if scale <= 0 or not np.isfinite(matrix).all():
        raise ValueError('AuraFace alignment transform is invalid.')
    return matrix.astype(np.float32)


def input_blob(aligned_bgr):
    import numpy as np
    pixels = np.asarray(aligned_bgr)
    if pixels.shape != (112, 112, 3) or pixels.dtype != np.uint8:
        raise ValueError('AuraFace requires a real 112x112 uint8 BGR aligned crop.')
    rgb = pixels[:, :, ::-1].astype(np.float32)
    return np.ascontiguousarray(((rgb - 127.5) / 127.5).transpose(2, 0, 1)[None])


class AuraFaces(YuNetDetector):
    def __init__(self, config, paths):
        ort, providers = runtime(config)
        self.session = None
        try:
            super().__init__(config, paths, runtime_info=(ort, providers))
            self.session, self.device = session(config, verified_bytes(paths['auraface'], ASSETS['auraface']), runtime_info=(ort, providers))
        except BaseException:
            self.close()
            raise
        inputs, outputs = self.session.get_inputs(), self.session.get_outputs()
        if (len(inputs) != 1 or len(outputs) != 1 or inputs[0].type != 'tensor(float)' or
                outputs[0].type != 'tensor(float)' or len(inputs[0].shape) != 4 or
                inputs[0].shape[1:] != [3, 112, 112] or len(outputs[0].shape) != 2 or
                outputs[0].shape[-1] != 512):
            self.close()
            raise ValueError('Pinned AuraFace ONNX input/output contract did not match.')
        self.input_name, self.output_name = inputs[0].name, outputs[0].name
        actual = self.session.get_providers()
        self.device = 'cuda' if actual[0] == 'CUDAExecutionProvider' else 'cpu'
        if config.data['models']['device'] == 'cuda' and self.device != 'cuda':
            self.close()
            raise ValueError('AuraFace CUDA initialization fell back to CPU; requested device was not met.')
        self.dimensions = ASSETS['auraface'].dimensions
        self.provenance.update({'onnxruntime': ort.__version__, 'providers': actual, 'preprocessing': CONTRACT})

    def prepare_embedding(self, image, detection):
        transform = similarity_matrix(detection.landmarks)
        aligned = self.cv.warpAffine(self._bgr(image), transform, (112, 112), borderValue=0.)
        return input_blob(aligned)

    def embed_prepared_many(self, tensors):
        from .embedding_batches import infer
        return infer(self, tensors)

    def embed(self, image, detection):
        return self.embed_prepared_many([self.prepare_embedding(image, detection)])[0]

    def close(self):
        super().close()
        self.session = None
