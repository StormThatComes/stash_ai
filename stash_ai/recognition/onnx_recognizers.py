"""Pinned ONNX face embeddings for LVFace-B and AdaFace IR50.

Both use YuNet five-landmark ArcFace 112px alignment; their pixel channel
orders differ. No PyTorch or InsightFace runtime is needed.
"""
from .aura import similarity_matrix
from .models import ASSETS, YuNetDetector, normalized_embedding, verified_bytes
from .runtime import runtime, session


CONTRACTS = {
    'lvface_b': {
        'alignment': 'five-point-similarity-arcface-112',
        'color': 'RGB',
        'normalization': '(pixel-127.5)/127.5',
        'input': 'float32 NCHW (1,3,112,112)',
        'output': 'unit-float32-512',
        'reference': 'https://github.com/bytedance/LVFace/blob/main/inference_onnx.py',
        'weights_terms': 'LVFace pretrained weights: non-commercial research only (upstream README).',
    },
    'adaface_ir50': {
        'alignment': 'five-point-similarity-arcface-112',
        'color': 'BGR',
        'normalization': '(pixel-127.5)/127.5',
        'input': 'float32 NCHW (1,3,112,112)',
        'output': 'unit-float32-512',
        'reference': 'https://huggingface.co/globalnebula/adaface-ir50-ms1mv2-onnx',
        'weights_terms': 'AdaFace MS1MV2 weights: dataset/research-use licensing; verify before commercial use.',
    },
}


def input_blob(aligned_bgr, *, recognizer):
    """Use model-specific channel order; normalize *once*, after alignment."""
    import numpy as np

    if recognizer not in CONTRACTS:
        raise ValueError('Unknown ONNX face recognizer.')
    pixels = np.asarray(aligned_bgr)
    if pixels.shape != (112, 112, 3) or pixels.dtype != np.uint8:
        raise ValueError('Face recognizer requires a 112x112 uint8 BGR aligned crop.')
    if CONTRACTS[recognizer]['color'] == 'RGB':
        pixels = pixels[:, :, ::-1]
    tensor = (pixels.astype(np.float32) - 127.5) / 127.5
    return np.ascontiguousarray(tensor.transpose(2, 0, 1)[None])


class OnnxEmbeddingFaces(YuNetDetector):
    """Share the verified YuNet detector and ORT provider policy."""

    def __init__(self, config, paths):
        key = config.data['models']['recognizer']
        if key not in CONTRACTS:
            raise ValueError('Unsupported ONNX embedding recognizer.')
        self.recognizer_key = key
        self.session = None
        ort, providers = runtime(config)
        try:
            super().__init__(config, paths, runtime_info=(ort, providers))
            model = ASSETS[key]
            self.session, self.device = session(
                config, verified_bytes(paths[key], model), runtime_info=(ort, providers))
            inputs, outputs = self.session.get_inputs(), self.session.get_outputs()
            if (len(inputs) != 1 or len(outputs) != 1
                    or inputs[0].type != 'tensor(float)'
                    or outputs[0].type != 'tensor(float)'
                    or len(inputs[0].shape) != 4
                    or inputs[0].shape[1:] != [3, 112, 112]
                    or len(outputs[0].shape) != 2
                    or outputs[0].shape[-1] != model.dimensions):
                raise ValueError(f'{key}: unexpected ONNX embedding input/output contract.')
            # A dynamic or static single-image batch is acceptable.
            if inputs[0].shape[0] not in (1, None) and isinstance(inputs[0].shape[0], int):
                raise ValueError(f'{key}: ONNX input must support a batch of one.')
            if outputs[0].shape[0] not in (1, None) and isinstance(outputs[0].shape[0], int):
                raise ValueError(f'{key}: ONNX output must support a batch of one.')
            self.input_name, self.output_name = inputs[0].name, outputs[0].name
            actual = self.session.get_providers()
            self.device = 'cuda' if actual and actual[0] == 'CUDAExecutionProvider' else 'cpu'
            if config.data['models']['device'] == 'cuda' and self.device != 'cuda':
                raise ValueError(f'{key}: CUDA initialization fell back to CPU.')
            self.dimensions = model.dimensions
            self.provenance.update({
                'recognizer': key, 'onnxruntime': ort.__version__,
                'providers': actual, 'preprocessing': CONTRACTS[key],
                'weights': {'repo': model.repo, 'revision': model.revision,
                            'filename': model.filename, 'sha256': model.sha256},
            })
        except BaseException:
            self.close()
            raise

    def embed(self, image, detection):
        transform = similarity_matrix(detection.landmarks)
        aligned = self.cv.warpAffine(self._bgr(image), transform, (112, 112), borderValue=0.)
        tensor = input_blob(aligned, recognizer=self.recognizer_key)
        output = self.session.run([self.output_name], {self.input_name: tensor})
        if len(output) != 1 or output[0].shape != (1, self.dimensions):
            raise ValueError(f'{self.recognizer_key}: unexpected ONNX embedding output shape.')
        return normalized_embedding(output[0], self.dimensions)

    def close(self):
        super().close()
        self.session = None
