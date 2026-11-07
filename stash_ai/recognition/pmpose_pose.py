"""Optional bbox-only PMPose-B adapter with separate OKS/presence/visibility."""
from . import pmpose_registry as registry


def assets(config):
    from .sapiensid_assets import _require_download
    from .sapiensid_downloads import install
    from .sapiensid_loading import local_path
    value = config.data['sapiensid']['pose']['pmpose_directory']
    directory = local_path(config, value) if value else config.model_dir / 'pmpose_b' / registry.REVISION
    path = directory / registry.NAME
    if not path.exists():
        _require_download(config, path)
        install(registry.URL, path, registry.SIZE, registry.SHA256)
    return path  # Hash and deserialize the same descriptor in load_verified_weights.


class _TrainingHistory:
    """Inert container for checkpoint metadata; no upstream code is imported."""


def _history_attribute(obj, name):
    # The pinned checkpoint pickles four HistoryBuffer methods in training
    # metadata. Never allow unrestricted getattr during deserialization.
    if obj is not _TrainingHistory or name not in ('min', 'max', 'current', 'mean'):
        raise ValueError('Unexpected PMPose training metadata attribute.')
    return None


def load(config, device):
    try:
        import cv2
        import numpy as np
        import torch
    except ImportError as error:
        raise RuntimeError('PMPose requires PyTorch, NumPy and OpenCV in the CPU/GPU environment.') from error
    from .pmpose_network import PMPoseB
    from .sapiensid_inference import load_verified_weights
    path = assets(config)
    model = PMPoseB()
    # Restricted weights-only loading of this hash-pinned training checkpoint.
    # NumPy arrays and inert HistoryBuffer metadata are discarded by the loader.
    from numpy._core import multiarray
    allowed = [(multiarray._reconstruct, 'numpy.core.multiarray._reconstruct'),
               (multiarray.scalar, 'numpy.core.multiarray.scalar'), np.ndarray, np.dtype,
               *(type(np.dtype(t)) for t in ('float32', 'float64', 'uint8', 'int64')),
               (_TrainingHistory, 'mmengine.logging.history_buffer.HistoryBuffer'),
               (_history_attribute, 'builtins.getattr')]
    with torch.serialization.safe_globals(allowed):
        load_verified_weights(model, path, expected_size=registry.SIZE, expected_sha256=registry.SHA256)
    model = model.float().to(device).eval()
    pose = PMPose(model, device=device, batch_size=config.data['sapiensid']['pose']['pmpose_batch_size'])
    return _pipeline(pose, config, device)


def _pipeline(pose, config, device):
    """Assemble optional stages around one shared first-pass pose estimator."""
    from .cached_pose import RawPMPose
    settings = config.data['sapiensid']['pose'].get('refinement', {})
    from .inference_cache import current
    if current() is not None:
        pose=RawPMPose(pose,device)
    refined = None
    if settings.get('enabled', False):
        from .sam_pose2seg import load as load_segmenter
        from .pose_refinement import RefinedPMPose
        refined = RefinedPMPose(pose, load_segmenter(config, device), settings)
    deduplication = config.data['sapiensid']['pose'].get('deduplication', {})
    if deduplication.get('enabled', False):
        from .pose_deduplication import DeduplicatedPMPose
        return DeduplicatedPMPose(pose, deduplication, refined)
    return refined or pose


class PMPose:
    def __init__(self, model, *, device, batch_size):
        self.model, self.device, self.batch_size = model, device, batch_size

    def resident_modules(self):return [self.model]

    def predict(self, image, bounds, masks=None):
        return self.predict_many([(image, bounds, masks)])[0]

    def predict_many(self, frames):
        import numpy as np
        import torch
        from .pmpose_geometry import prepare, decode
        frames = list(frames)
        result, jobs = [], []
        for frame, item in enumerate(frames):
            image, bounds = item[:2]
            masks = item[2] if len(item) == 3 else None
            if masks is not None and len(masks) != len(bounds):
                raise ValueError('PMPose requires exactly one mask per person box.')
            count = len(bounds)
            result.append((np.empty((count, 17, 2), dtype=np.float32),
                           np.empty((count, 17), dtype=np.float32),
                           {key: np.empty((count, 17), dtype=np.float32) for key in ('presence', 'visibility')}))
            if not count:
                continue
            boxes = np.asarray(bounds, dtype=np.float32)
            if boxes.shape != (count, 4) or not np.isfinite(boxes).all() or np.any(boxes[:, 2:] <= boxes[:, :2]):
                raise ValueError('PMPose requires finite, non-empty person boxes.')
            jobs.extend((frame, index, box, None if masks is None else masks[index])
                        for index, box in enumerate(boxes))
        with torch.inference_mode():
            for start in range(0, len(jobs), self.batch_size):
                batch = jobs[start:start + self.batch_size]
                crops = [prepare(frames[frame][0], box, mask) for frame, _, box, mask in batch]
                inputs = torch.from_numpy(np.stack([crop for crop, _, _ in crops])).to(self.device)
                output = self._forward(inputs)
                flipped = self._forward(inputs.flip(-1))
                values = [(a + (b[:, registry.FLIP].flip(-1) if i == 0 else b[:, registry.FLIP])) * .5
                          for i, (a, b) in enumerate(zip(output, flipped))]
                heatmaps, scores, presence, visibility = [v.cpu().numpy() for v in values]
                for offset, (frame, index, _, _) in enumerate(batch):
                    _, center, scale = crops[offset]
                    result[frame][0][index] = decode(heatmaps[offset], center, scale)
                    result[frame][1][index] = scores[offset]
                    result[frame][2]['presence'][index] = presence[offset]
                    result[frame][2]['visibility'][index] = visibility[offset]
        return result

    def _forward(self, inputs):
        import torch
        output = self.model(inputs)
        shapes = ((len(inputs), 17, 64, 48),) + ((len(inputs), 17),) * 3
        if (not isinstance(output, (tuple, list)) or len(output) != 4 or
                any(not isinstance(v, torch.Tensor) or tuple(v.shape) != shape or
                    not torch.isfinite(v).all() or torch.any((v < 0) | (v > 1))
                    for v, shape in zip(output, shapes))):
            raise ValueError('PMPose returned malformed/non-finite heatmaps or probabilities.')
        return output
