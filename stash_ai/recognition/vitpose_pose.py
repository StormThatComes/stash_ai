"""Optional top-down COCO17 refinement of YOLO person boxes with ViTPose++ Large."""
import json

from . import vitpose_registry as registry


def assets(config):
    from .sapiensid_assets import _require_download
    from .sapiensid_downloads import install
    from .sapiensid_loading import local_path
    from .sapiensid_v1 import verify_checkpoint
    value = config.data['sapiensid']['pose']['vitpose_directory']
    directory = (local_path(config, value) if value else
                 config.model_dir / 'vitpose_plus_large' / registry.REVISION)
    for name, (size, checksum) in registry.ASSETS.items():
        path = directory / name
        if not path.exists():
            _require_download(config, path)
            install(registry.BASE_URL + name, path, size, checksum)
        # Verify existing files too; never silently replace a different model.
        verify_checkpoint(path, expected_size=size, expected_sha256=checksum)
    return directory


def load(config, device):
    # Transformers imports these conditionally, then uses them unconditionally
    # during affine crops/heatmap decoding. Check before downloading any assets.
    try:
        import scipy.linalg
        import scipy.ndimage
    except ImportError as error:
        raise RuntimeError('ViTPose++ Large requires SciPy for image preprocessing and pose decoding; '
                           'install requirements-sapiensid.txt in the CPU/GPU environment.') from error
    try:
        from transformers import VitPoseConfig, VitPoseImageProcessor, VitPoseForPoseEstimation
    except ImportError as error:
        raise RuntimeError('ViTPose++ Large requires transformers>=4.49; '
                           'install requirements-sapiensid.txt in the CPU/GPU environment.') from error
    from .sapiensid_loading import checked_bytes
    directory = assets(config)
    def document(name):
        return json.loads(checked_bytes(directory / name, registry.ASSETS[name][1], limit=131072))
    processor = VitPoseImageProcessor.from_dict(document('preprocessor_config.json'))
    model_config = VitPoseConfig.from_dict(document('config.json'))
    model, loading = VitPoseForPoseEstimation.from_pretrained(
        str(directory), config=model_config, local_files_only=True,
        use_safetensors=True, trust_remote_code=False, output_loading_info=True)
    if any(loading.get(key) for key in ('missing_keys', 'unexpected_keys', 'mismatched_keys', 'error_msgs')):
        raise ValueError('ViTPose++ checkpoint does not match the Transformers architecture; '
                         'install requirements-sapiensid.txt in the CPU/GPU environment.')
    model = model.float().to(device).eval()
    return ViTPoseLarge(model, processor, device=device,
                        batch_size=config.data['sapiensid']['pose']['batch_size'])


class ViTPoseLarge:
    """Preserve box order and image coordinates; bound work by people per batch."""

    def __init__(self, model, processor, *, device, batch_size):
        self.model, self.processor = model, processor
        self.device, self.batch_size = device, batch_size

    def predict(self, image, bounds):
        import numpy as np
        import torch
        count = len(bounds)
        points = np.empty((count, 17, 2), dtype=np.float32)
        scores = np.empty((count, 17), dtype=np.float32)
        if not count:
            return points, scores
        boxes = np.asarray(bounds, dtype=np.float32).copy()
        if (boxes.shape != (count, 4) or not np.isfinite(boxes).all()
                or np.any(boxes[:, 2:] <= boxes[:, :2])):
            raise ValueError('ViTPose requires finite, non-empty person boxes.')
        boxes[:, 2:] -= boxes[:, :2]  # HF processor expects COCO xywh, not xyxy.
        with torch.inference_mode():
            for start in range(0, count, self.batch_size):
                batch = boxes[start:start + self.batch_size]
                inputs = self.processor(images=image, boxes=[batch], return_tensors='pt').to(self.device)
                # One COCO expert index per person crop, including the final partial batch.
                dataset_index = torch.zeros(len(batch), dtype=torch.long, device=self.device)
                outputs = self.model(**inputs, dataset_index=dataset_index)
                # No threshold here: retain all 17 labels and filter once in the shared adapter.
                result = self.processor.post_process_pose_estimation(outputs, boxes=[batch], threshold=None)
                if len(result) != 1 or len(result[0]) != len(batch):
                    raise ValueError('ViTPose returned an invalid person count.')
                for offset, person in enumerate(result[0]):
                    xy = person['keypoints'].detach().cpu().numpy()
                    confidence = person['scores'].detach().cpu().numpy()
                    labels = person['labels'].detach().cpu().numpy()
                    if (xy.shape != (17, 2) or confidence.shape != (17,) or labels.shape != (17,)
                            or not np.array_equal(np.sort(labels), np.arange(17))
                            or not np.isfinite(xy).all() or not np.isfinite(confidence).all()):
                        raise ValueError('ViTPose returned malformed/non-finite COCO17 geometry.')
                    order = np.argsort(labels)
                    points[start + offset] = xy[order]
                    # Heatmap maxima are not probabilities and may exceed 1 or be negative.
                    scores[start + offset] = np.clip(confidence[order], 0., 1.)
                del inputs, outputs, result
        return points, scores
