"""Optional Sapiens v1 pose, using verified public Meta TorchScript exports."""
from . import sapiens_pose_registry as registry


def asset(config):
    from .sapiensid_assets import _require_download
    from .sapiensid_downloads import install
    from .sapiensid_loading import local_path
    from .sapiensid_v1 import verify_checkpoint
    settings = config.data['sapiensid']['pose']
    model = registry.MODELS[settings['sapiens_variant']]
    value = settings['sapiens_directory']
    directory = (local_path(config, value) if value else config.model_dir / 'sapiens_pose' /
                 settings['sapiens_variant'] / model['revision'])
    path = directory / model['filename']
    if not path.exists():
        _require_download(config, path)
        url = f"https://huggingface.co/{model['repository']}/resolve/{model['revision']}/{model['filename']}"
        install(url, path, model['size'], model['sha256'])
    return verify_checkpoint(path, expected_size=model['size'], expected_sha256=model['sha256'])


def load(config, device):
    try:
        import cv2
        import torch
    except ImportError as error:
        raise RuntimeError('Sapiens Pose requires PyTorch and OpenCV; install requirements-sapiensid.txt '
                           'alongside the CPU/GPU requirements.') from error
    # TorchScript is executable model code: only the exact pinned, hash-checked
    # upstream export is admitted. Remap saved CUDA tensors before moving it.
    model = torch.jit.load(str(asset(config)), map_location='cpu').float().eval().to(device)
    return SapiensPose(model, device=device,
        batch_size=config.data['sapiensid']['pose']['sapiens_batch_size'])


class SapiensPose:
    def __init__(self, model, *, device, batch_size):
        self.model, self.device, self.batch_size = model, device, batch_size

    def predict(self, image, bounds):
        return self.predict_many([(image, bounds)])[0]

    def predict_many(self, frames):
        """Bound crop/heatmap memory by people per batch, including across cues."""
        import numpy as np
        import torch
        from .sapiens_pose_geometry import prepare, decode
        from .profiling import count
        frames = list(frames)
        result, jobs = [], []
        for frame, (image, bounds) in enumerate(frames):
            result.append((np.empty((len(bounds), 17, 2), dtype=np.float32),
                           np.empty((len(bounds), 17), dtype=np.float32)))
            if not len(bounds):
                continue
            boxes = np.asarray(bounds, dtype=np.float32)
            if (boxes.shape != (len(bounds), 4) or not np.isfinite(boxes).all()
                    or np.any(boxes[:, 2:] <= boxes[:, :2])):
                raise ValueError('Sapiens Pose requires finite, non-empty person boxes.')
            jobs.extend((frame, index, box) for index, box in enumerate(boxes))
        with torch.inference_mode():
            for start in range(0, len(jobs), self.batch_size):
                batch = jobs[start:start + self.batch_size]
                crops = [prepare(frames[frame][0], box) for frame, _, box in batch]
                inputs = torch.from_numpy(np.stack([crop for crop, _, _ in crops])).to(self.device)
                count('sapiens_pose_batches')
                count('sapiens_pose_people', len(batch))
                output = self.model(inputs)
                width, height = registry.HEATMAP_SIZE
                if not isinstance(output, torch.Tensor) or output.shape != (len(batch), 308, height, width):
                    raise ValueError('Sapiens Pose returned an invalid Goliath heatmap shape.')
                # Transfer/decode only the 17 required channels, in COCO order.
                heatmaps = output[:, registry.COCO17_INDICES].detach().float().cpu().numpy()
                del inputs, output
                for offset, (frame, index, _) in enumerate(batch):
                    points, scores = decode(heatmaps[offset], *crops[offset][1:])
                    result[frame][0][index] = points
                    result[frame][1][index] = scores
                del heatmaps, crops
        return result
