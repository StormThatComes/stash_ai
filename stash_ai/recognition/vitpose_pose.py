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
        return self.predict_many([(image, bounds)])[0]

    def predict_many(self, frames):
        """Batch people across frames without changing their image coordinates."""
        import numpy as np
        import torch
        result, jobs = [], []
        for frame, (image, bounds) in enumerate(frames):
            count = len(bounds)
            result.append((np.empty((count,17,2), dtype=np.float32), np.empty((count,17), dtype=np.float32)))
            if not count:continue
            boxes = np.asarray(bounds,dtype=np.float32).copy()
            if boxes.shape != (count,4) or not np.isfinite(boxes).all() or np.any(boxes[:,2:]<=boxes[:,:2]):
                raise ValueError('ViTPose requires finite, non-empty person boxes.')
            boxes[:,2:] -= boxes[:,:2]
            jobs.extend((frame, index, box) for index,box in enumerate(boxes))
        with torch.inference_mode():
            for start in range(0,len(jobs),self.batch_size):
                batch = jobs[start:start+self.batch_size]
                owners = list(dict.fromkeys(frame for frame,_,_ in batch))
                images = [frames[frame][0] for frame in owners]
                boxes = [np.asarray([box for frame,_,box in batch if frame==owner]) for owner in owners]
                inputs = self.processor(images=images[0] if len(images)==1 else images,
                    boxes=boxes,return_tensors='pt').to(self.device)
                dataset_index = torch.zeros(len(batch),dtype=torch.long,device=self.device)
                outputs = self.model(**inputs,dataset_index=dataset_index)
                decoded = self.processor.post_process_pose_estimation(outputs,boxes=boxes,threshold=None)
                if len(decoded)!=len(owners) or any(len(rows)!=len(b) for rows,b in zip(decoded,boxes)):
                    raise ValueError('ViTPose returned an invalid person count.')
                people = [person for rows in decoded for person in rows]
                # One transfer per field for the entire people batch.
                xy = torch.stack([p['keypoints'] for p in people]).detach().cpu().numpy()
                confidence = torch.stack([p['scores'] for p in people]).detach().cpu().numpy()
                labels = torch.stack([p['labels'] for p in people]).detach().cpu().numpy()
                if (xy.shape!=(len(batch),17,2) or confidence.shape!=(len(batch),17) or labels.shape!=(len(batch),17)
                        or not np.all(np.sort(labels,axis=1)==np.arange(17))
                        or not np.isfinite(xy).all() or not np.isfinite(confidence).all()):
                    raise ValueError('ViTPose returned malformed/non-finite COCO17 geometry.')
                for offset,(frame,index,_) in enumerate(batch):
                    order = np.argsort(labels[offset])
                    result[frame][0][index] = xy[offset,order]
                    result[frame][1][index] = np.clip(confidence[offset,order],0.,1.)
                del inputs, outputs, decoded
        return result
