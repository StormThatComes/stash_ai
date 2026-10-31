"""D-FINE person boxes for top-down pose, without loading Ultralytics or YOLO."""
from contextlib import ExitStack
import json
from types import SimpleNamespace

from . import dfine_registry as registry


def assets(config):
    from .sapiensid_assets import _require_download
    from .sapiensid_downloads import install
    from .sapiensid_loading import local_path
    from .sapiensid_v1 import verify_checkpoint
    settings = config.data['sapiensid']['pose']
    model = registry.MODELS[settings['detector']]
    directory = (local_path(config, settings['dfine_directory']) if settings['dfine_directory'] else
                 config.model_dir / settings['detector'] / model['revision'])
    for name, (size, checksum) in model['assets'].items():
        path = directory / name
        if not path.exists():
            _require_download(config, path)
            install(f"https://huggingface.co/{model['repository']}/resolve/{model['revision']}/{name}",
                    path, size, checksum)
        verify_checkpoint(path, expected_size=size, expected_sha256=checksum)
    return directory


def load(config):
    try:
        from transformers import DFineConfig, DFineForObjectDetection, RTDetrImageProcessor
        from torchvision.ops import nms
    except ImportError as error:
        raise RuntimeError('D-FINE requires transformers>=4.53 and torchvision; '
                           'install requirements-sapiensid.txt in the CPU/GPU environment.') from error
    from .sapiensid_inference import resolve_device
    from .sapiensid_loading import checked_bytes
    settings = config.data['sapiensid']['pose']
    model_assets = registry.MODELS[settings['detector']]['assets']
    directory = assets(config)
    def document(name):
        return json.loads(checked_bytes(directory / name, model_assets[name][1], limit=131072))
    processor = RTDetrImageProcessor.from_dict(document('preprocessor_config.json'))
    architecture = document('config.json')
    model_config = DFineConfig.from_dict(architecture)
    if (model_config.num_labels != 80 or model_config.id2label.get(registry.PERSON_LABEL) != 'person'
            or not model_config.use_focal_loss or architecture.get('use_pretrained_backbone', False)):
        raise ValueError('D-FINE requires the pinned COCO80 architecture and person label.')
    model, loading = DFineForObjectDetection.from_pretrained(
        str(directory), config=model_config, local_files_only=True, use_safetensors=True,
        trust_remote_code=False, output_loading_info=True)
    if any(loading.get(key) for key in ('missing_keys', 'unexpected_keys', 'mismatched_keys', 'error_msgs')):
        raise ValueError('D-FINE checkpoint does not match the Transformers architecture; '
                         'install requirements-sapiensid.txt in the CPU/GPU environment.')
    device = resolve_device(config.data['models']['device'])
    return DFineDetector(model.float().to(device).eval(), processor, device=device,
                         batch_size=settings['batch_size'])


class DFineDetector:
    """Minimal shared box-result interface; output tensors remain on the CPU."""
    task = 'detect'

    def __init__(self, model, processor, *, device, batch_size):
        self.model, self.processor = model, processor
        self.device, self.batch_size = device, batch_size

    def predict(self, source, *, conf, iou, max_det, device=None, verbose=False, nms=True):
        import torch
        images = source if isinstance(source, list) else [source]
        result = []
        with torch.inference_mode():
            for start in range(0, len(images), self.batch_size):
                batch = images[start:start+self.batch_size]
                with ExitStack() as stack:
                    rgb = [image if image.mode == 'RGB' else stack.enter_context(image.convert('RGB'))
                           for image in batch]
                    inputs = self.processor(images=rgb, return_tensors='pt').to(self.device)
                    outputs = self.model(**inputs)
                result.extend(self._decode(outputs, [image.size for image in batch], conf, iou, max_det, nms))
                del inputs, outputs
        return result

    @staticmethod
    def _decode(outputs, sizes, threshold, iou, maximum, suppress):
        import torch
        from torchvision.ops import nms
        logits = outputs.logits.detach().float().cpu()
        boxes = outputs.pred_boxes.detach().float().cpu()
        if (logits.ndim != 3 or logits.shape[0] != len(sizes) or logits.shape[2] != 80
                or boxes.shape != (*logits.shape[:2], 4)
                or not torch.isfinite(logits).all() or not torch.isfinite(boxes).all()
                or (boxes[..., 2:] < 0).any()):
            raise ValueError('D-FINE returned malformed/non-finite person detections.')
        # The pinned models use independent sigmoid class scores. Select people
        # before limiting results so other COCO classes cannot consume the budget.
        scores = logits[..., registry.PERSON_LABEL].sigmoid()
        corners = torch.cat((boxes[..., :2] - boxes[..., 2:] / 2,
                             boxes[..., :2] + boxes[..., 2:] / 2), dim=-1)
        result = []
        for raw, confidence, (width, height) in zip(corners, scores, sizes):
            box = raw * torch.tensor([width, height, width, height])
            box[:, 0::2].clamp_(0, width)
            box[:, 1::2].clamp_(0, height)
            keep = (confidence > threshold) & (box[:, 2:] > box[:, :2]).all(dim=1)
            box, confidence = box[keep], confidence[keep]
            order = torch.argsort(confidence, descending=True, stable=True)
            box, confidence = box[order], confidence[order]
            if suppress:
                selected = nms(box, confidence, iou)[:maximum]
            else:
                selected = slice(None, maximum)
            result.append(SimpleNamespace(boxes=SimpleNamespace(xyxy=box[selected], conf=confidence[selected])))
        return result
