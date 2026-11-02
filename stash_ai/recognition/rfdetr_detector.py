"""Verified RF-DETR-XL person boxes for the shared top-down pose pipeline."""
from contextlib import ExitStack, contextmanager
from importlib.metadata import version, PackageNotFoundError
import logging
from threading import get_ident
from types import SimpleNamespace

from . import rfdetr_registry as registry


_BACKBONE_NOTICES = frozenset({
    "Using a different number of positional encodings than DINOv2, which means"
    " we're not loading DINOv2 backbone weights. This is not a problem if"
    " finetuning a pretrained RF-DETR model.",
    "Using patch size 20 instead of 14, which means we're not loading"
    " DINOv2 backbone weights. This is not a problem if finetuning a pretrained"
    " RF-DETR model.",
})


@contextmanager
def _backbone_diagnostics():
    """Route only the expected XL construction notices to verbose diagnostics."""
    from stash_ai.runtime.console import current_report
    logger, owner = logging.getLogger('rf-detr'), get_ident()
    def keep(record):
        if (record.thread != owner or record.module != 'dinov2' or record.levelno != logging.WARNING
                or record.exc_info or record.stack_info or record.getMessage() not in _BACKBONE_NOTICES):
            return True
        message = 'RF-DETR-XL initialization: ' + record.getMessage()
        report = current_report()
        if report is not None:
            report.detail(message)
        else:
            logging.getLogger(__name__).debug(message)
        return False
    logger.addFilter(keep)
    try:
        yield
    finally:
        logger.removeFilter(keep)


def assets(config):
    from .sapiensid_assets import _require_download
    from .sapiensid_downloads import install
    from .sapiensid_loading import local_path
    from .sapiensid_v1 import verify_checkpoint
    selected = config.data['sapiensid']['pose'].get('rfdetr_directory', '')
    directory = (local_path(config, selected) if selected else
                 config.model_dir / 'rfdetr_xl' / registry.SHA256)
    path = directory / registry.NAME
    if not path.exists():
        _require_download(config, path)
        install(registry.URL, path, registry.SIZE, registry.SHA256)
    verify_checkpoint(path, expected_size=registry.SIZE, expected_sha256=registry.SHA256)
    return path


def load(config):
    hint = ('RF-DETR-XL requires rfdetr==' + registry.VERSION + ' and rfdetr-plus==' + registry.PLUS_VERSION +
            '; install requirements-rfdetr.txt in the existing CPU/GPU environment.')
    try:
        if version('rfdetr') != registry.VERSION or version('rfdetr-plus') != registry.PLUS_VERSION:
            raise RuntimeError(hint)
        from rfdetr_plus.models.detection import RFDETRXLargeConfig
        from rfdetr.models import build_model_from_config
        import torch
        from torchvision.ops import nms
    except (ImportError, PackageNotFoundError) as error:
        raise RuntimeError(hint) from error
    from .sapiensid_inference import load_verified_weights, resolve_device
    device = resolve_device(config.data['models']['device'])
    path = assets(config)
    # The builder constructs architecture only. A non-null pretrain_weights
    # disables backbone downloads; our verified loader owns all weight I/O.
    architecture = RFDETRXLargeConfig(pretrain_weights=str(path.resolve()), num_classes=90, device='cpu')
    with _backbone_diagnostics():
        model = build_model_from_config(architecture)
    # Recent upstream detection models generate an empty keypoint-schema buffer,
    # absent from the original XL checkpoint. It carries no learned parameters.
    mask = model._kp_active_mask
    if not isinstance(mask, torch.Tensor) or tuple(mask.shape) != (0, 0):
        raise ValueError('RF-DETR-XL requires the pinned detection-only architecture.')
    model.register_buffer('_kp_active_mask', mask, persistent=False)
    load_verified_weights(model, path, expected_size=registry.SIZE, expected_sha256=registry.SHA256)
    return RFDETRDetector(model.float().to(device).eval(), device=device,
                          batch_size=config.data['sapiensid']['pose']['batch_size'])


class RFDETRDetector:
    """Person-only result tensors in original-image coordinates, on the CPU."""
    task = 'detect'

    def __init__(self, model, *, device, batch_size):
        self.model, self.device, self.batch_size = model, device, batch_size

    def predict(self, source, *, conf, iou, max_det, device=None, verbose=False, nms=True):
        import torch
        images = source if isinstance(source, list) else [source]
        results = []
        with torch.inference_mode():
            for start in range(0, len(images), self.batch_size):
                batch = images[start:start+self.batch_size]
                outputs = self._forward(batch)
                results.extend(self._decode(outputs, [image.size for image in batch], conf, iou, max_det,
                                            nms, frame_offset=start))
                del outputs
        return results

    def _forward(self, batch):
        import torch
        from torchvision.transforms import functional as transforms
        with ExitStack() as stack:
            rgb = [image if image.mode == 'RGB' else stack.enter_context(image.convert('RGB')) for image in batch]
            # Match the pinned upstream predict(): RGB /255, bilinear square
            # resize without antialiasing, then ImageNet normalization.
            inputs = torch.stack([transforms.resize(transforms.to_tensor(image).to(self.device),
                [registry.RESOLUTION, registry.RESOLUTION], antialias=False) for image in rgb])
            inputs = transforms.normalize(inputs, registry.MEAN, registry.STD)
            with torch.autocast(device_type=torch.device(self.device).type, enabled=False):
                return self.model(inputs.float())

    @staticmethod
    def _decode(outputs, sizes, threshold, iou, maximum, suppress, *, frame_offset=0):
        import torch
        from torchvision.ops import nms
        logits = outputs['pred_logits'].detach().float().cpu()
        boxes = outputs['pred_boxes'].detach().float().cpu()
        if (logits.ndim != 3 or logits.shape[0] != len(sizes) or logits.shape[2] != 91
                or boxes.shape != (*logits.shape[:2], 4)):
            raise ValueError('RF-DETR-XL returned malformed COCO91 detection shapes: '
                             f'logits={tuple(logits.shape)}, boxes={tuple(boxes.shape)}; '
                             f'input_frame_offset={frame_offset}.')
        for name, values in (('logits', logits), ('pred_boxes', boxes)):
            invalid = ~torch.isfinite(values)
            if invalid.any():
                frame, query, component = invalid.nonzero()[0].tolist()
                raise ValueError(f'RF-DETR-XL returned non-finite {name}; input_frame={frame_offset+frame}, '
                                 f'query={query}, component={component} (zero-based).')
        # Select the person channel before any top-k limit: other classes and
        # the same query's alternative labels cannot consume the people budget.
        scores = logits[..., registry.PERSON_LABEL].sigmoid()
        valid_geometry = (boxes[..., 2:] > 0).all(dim=-1)
        corners = torch.cat((boxes[..., :2] - boxes[..., 2:] / 2,
                             boxes[..., :2] + boxes[..., 2:] / 2), dim=-1)
        results = []
        for raw, confidence, valid, (width, height) in zip(corners, scores, valid_geometry, sizes):
            box = raw * torch.tensor([width, height, width, height])
            box[:, 0::2].clamp_(0, width)
            box[:, 1::2].clamp_(0, height)
            keep = valid & (confidence > threshold) & (box[:, 2:] > box[:, :2]).all(dim=1)
            box, confidence = box[keep], confidence[keep]
            order = torch.argsort(confidence, descending=True, stable=True)
            box, confidence = box[order], confidence[order]
            selected = nms(box, confidence, iou)[:maximum] if suppress else slice(None, maximum)
            results.append(SimpleNamespace(boxes=SimpleNamespace(xyxy=box[selected], conf=confidence[selected])))
        return results
