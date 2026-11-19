"""Person instance masks from RF-DETR-Seg Large, associated by pose prompts."""
from importlib.metadata import version
from .mask_backends import FrameSegmenter

VERSION = '1.11.2'
NAME = 'rf-detr-seg-large.pt'
URL = 'https://repo.roboflow.com/rfdetr/rf-detr-seg-l-ft.pth'
SIZE = 145055866
SHA256 = 'ca7b7c630ba22496067cc4f034c4e70c8e47fd7adcea04c3a49aa8c1755cbe6b'


def load(config, device):
    try:
        from rfdetr.config import RFDETRSegLargeConfig
        from rfdetr.models import build_model_from_config
        if version('rfdetr') != VERSION:
            raise ImportError('Unsupported rfdetr version')
    except ImportError as error:
        raise RuntimeError('RF-DETR-Seg requires rfdetr==' + VERSION + '; install the CPU/GPU requirements.') from error
    from .mask_backends import checkpoint
    from .sapiensid_inference import load_verified_weights
    path = checkpoint(config, 'rfdetr_seg_large', SHA256, NAME, URL, SIZE, SHA256)
    # Published Large weights contain 300 queries (3900 with group_detr=13),
    # while rfdetr 1.11.2's constructor defaults to 200. Match the checkpoint.
    architecture = RFDETRSegLargeConfig(pretrain_weights=str(path.resolve()), num_classes=90, num_queries=300, num_select=300, device='cpu')
    model = build_model_from_config(architecture)
    # Detection/segmentation checkpoints predate the optional keypoint buffer.
    if hasattr(model, '_kp_active_mask'):
        if tuple(model._kp_active_mask.shape) != (0, 0):
            raise ValueError('RF-DETR-Seg requires the COCO segmentation architecture.')
        model.register_buffer('_kp_active_mask', model._kp_active_mask, persistent=False)
    load_verified_weights(model, path, expected_size=SIZE, expected_sha256=SHA256)
    return RFDETRMasks(model.float().to(device).eval(), device, config.data['execution']['sam_precision'],
                       config.data['sapiensid']['pose']['refinement']['min_mask_score'],
                       settings=config.data['sapiensid']['pose']['refinement'])


class RFDETRMasks(FrameSegmenter):
    def __init__(self, model, device, precision, threshold, *, settings=None):
        super().__init__(model, device, precision)
        from .pose_refinement_config import defaults
        self.settings = dict(defaults() if settings is None else settings, min_mask_score=threshold)
        self.threshold = threshold
        self.masks = self.scores = self.instance_ids = None

    def clear(self):
        super().clear()
        self.masks = self.scores = self.instance_ids = None

    def encode(self, image):
        import numpy as np
        import torch
        import torch.nn.functional as F
        from torchvision.transforms import functional as T
        with image.convert('RGB') as rgb:
            pixels = T.to_tensor(rgb).to(self.device)
        pixels = T.resize(pixels, [504, 504], antialias=False)
        pixels = T.normalize(pixels, [.485, .456, .406], [.229, .224, .225])
        output = self.model(pixels[None])
        logits, masks = output['pred_logits'], output['pred_masks']
        if (logits.ndim != 3 or logits.shape[0] != 1 or logits.shape[2] != 91
                or masks.ndim != 4 or masks.shape[:2] != logits.shape[:2]
                or not torch.isfinite(logits).all() or not torch.isfinite(masks).all()):
            raise ValueError('RF-DETR-Seg returned malformed masks or class scores.')
        scores = logits[0, :, 1].float().sigmoid()  # COCO sparse class ID 1: person
        indices = torch.nonzero(scores >= self.threshold).flatten()
        indices = indices[torch.argsort(scores[indices], descending=True, stable=True)]
        self.instance_ids = indices.cpu().tolist()
        self.scores = scores[indices].cpu().numpy()
        # Keep packed masks on the CPU, resize one plane at a time to avoid an
        # N-person full-resolution floating point allocation on the GPU.
        self.masks = []
        for index in indices:
            mask = F.interpolate(masks[0, index][None, None].float(), size=self.size[::-1],
                                 mode='bilinear', align_corners=False)[0, 0] > 0
            self.masks.append(np.packbits(mask.cpu().numpy(), axis=None))

    def iter_instances(self):
        if self.masks is None:
            raise RuntimeError('RF-DETR-Seg requires an active image context.')
        ids = self.instance_ids if self.instance_ids is not None else range(len(self.masks))
        for index, instance in enumerate(ids):
            yield instance, self._instance(index)

    def _instance(self, index):
        import numpy as np
        mask = np.unpackbits(self.masks[index], count=self.size[0]*self.size[1]).reshape(self.size[::-1])
        return self.checked(mask[None], [self.scores[index]])[0]

    def predict_instance(self, instance):
        """Replay the selected query, independent of confidence/ranking order."""
        if self.masks is None:
            raise RuntimeError('RF-DETR-Seg requires an active image context.')
        ids = self.instance_ids if self.instance_ids is not None else range(len(self.masks))
        if type(instance) is not int or instance not in ids:
            return None
        return self._instance(ids.index(instance))

    def options(self, positive, negative, multi):
        """Standalone prompts; refinement uses iter_instances with full context."""
        import numpy as np
        from .mask_candidates import check_mask
        from .mask_cleanup import inside
        from .instance_mask_candidates import select_instances
        if self.size is None:
            raise RuntimeError('RF-DETR-Seg requires an active image context.')
        def assess(variant):
            raw, score = variant
            mask, box, reason = check_mask(raw, score, positive, negative,
                [0, 0, *self.size], self.settings, self.size)
            target = raw if mask is None else mask
            rank = (reason is None, -int(inside(target, negative).sum()),
                    int(inside(target, positive).sum()), score)
            return rank, mask, box, reason, {}
        _, assessed, _ = select_instances(self, assess, positive, negative, limit=3 if multi else 1)
        accepted = [(mask, rank[-1]) for rank, mask, _, reason, _ in assessed if reason is None]
        return accepted or [(np.zeros(self.size[::-1], dtype=bool), 0.)]
