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
                       config.data['sapiensid']['pose']['refinement']['min_mask_score'])


class RFDETRMasks(FrameSegmenter):
    def __init__(self, model, device, precision, threshold):
        super().__init__(model, device, precision)
        self.threshold = threshold
        self.masks = self.scores = None

    def clear(self):
        super().clear()
        self.masks = self.scores = None

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
        self.scores = scores[indices].cpu().numpy()
        # Keep packed masks on the CPU, resize one plane at a time to avoid an
        # N-person full-resolution floating point allocation on the GPU.
        self.masks = []
        for index in indices:
            mask = F.interpolate(masks[0, index][None, None].float(), size=self.size[::-1],
                                 mode='bilinear', align_corners=False)[0, 0] > 0
            self.masks.append(np.packbits(mask.cpu().numpy(), axis=None))

    def options(self, positive, negative, multi):
        import numpy as np
        from .mask_cleanup import inside
        if self.masks is None:
            raise RuntimeError('RF-DETR-Seg requires an active image context.')
        candidates = []
        for packed, score in zip(self.masks, self.scores):
            mask = np.unpackbits(packed, count=self.size[0]*self.size[1]).reshape(self.size[::-1]).astype(bool)
            # Never combine two instances or clip a foreign arm into a guessed
            # person. Prompts select whole instances; common admission checks
            # subsequently validate all reliable foreign core landmarks.
            if len(positive) and inside(mask, positive).all() and not inside(mask, negative).any():
                candidates.append((mask, float(score)))
                if len(candidates) == (3 if multi else 1):
                    break
        if not candidates:
            candidates = [(np.zeros(self.size[::-1], dtype=bool), 0.)]
        return self.checked(np.stack([m for m, _ in candidates]), [s for _, s in candidates])
