"""Semantic visible-face segmentation and policy-independent measurements."""
from dataclasses import asdict
import math

from .config import digest
from .models import ASSETS, verified_bytes

CONTRACT = {'version': 1, 'model': asdict(ASSETS['faceocc']), 'input': 'RGB-256-ImageNet-mean-std',
            'crop': 'square-face-box-padding-0.25', 'region': 'face-box-ellipse-0.45w-0.48h',
            'mask_threshold': 'logit>0', 'fraction': 'visible-pixels/ellipse-pixels'}
CONTRACT_ID = digest(CONTRACT)


class VisibilityError(RuntimeError):
    pass


class VisibilityGeometryError(ValueError):
    """This observation has no measurable face region in the source image."""


def required(policy):
    return policy['weights']['visibility'] > 0


def valid_fraction(value):
    return type(value) in (int, float) and 0 <= value <= 1 and math.isfinite(value)


def cached(quality):
    record = quality.get('visibility')
    return (isinstance(record, dict) and record.get('contract') == CONTRACT_ID
            and valid_fraction(quality.get('visible_face_fraction')))


class LazyVisibility:
    """One lazy segmentation session for the complete recognition invocation."""
    def __init__(self, config, paths, factory=None):
        self.config, self.paths, self.factory = config, paths, factory or FaceVisibility
        self.model = None
        self._initialization_error = None
        self._inference_failures = 0

    def measure(self, image, face):
        from stash_ai.runtime.shutdown import require_idle
        require_idle()
        try:
            if self.model is None:
                if self._initialization_error is not None:
                    raise ValueError(self._initialization_error)
                try:
                    self.model = self.factory(self.config, self.paths)
                except Exception as error:
                    self._initialization_error = str(error)[:500]
                    raise
        except Exception as error:
            raise VisibilityError('Face visibility measurement failed: ' + str(error)) from error
        try:
            value = self.model.measure(image, face)
            if not isinstance(value, dict) or not cached(value):
                raise ValueError('Visibility model returned an invalid or incompatible measurement.')
        except VisibilityGeometryError:
            raise
        except Exception as error:
            from stash_ai.runtime.cleanup import cleanup
            model, self.model = self.model, None
            self._inference_failures += 1
            if self._inference_failures >= 2:
                self._initialization_error = 'Visibility session recovery exhausted after repeated inference failures.'
            cleanup([('close failed visibility model', model.close)])
            raise VisibilityError('Face visibility measurement failed: ' + str(error)) from error
        return value

    def close(self):
        self._initialization_error = None
        self._inference_failures = 0
        if self.model is not None:
            model, self.model = self.model, None
            model.close()


def prepare(image, face):
    """Fixed crop and elliptical face ROI; portrait padding never affects ratio."""
    import numpy as np
    from PIL import Image
    x, y, width, height = face.box
    side = max(width, height) * 1.5
    left, top = x + width / 2 - side / 2, y + height / 2 - side / 2
    with image.convert('RGB') as rgb, rgb.transform((256, 256), Image.Transform.AFFINE,
            (side / 256, 0, left, 0, side / 256, top), resample=Image.Resampling.BILINEAR) as crop:
        data = np.asarray(crop, dtype=np.float32) / 255.
    data = (data - np.array([.485, .456, .406], dtype=np.float32)) / np.array([.229, .224, .225], dtype=np.float32)
    rows, columns = np.indices((256, 256), dtype=np.float32)
    region = (((columns + .5 - 128) / (width * .45 * 256 / side)) ** 2
              + ((rows + .5 - 128) / (height * .48 * 256 / side)) ** 2 <= 1)
    # Exclude synthetic padding from the visibility denominator.
    real = ((left + (columns + .5) * side / 256 >= 0) & (left + (columns + .5) * side / 256 < image.width)
            & (top + (rows + .5) * side / 256 >= 0) & (top + (rows + .5) * side / 256 < image.height))
    if not region.any() or not real[region].all():
        raise VisibilityGeometryError('Visibility face region extends beyond the real source.')
    return np.ascontiguousarray(data.transpose(2, 0, 1)[None]), region


def measurement(logits, region, device):
    import numpy as np
    values = np.asarray(logits)
    if values.shape != (1, 1, 256, 256) or not np.isfinite(values).all():
        raise ValueError('FaceOcc returned invalid/non-finite mask logits.')
    count = int(region.sum())
    if count <= 0:
        raise ValueError('Visibility face region is empty.')
    visible = int(((values[0, 0] > 0) & region).sum())
    return {'visible_face_fraction': visible / count,
            'visibility': {'contract': CONTRACT_ID, 'method': 'faceocc-ellipse-v1', 'device': device,
                           'model_sha256': ASSETS['faceocc'].sha256, 'model_revision': ASSETS['faceocc'].revision,
                           'visible_pixels': visible, 'face_pixels': count}}


class FaceVisibility:
    def __init__(self, config, paths):
        import torch
        from safetensors.torch import load
        from .faceocc import build_model
        path = paths['faceocc']
        blob = verified_bytes(path, ASSETS['faceocc'])
        requested = config.data['models']['device']
        available = torch.cuda.is_available()
        if requested == 'cuda' and not available:
            raise ValueError('Face visibility requested CUDA but PyTorch CUDA is unavailable.')
        self.device = 'cuda' if available and requested != 'cpu' else 'cpu'
        self.model = build_model()
        self.model.load_state_dict(load(blob), strict=True)
        try:
            self.model.to(self.device).eval()
        except Exception:
            if requested != 'auto' or self.device != 'cuda':
                self.close()
                raise
            self.device = 'cpu'
            self.model.to('cpu').eval()

    def measure(self, image, face):
        import torch
        blob, region = prepare(image, face)
        with torch.inference_mode():
            logits = self.model(torch.from_numpy(blob).to(self.device)).float().cpu().numpy()
        return measurement(logits, region, self.device)

    def close(self):
        self.model = None
