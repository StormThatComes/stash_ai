"""Semantic visible-face segmentation and policy-independent measurements."""
from dataclasses import asdict
from functools import lru_cache
import math

from .config import DEFAULT_FACEOCC_LOGIT_THRESHOLD, MAX_FLOAT32, digest
from .models import ASSETS, verified_bytes

CONTRACT = {'version': 6, 'model': asdict(ASSETS['faceocc']), 'input': 'RGB-256-ImageNet-mean-std',
            'crop': 'square-face-box-padding-0.25-profile-roll-aligned',
            'region': 'source-face-box-ellipse-0.45w-0.48h-inverse-mapped',
            'mask_threshold': 'logit>0', 'fraction': 'visible-pixels/ellipse-pixels',
            'sharpness': 'foreground-interior-laplacian-256-v1'}
CONTRACT_ID = digest(CONTRACT)
TUNED_CONTRACT_ID = digest(dict(CONTRACT, mask_threshold='logit>record.logit_threshold'))


class VisibilityError(RuntimeError):
    pass


class VisibilityGeometryError(ValueError):
    """This observation has no measurable face region in the source image."""

    def __init__(self, message, *, source_pixels=None, face_pixels=None):
        super().__init__(message)
        self.diagnostics = {'visibility_geometry_error': message}
        if face_pixels is not None:
            self.diagnostics.update(visibility_source_pixels=source_pixels, visibility_face_pixels=face_pixels)
            if face_pixels:
                self.diagnostics['visibility_source_fraction'] = source_pixels / face_pixels


def required(policy):
    return policy['weights']['visibility'] > 0


def valid_fraction(value):
    return type(value) in (int, float) and 0 <= value <= 1 and math.isfinite(value)


def cached(quality, threshold=None):
    """Validate a recorded measurement, optionally against the requested pixel cutoff."""
    from collections.abc import Mapping
    record = quality.get('visibility')
    if not isinstance(record, Mapping) or not valid_fraction(quality.get('visible_face_fraction')):
        return False
    cutoff = record.get('logit_threshold', DEFAULT_FACEOCC_LOGIT_THRESHOLD)
    if type(cutoff) not in (int, float) or not -MAX_FLOAT32 <= cutoff <= MAX_FLOAT32:
        return False
    contract = CONTRACT_ID if cutoff == DEFAULT_FACEOCC_LOGIT_THRESHOLD else TUNED_CONTRACT_ID
    return record.get('contract') == contract and (threshold is None or cutoff == threshold)


class LazyVisibility:
    """One lazy segmentation session for the complete recognition invocation."""
    def __init__(self, config, paths, factory=None):
        self.config, self.paths, self.factory = config, paths, factory or FaceVisibility
        self.logit_threshold = config.data['detection']['faceocc_logit_threshold'] if config is not None else DEFAULT_FACEOCC_LOGIT_THRESHOLD
        self.model = None
        self._initialization_error = None
        self._inference_failures = 0
        self._batch_disabled = False

    def _ensure(self):
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
        return self.model

    def supports_prepared(self):
        return callable(getattr(self._ensure(),'measure_prepared_many',None))

    def measure(self, image, face):
        self._ensure()
        try:
            value = self.model.measure(image, face)
            if not isinstance(value, dict) or not cached(value, self.logit_threshold):
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

    def measure_many(self, items):
        """At most eight detached faces; geometry errors stay observation-local."""
        from itertools import islice
        items=list(islice(iter(items),9))
        if len(items)>8:raise ValueError('Visibility microbatch exceeds eight faces.')
        if not items:return []
        try:model=self._ensure()
        except Exception as error:return [error for _ in items]
        many=getattr(model,'measure_many',None)
        if not self._batch_disabled and callable(many) and len(items)>1:
            try:
                values=many(items)
                if (not isinstance(values,list) or len(values)!=len(items)
                        or any(not isinstance(v,VisibilityGeometryError) and not (isinstance(v,dict) and cached(v, self.logit_threshold)) for v in values)):
                    raise ValueError('Visibility model returned an invalid batch.')
                return values
            except Exception:
                # Unsupported batch/session failure is an optimization failure;
                # release its session and use existing bounded scalar recovery.
                from stash_ai.runtime.cleanup import cleanup
                self.model=None;self._batch_disabled=True
                cleanup([('close failed visibility batch',model.close)])
        result=[]
        for image,face in items:
            try:result.append(self.measure(image,face))
            except Exception as error:result.append(error)
        return result

    def measure_prepared_many(self,prepared):
        """Model-owner entry for independently owned CPU preparation buffers."""
        if len(prepared)>8:raise ValueError('Visibility microbatch exceeds eight faces.')
        def scalar(item):
            try:
                model=self._ensure()
                if isinstance(item,VisibilityGeometryError):return item
                value=model.measure_prepared_many([item])[0]
                if not cached(value, self.logit_threshold):raise ValueError('Invalid visibility measurement.')
                return value
            except VisibilityGeometryError as error:return error
            except Exception as error:
                if self.model is not None:
                    from stash_ai.runtime.cleanup import cleanup
                    model,self.model=self.model,None
                    self._inference_failures+=1
                    if self._inference_failures>=2:
                        self._initialization_error='Visibility session recovery exhausted after repeated inference failures.'
                    cleanup([('close failed prepared visibility model',model.close)])
                return error if isinstance(error,VisibilityError) else VisibilityError('Face visibility measurement failed: '+str(error))
        if self._batch_disabled or len(prepared)<2:return [scalar(item) for item in prepared]
        try:
            model=self._ensure()
        except Exception as error:return [error for _ in prepared]
        try:
            values=model.measure_prepared_many(prepared)
            if len(values)!=len(prepared) or any(not isinstance(v,VisibilityGeometryError) and not cached(v, self.logit_threshold) for v in values):
                raise ValueError('Invalid visibility batch.')
            return values
        except Exception:
            from stash_ai.runtime.cleanup import cleanup
            self.model=None;self._batch_disabled=True
            cleanup([('close failed prepared visibility batch',model.close)])
            return [scalar(item) for item in prepared]

    def close(self):
        self._initialization_error = None
        self._inference_failures = 0
        self._batch_disabled = False
        if self.model is not None:
            model, self.model = self.model, None
            model.close()


@lru_cache(maxsize=1)
def _constants():
    import numpy as np
    def immutable(values):
        return np.frombuffer(np.asarray(values,dtype=np.float32).tobytes(),dtype=np.float32)
    return (immutable([.485,.456,.406]),immutable([.229,.224,.225]),
            immutable(np.arange(256))[:,None],immutable(np.arange(256))[None,:])


def prepare(image, face, *, allow_partial=False):
    """Return aligned pixels and the real ellipse for geometric sharpness."""
    blob, region, _ = prepare_measurement(image, face, 0. if allow_partial else 1.)
    return blob, region


def prepare_measurement(image, face, minimum_source_fraction):
    """Prepare FaceOcc input only when the source covers the configured fraction of its ellipse."""
    return prepare_aligned(image, face, minimum_source_fraction)[:3]


def ellipse_radii(width, height):
    return width * .45, height * .48


def prepare_aligned(image, face, minimum_source_fraction):
    """Return the FaceOcc input, real ROI, full count and model-to-source affine."""
    import numpy as np
    from PIL import Image
    points = np.asarray(face.landmarks, dtype=np.float64)
    if points.shape != (5, 2) or not np.isfinite(points).all():
        raise VisibilityGeometryError('Visibility alignment requires five finite landmarks.')
    eye = points[1] - points[0]
    distance = float(np.linalg.norm(eye))
    if not math.isfinite(distance) or distance <= 1:
        raise VisibilityGeometryError('Visibility alignment requires distinct eye landmarks.')
    x, y, width, height = face.box
    axis = eye / distance
    # At a narrow eye baseline, a one-pixel error can rotate the entire crop.
    # Use the longer vertical facial axis only when both nose and mouth agree.
    minimum = max(2., .16 * min(width, height))
    if distance < minimum:
        eyes = points[:2].mean(axis=0)
        mouth = points[3:].mean(axis=0)
        down = mouth - eyes
        length = np.linalg.norm(down)
        if length > max(3., .15 * min(width, height)):
            down /= length
            nose_height = np.dot(points[2]-eyes, down) / length
            if .12 < nose_height < .90:
                horizontal = np.array([down[1], -down[0]])
                weight = min(1., 2 * (1 - distance / minimum))
                mixed = (1-weight)*axis + weight*horizontal
                if np.linalg.norm(mixed) > .25:
                    axis = mixed / np.linalg.norm(mixed)
    cosine, sine = map(float, axis)
    side = max(width, height) * 1.5
    center_x, center_y = x + width / 2, y + height / 2
    scale = side / 256
    # PIL maps output coordinates back into the source. Rotate in this single
    # resampling pass, including upside-down faces; never mirror eye ordering.
    transform = (scale * cosine, -scale * sine, center_x - side / 2 * (cosine - sine),
                 scale * sine, scale * cosine, center_y - side / 2 * (sine + cosine))
    with image.convert('RGB') as rgb, rgb.transform((256, 256), Image.Transform.AFFINE,
            transform, resample=Image.Resampling.BILINEAR) as crop:
        data = np.array(crop,dtype=np.float32,copy=True)
        np.divide(data,255.,out=data)
    mean,std,rows,columns=_constants()
    np.subtract(data,mean,out=data)
    np.divide(data,std,out=data)
    dx = cosine * (columns + .5 - 128) - sine * (rows + .5 - 128)
    dy = sine * (columns + .5 - 128) + cosine * (rows + .5 - 128)
    # Keep the original physical ROI, rather than replacing it with a smaller
    # aligned ellipse that could silently discard an occluder near its edges.
    radius_x, radius_y = ellipse_radii(width, height)
    region = ((dx / (radius_x * 256 / side)) ** 2
              + (dy / (radius_y * 256 / side)) ** 2 <= 1)
    # Padding cannot contribute to foreground or sharpness.
    source_x, source_y = center_x + dx * scale, center_y + dy * scale
    real = ((source_x >= 0) & (source_x < image.width)
            & (source_y >= 0) & (source_y < image.height))
    full_count = int(region.sum())
    region &= real
    source_count = int(region.sum())
    if not source_count or source_count / full_count < minimum_source_fraction:
        raise VisibilityGeometryError('Insufficient face ellipse coverage in the real source.',
                                      source_pixels=source_count, face_pixels=full_count)
    return np.ascontiguousarray(data.transpose(2, 0, 1)[None]), region, full_count, transform


def _pixel_logits(logits):
    import numpy as np
    values = np.asarray(logits)
    if values.shape != (1, 1, 256, 256) or not np.isfinite(values).all():
        raise ValueError('FaceOcc returned invalid/non-finite mask logits.')
    return values[0, 0]


def foreground_mask(pixel_logits, region, threshold):
    return (pixel_logits > threshold) & region


def measurement(logits, region, device, *, blob=None, face_pixels=None, threshold=DEFAULT_FACEOCC_LOGIT_THRESHOLD):
    foreground = foreground_mask(_pixel_logits(logits), region, threshold)
    count = int(region.sum())
    if count <= 0:
        raise ValueError('Visibility face region is empty.')
    visible = int(foreground.sum())
    result = _counts(visible, count, device, face_pixels, threshold)
    if blob is not None:
        from .foreground_quality import face
        result.update(face(blob,foreground,'faceocc_mask'))
    return result


def _counts(visible, source_count, device, face_pixels, threshold):
    count = source_count if face_pixels is None else face_pixels
    if (type(visible) is not int or type(count) is not int or
            not 0 < source_count <= count or not 0 <= visible <= source_count):
        raise ValueError('Visibility mask counts are invalid.')
    return {'visible_face_fraction': visible / count,
            'visibility': {'contract': CONTRACT_ID if threshold == DEFAULT_FACEOCC_LOGIT_THRESHOLD else TUNED_CONTRACT_ID,
                           'logit_threshold': threshold, 'method': 'faceocc-ellipse-profile-roll-v4', 'device': device,
                           'model_sha256': ASSETS['faceocc'].sha256, 'model_revision': ASSETS['faceocc'].revision,
                           'visible_pixels': visible, 'face_pixels': count, 'source_pixels': source_count,
                           'source_fraction': source_count / count}}

def device_measurements(logits,regions,device,*,inputs=None,face_pixels=None,threshold=DEFAULT_FACEOCC_LOGIT_THRESHOLD):
    """Transfer scalar visibility/sharpness summaries, never full masks."""
    import numpy as np
    import torch
    if tuple(logits.shape)!=(len(regions),1,256,256) or not bool(torch.isfinite(logits).all().item()):
        raise ValueError('FaceOcc returned invalid/non-finite mask logits.')
    if any(np.asarray(region).shape!=(256,256) or np.asarray(region).dtype!=np.bool_ for region in regions):
        raise ValueError('Visibility face region has invalid shape/type.')
    counts=[int(region.sum()) for region in regions]
    if any(value<=0 for value in counts):raise ValueError('Visibility face region is empty.')
    mask=torch.from_numpy(np.stack(regions)).to(logits.device)
    foreground=foreground_mask(logits[:,0], mask, threshold)
    visible=foreground.sum(dim=(1,2)).cpu().tolist()
    totals = counts if face_pixels is None else face_pixels
    results = [_counts(int(value), count, device, total, threshold)
               for value, count, total in zip(visible, counts, totals, strict=True)]
    if inputs is not None:
        from .foreground_quality import device_faces
        for result,quality in zip(results,device_faces(inputs,foreground)):
            result.update(quality)
    return results



class FaceVisibility:
    def __init__(self, config, paths):
        self.logit_threshold = config.data['detection']['faceocc_logit_threshold']
        self.minimum_source_fraction = config.data['detection']['min_face_source_fraction']
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

    def inspect_logits(self, blob):
        """Keep one aligned score map on the CPU for interactive threshold previews."""
        import torch
        with torch.inference_mode():
            logits = self.model(torch.from_numpy(blob).to(self.device)).float().cpu().numpy()
        return _pixel_logits(logits)

    def inspect_foreground(self, blob, region):
        return foreground_mask(self.inspect_logits(blob), region, self.logit_threshold)

    def measure(self, image, face):
        return self.measure_many([(image,face)])[0]

    def measure_many(self, items):
        import numpy as np
        import torch
        if len(items)>8:raise ValueError('Visibility microbatch exceeds eight faces.')
        prepared=[]
        for image,face in items:
            try:prepared.append(prepare_measurement(image, face, self.minimum_source_fraction))
            except VisibilityGeometryError as error:prepared.append(error)
        result=self.measure_prepared_many(prepared)
        if len(items)==1 and isinstance(result[0],Exception):raise result[0]
        return result

    def measure_prepared_many(self,prepared):
        import numpy as np
        import torch
        if len(prepared)>8:raise ValueError('Visibility microbatch exceeds eight faces.')
        result = [None] * len(prepared)
        blobs, regions, positions, totals = [], [], [], []
        for index,item in enumerate(prepared):
            if isinstance(item,VisibilityGeometryError):
                result[index]=item;continue
            blob, region, full_count = item
            if not isinstance(blob,np.ndarray) or blob.shape!=(1,3,256,256) or blob.dtype!=np.float32 or not np.isfinite(blob).all():
                raise ValueError('Invalid prepared FaceOcc tensor.')
            blobs.append(blob)
            regions.append(region)
            positions.append(index)
            totals.append(full_count)
        if blobs:
            with torch.inference_mode():
                inputs=torch.from_numpy(blobs[0] if len(blobs)==1 else np.concatenate(blobs)).to(self.device)
                logits=self.model(inputs).float()
                if self.device=='cpu':
                    values=logits.cpu().numpy()
                    if values.shape!=(len(blobs),1,256,256) or not np.isfinite(values).all():
                        raise ValueError('FaceOcc returned invalid/non-finite mask logits.')
                    measurements = [measurement(values[i:i+1], region, self.device, blob=blobs[i], face_pixels=totals[i], threshold=self.logit_threshold)
                                    for i, region in enumerate(regions)]
                else:
                    measurements=device_measurements(logits,regions,self.device,inputs=inputs,face_pixels=totals,threshold=self.logit_threshold)
            for index,value in zip(positions,measurements):result[index]=value
        return result

    def close(self):
        self.model = None
