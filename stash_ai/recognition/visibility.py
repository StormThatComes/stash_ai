"""Semantic visible-face segmentation and policy-independent measurements."""
from dataclasses import asdict
from functools import lru_cache
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
    from collections.abc import Mapping
    record = quality.get('visibility')
    return (isinstance(record, Mapping) and record.get('contract') == CONTRACT_ID
            and valid_fraction(quality.get('visible_face_fraction')))


class LazyVisibility:
    """One lazy segmentation session for the complete recognition invocation."""
    def __init__(self, config, paths, factory=None):
        self.config, self.paths, self.factory = config, paths, factory or FaceVisibility
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

    def measure(self, image, face):
        self._ensure()
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
                        or any(not isinstance(v,VisibilityGeometryError) and not (isinstance(v,dict) and cached(v)) for v in values)):
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
                if not cached(value):raise ValueError('Invalid visibility measurement.')
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
            if len(values)!=len(prepared) or any(not isinstance(v,VisibilityGeometryError) and not cached(v) for v in values):
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


def prepare(image, face):
    """Fixed crop and elliptical face ROI; portrait padding never affects ratio."""
    import numpy as np
    from PIL import Image
    x, y, width, height = face.box
    side = max(width, height) * 1.5
    left, top = x + width / 2 - side / 2, y + height / 2 - side / 2
    with image.convert('RGB') as rgb, rgb.transform((256, 256), Image.Transform.AFFINE,
            (side / 256, 0, left, 0, side / 256, top), resample=Image.Resampling.BILINEAR) as crop:
        data = np.array(crop,dtype=np.float32,copy=True)
        np.divide(data,255.,out=data)
    mean,std,rows,columns=_constants()
    np.subtract(data,mean,out=data)
    np.divide(data,std,out=data)
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
    return _counts(visible,count,device)


def _counts(visible,count,device):
    if type(visible) is not int or type(count) is not int or count<=0 or not 0<=visible<=count:
        raise ValueError('Visibility mask counts are invalid.')
    return {'visible_face_fraction': visible / count,
            'visibility': {'contract': CONTRACT_ID, 'method': 'faceocc-ellipse-v1', 'device': device,
                           'model_sha256': ASSETS['faceocc'].sha256, 'model_revision': ASSETS['faceocc'].revision,
                           'visible_pixels': visible, 'face_pixels': count}}

def device_measurements(logits,regions,device):
    """Exact boolean/integer reduction; transfer N counts, never N full masks."""
    import numpy as np
    import torch
    if tuple(logits.shape)!=(len(regions),1,256,256) or not bool(torch.isfinite(logits).all().item()):
        raise ValueError('FaceOcc returned invalid/non-finite mask logits.')
    if any(np.asarray(region).shape!=(256,256) or np.asarray(region).dtype!=np.bool_ for region in regions):
        raise ValueError('Visibility face region has invalid shape/type.')
    counts=[int(region.sum()) for region in regions]
    if any(value<=0 for value in counts):raise ValueError('Visibility face region is empty.')
    mask=torch.from_numpy(np.stack(regions)).to(logits.device)
    visible=((logits[:,0]>0)&mask).sum(dim=(1,2)).cpu().tolist()
    return [_counts(int(value),count,device) for value,count in zip(visible,counts)]



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
        return self.measure_many([(image,face)])[0]

    def measure_many(self, items):
        import numpy as np
        import torch
        if len(items)>8:raise ValueError('Visibility microbatch exceeds eight faces.')
        prepared=[]
        for image,face in items:
            try:prepared.append(prepare(image,face))
            except VisibilityGeometryError as error:prepared.append(error)
        result=self.measure_prepared_many(prepared)
        if len(items)==1 and isinstance(result[0],Exception):raise result[0]
        return result

    def measure_prepared_many(self,prepared):
        import numpy as np
        import torch
        if len(prepared)>8:raise ValueError('Visibility microbatch exceeds eight faces.')
        result=[None]*len(prepared);blobs=[];regions=[];positions=[]
        for index,item in enumerate(prepared):
            if isinstance(item,VisibilityGeometryError):
                result[index]=item;continue
            blob,region=item
            if not isinstance(blob,np.ndarray) or blob.shape!=(1,3,256,256) or blob.dtype!=np.float32 or not np.isfinite(blob).all():
                raise ValueError('Invalid prepared FaceOcc tensor.')
            blobs.append(blob);regions.append(region);positions.append(index)
        if blobs:
            with torch.inference_mode():
                logits=self.model(torch.from_numpy(blobs[0] if len(blobs)==1 else np.concatenate(blobs)).to(self.device)).float()
                if self.device=='cpu':
                    values=logits.cpu().numpy()
                    if values.shape!=(len(blobs),1,256,256) or not np.isfinite(values).all():
                        raise ValueError('FaceOcc returned invalid/non-finite mask logits.')
                    measurements=[measurement(values[i:i+1],region,self.device) for i,region in enumerate(regions)]
                else:
                    measurements=device_measurements(logits,regions,self.device)
            for index,value in zip(positions,measurements):result[index]=value
        return result

    def close(self):
        self.model = None
