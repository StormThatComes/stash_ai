"""Lazy, verified SAM-pose2seg loading without the OpenMMLab training stack."""
from .profiling import pipeline_stage, pipeline_timing, pipeline_cuda_timing
from contextlib import contextmanager
import hashlib
import importlib.abc
import importlib.util
import sys

from . import sam_pose2seg_registry as registry


def source_manifest():
    return {name: dict(spec) for name, spec in registry.SOURCE_FILES.items()}


def assets(config):
    from .sapiensid_assets import _require_download
    from .sapiensid_downloads import install
    from .sapiensid_loading import local_path, checked_bytes
    settings = config.data['sapiensid']['pose']['refinement']
    root = (local_path(config, settings['directory']) if settings['directory'] else
            config.model_dir / 'sam_pose2seg' / registry.REVISION)
    manifest = source_manifest()
    sources = {}
    for name, spec in manifest.items():
        path = root / 'source' / name
        if not path.exists():
            _require_download(config, path)
            install(registry.SOURCE_URL + name, path, spec['size'], spec['sha256'])
        sources[name] = checked_bytes(path, spec['sha256'], limit=spec['size'])
    path = root / registry.NAME
    if not path.exists():
        _require_download(config, path)
        install(registry.URL, path, registry.SIZE, registry.SHA256)
    return sources, path


class _Sources(importlib.abc.MetaPathFinder, importlib.abc.Loader):
    """Import only pinned bytes; never execute upstream package initializers.

    The initializers import the full detection/training stack. Namespace
    packages here expose only the verified SAM inference dependency closure.
    No downloaded directory is added to sys.path or a package search path.
    """
    def __init__(self, sources):
        self.modules = {name[:-3].replace('/', '.'): raw
                        for name, raw in sources.items() if name.endswith('.py')}
        self.packages = {name.rsplit('.', i)[0] for name in self.modules
                         for i in range(1, name.count('.') + 1)}

    def fingerprint(self, name):
        return hashlib.sha256(self.modules.get(name, b'')).hexdigest()

    def find_spec(self, fullname, path=None, target=None):
        if fullname in self.modules or fullname in self.packages:
            return importlib.util.spec_from_loader(fullname, self, is_package=fullname in self.packages)
        return None

    def create_module(self, spec):
        return None

    def exec_module(self, module):
        name = module.__name__
        module.__stash_sam_source__ = self.fingerprint(name)
        if name in self.modules:
            filename = 'sam-pose2seg@' + registry.SOURCE_REVISION + '/' + name.replace('.', '/') + '.py'
            module.__file__ = filename
            exec(compile(self.modules[name], filename, 'exec'), module.__dict__)


@contextmanager
def source_imports(sources):
    loader = _Sources(sources)
    for name, module in tuple(sys.modules.items()):
        if name == 'bboxmaskpose' or name.startswith('bboxmaskpose.'):
            if (name not in loader.modules and name not in loader.packages or
                    getattr(module, '__stash_sam_source__', None) != loader.fingerprint(name)):
                raise RuntimeError('SAM-pose2seg cannot reuse foreign Python package ' + name)
    sys.meta_path.insert(0, loader)
    try:
        yield
    finally:
        sys.meta_path.remove(loader)


@pipeline_stage('sam_model_load')
def load(config, device):
    from .inference_precision import validate
    validate(device,config.data['execution']['sam_precision'])
    try:
        import torch
        import torchvision
        import iopath
        from hydra.utils import instantiate
        from omegaconf import OmegaConf
    except ImportError as error:
        raise RuntimeError('SAM-pose2seg requires requirements-cpu.txt or requirements-gpu.txt in the CPU/GPU environment.') from error
    sources, path = assets(config)
    from .sapiensid_inference import load_verified_weights
    with source_imports(sources):
        cfg = OmegaConf.create(sources[registry.CONFIG].decode('utf-8'))
        # Same decoder stability fallback as upstream build_sam2; no CUDA
        # extension is needed (hole/sprinkle postprocessing stays disabled).
        cfg.model.sam_mask_decoder_extra_args = dict(dynamic_multimask_via_stability=True,
            dynamic_multimask_stability_delta=.05, dynamic_multimask_stability_thresh=.98)
        model = instantiate(cfg.model, _recursive_=True)
        load_verified_weights(model, path, expected_size=registry.SIZE, expected_sha256=registry.SHA256)
        model = model.float().to(device).eval()
        from bboxmaskpose.sam2.sam2_image_predictor import SAM2ImagePredictor
        predictor = SAM2ImagePredictor(model, max_hole_area=0., max_sprinkle_area=0.)
    return Segmenter(predictor, device, frame_batch_size=config.data['execution']['sam_frame_batch_size'],
                     prompt_batch_size=config.data['execution']['sam_prompt_batch_size'],
                     precision=config.data['execution']['sam_precision'])


class Segmenter:
    def __init__(self, predictor, device, *, frame_batch_size=4, prompt_batch_size=8, precision='fp32'):
        from .inference_precision import validate
        validate(device,precision)
        self.precision=precision
        self.predictor, self.device = predictor, device
        self.frame_batch_size, self.prompt_batch_size = frame_batch_size, prompt_batch_size

    def resident_modules(self):
        return [self.predictor.model]

    def prepare_residency(self):
        self.predictor.reset_predictor()

    @contextmanager
    def frame(self, image):
        import numpy as np
        import torch
        # One image embedding shared by all people; never retain frame features
        # across cues, including when prompting or the second pose pass fails.
        from .inference_precision import autocast
        with torch.inference_mode(), autocast(self.device,self.precision):
            try:
                with pipeline_timing('sam_image_embedding'), pipeline_cuda_timing('sam_image_encoder',self.device), image.convert('RGB') as rgb:
                    self.predictor.set_image(np.array(rgb, copy=True))
                yield self
            finally:
                self.predictor.reset_predictor()

    @pipeline_stage('sam_mask_prediction')
    def predict(self, points, negative=()):
        import numpy as np
        labels = np.ones(len(points), dtype=np.int32)
        if len(negative):
            points = np.concatenate((points, negative))
            labels = np.concatenate((labels, np.zeros(len(negative), dtype=np.int32)))
        if self.precision!='fp32':
            mask_input,coords,label,boxes=self.predictor._prep_prompts(points,labels,None,None,True)
            with pipeline_cuda_timing('sam_prompt_decoder',self.device):
                masks,scores,_=self.predictor._predict(coords,label,boxes,mask_input,multimask_output=False)
            masks=masks[0].cpu().numpy();scores=scores[0].float().cpu().numpy()
        else:
            with pipeline_cuda_timing('sam_prompt_decoder',self.device):
                masks, scores, _ = self.predictor.predict(point_coords=points, point_labels=labels,
                    box=None, multimask_output=False)
        masks, scores = np.asarray(masks), np.asarray(scores)
        if (masks.ndim != 3 or len(masks) != 1 or scores.shape != (1,) or
                not np.isfinite(masks).all() or not np.isin(masks, [0, 1]).all() or
                not np.isfinite(scores).all() or not 0 <= scores[0] <= 1):
            raise ValueError('SAM-pose2seg returned malformed/non-finite masks or scores.')
        return masks[0].astype(bool), float(scores[0])

    def predict_frames(self, frames):
        """Batch image embeddings and equal-length prompt sets; never pad tokens."""
        import numpy as np
        import torch
        from .adaptive_gpu import Batches
        from .profiling import pipeline_count
        frames=list(frames)
        image_batches=Batches(self.frame_batch_size,'SAM images')
        prompt_batches=Batches(self.prompt_batch_size,'SAM prompts')

        def calculate(batch):
            try:
                arrays=[]
                with pipeline_timing('sam_image_prepare'):
                    for image,_ in batch:
                        with image.convert('RGB') as rgb:
                            arrays.append(np.array(rgb,copy=True))
                with pipeline_timing('sam_image_embedding'), pipeline_cuda_timing('sam_image_encoder',self.device):
                    self.predictor.set_image_batch(arrays)
                arrays.clear()
                pipeline_count('sam_embedded_images',len(batch))
                results=[]
                for image_index,(_,prompts) in enumerate(batch):
                    groups={};ordered=[None]*len(prompts)
                    for index,(positive,negative) in enumerate(prompts):
                        groups.setdefault(len(positive)+len(negative),[]).append((index,positive,negative))
                    def decode(group):
                        coordinates=np.stack([np.concatenate((p,np.asarray(n).reshape(-1,2))) for _,p,n in group])
                        labels=np.stack([np.r_[np.ones(len(p),dtype=np.int32),np.zeros(len(n),dtype=np.int32)] for _,p,n in group])
                        with pipeline_timing('sam_mask_prediction'), pipeline_cuda_timing('sam_prompt_decoder',self.device):
                            # Private API is pinned by the verified source manifest.
                            mask_input,coords,label,boxes=self.predictor._prep_prompts(
                                coordinates,labels,None,None,True,img_idx=image_index)
                            masks,scores,_=self.predictor._predict(coords,label,boxes,mask_input,
                                multimask_output=False,return_logits=False,img_idx=image_index)
                        # Avoid transferring the unused low-resolution logits.
                        with pipeline_timing('sam_mask_readback'):
                            masks=masks.cpu().numpy();scores=scores.float().cpu().numpy()
                        if len(masks)!=len(group) or len(scores)!=len(group):
                            raise ValueError('SAM returned an invalid prompt batch count.')
                        pipeline_count('sam_decoded_prompts',len(group))
                        return [self._checked(mask,score) for mask,score in zip(masks,scores)]
                    for group in groups.values():
                        values=prompt_batches.run(group,decode,self.device)
                        for (index,_,_),value in zip(group,values):ordered[index]=value
                    results.append(ordered)
                return results
            finally:
                self.predictor.reset_predictor()
        from .inference_precision import autocast
        with torch.inference_mode(), autocast(self.device,self.precision):
            return image_batches.run(frames,calculate,self.device)

    @staticmethod
    def _checked(masks,scores):
        import numpy as np
        if (masks.ndim!=3 or len(masks)!=1 or scores.shape!=(1,) or
                not np.isfinite(masks).all() or not np.isin(masks,[0,1]).all() or
                not np.isfinite(scores).all() or not 0<=scores[0]<=1):
            raise ValueError('SAM-pose2seg returned malformed/non-finite masks or scores.')
        return masks[0].astype(bool,copy=False),float(scores[0])
