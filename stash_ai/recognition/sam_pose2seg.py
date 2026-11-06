"""Lazy, verified SAM-pose2seg loading without the OpenMMLab training stack."""
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


def load(config, device):
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
    return Segmenter(predictor, device)


class Segmenter:
    def __init__(self, predictor, device):
        self.predictor, self.device = predictor, device

    @contextmanager
    def frame(self, image):
        import numpy as np
        import torch
        # One image embedding shared by all people; never retain frame features
        # across cues, including when prompting or the second pose pass fails.
        with torch.inference_mode(), torch.autocast(device_type=torch.device(self.device).type, enabled=False):
            try:
                with image.convert('RGB') as rgb:
                    self.predictor.set_image(np.array(rgb, copy=True))
                yield self
            finally:
                self.predictor.reset_predictor()

    def predict(self, points):
        import numpy as np
        # Upstream BMPv2 uses the three most visible positive points, no box.
        # This leaves missing limbs outside a D-FINE box available to SAM.
        masks, scores, _ = self.predictor.predict(point_coords=points,
            point_labels=np.ones(len(points), dtype=np.int32), box=None, multimask_output=False)
        masks, scores = np.asarray(masks), np.asarray(scores)
        if (masks.ndim != 3 or len(masks) != 1 or scores.shape != (1,) or
                not np.isfinite(masks).all() or not np.isin(masks, [0, 1]).all() or
                not np.isfinite(scores).all() or not 0 <= scores[0] <= 1):
            raise ValueError('SAM-pose2seg returned malformed/non-finite masks or scores.')
        return masks[0].astype(bool), float(scores[0])
