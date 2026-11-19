"""HQ-SAM 2 with hash-pinned sources, isolated imports and strict weights."""
import re
from . import hq_sam2_registry as registry


def load(config, device):
    from .mask_backends import asset_directory, checkpoint
    from .sapiensid_assets import _require_download
    from .sapiensid_downloads import install
    from .sapiensid_loading import checked_bytes
    from .sapiensid_inference import load_verified_weights
    from .sam_pose2seg import source_imports, Segmenter
    try:
        from hydra.utils import instantiate
        from omegaconf import OmegaConf
    except ImportError as error:
        raise RuntimeError('HQ-SAM 2 requires requirements-cpu.txt or requirements-gpu.txt.') from error
    root = asset_directory(config, 'hq_sam2_large', registry.SOURCE_REVISION)
    sources = {}
    for name, spec in registry.SOURCE_FILES.items():
        path = root / 'source' / name
        if not path.exists():
            _require_download(config, path)
            install(registry.SOURCE_URL + name, path, spec['size'], spec['sha256'])
        sources[name] = checked_bytes(path, spec['sha256'], limit=spec['size'])
    path = checkpoint(config, 'hq_sam2_large', registry.SOURCE_REVISION, registry.NAME,
                      registry.URL, registry.SIZE, registry.SHA256)
    # Rename only after verification; never shadow an installed sam2 package or
    # run upstream package initializers (which mutate Hydra's global state).
    namespace = '_stash_hqsam2'
    modules = {name.removeprefix('sam-hq2/').replace('sam2/', namespace + '/', 1):
               re.sub(rb'\bsam2\b', namespace.encode(), raw)
               for name, raw in sources.items() if name.endswith('.py')}
    configuration = re.sub(rb'\bsam2\b', namespace.encode(), sources[registry.CONFIG])
    with source_imports(modules, namespace=namespace):
        cfg = OmegaConf.create(configuration.decode())
        cfg.model.sam_mask_decoder_extra_args = dict(dynamic_multimask_via_stability=True,
            dynamic_multimask_stability_delta=.05, dynamic_multimask_stability_thresh=.98)
        model = instantiate(cfg.model, _recursive_=True)
        load_verified_weights(model, path, expected_size=registry.SIZE, expected_sha256=registry.SHA256)
        import importlib
        predictor_cls = importlib.import_module(namespace + '.sam2_image_predictor').SAM2ImagePredictor
        predictor = predictor_cls(model.float().to(device).eval(), max_hole_area=0., max_sprinkle_area=0.)
    return Segmenter(predictor, device,
                     frame_batch_size=config.data['execution']['sam_frame_batch_size'],
                     prompt_batch_size=config.data['execution']['sam_prompt_batch_size'],
                     precision=config.data['execution']['sam_precision'])
