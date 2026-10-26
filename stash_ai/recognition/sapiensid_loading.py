"""Revision-pinned SapiensID construction with lazy missing-asset downloads."""
from contextlib import contextmanager
import hashlib
import importlib
from pathlib import Path
import subprocess
import sys
import tempfile

from .config import parse_yaml
from .sapiensid_inference import SapiensIDV1Encoder, load_verified_weights

from .sapiensid_registry import UPSTREAM_COMMIT


def local_path(config, value):
    root = config.source_path.parent if config.source_path else Path.cwd()
    path = Path(value).expanduser()
    return path if path.is_absolute() else root / path


def checked_bytes(path, checksum, *, limit):
    with Path(path).open('rb') as stream:
        data = stream.read(limit + 1)
    if not data or len(data) > limit or hashlib.sha256(data).hexdigest() != checksum:
        raise ValueError(f'SapiensID asset size/SHA256 mismatch: {path}')
    return data


def trusted_checkout(path):
    root = Path(path).resolve(strict=True)
    from .sapiensid_assets import MARKER, verify_source
    if (root / MARKER).exists():
        return verify_source(root)
    required = ('tasks/sapiensID/src/models/sapiens_id/__init__.py', 'general_utils/img_utils.py')
    try:
        def git(*args):
            return subprocess.run(['git', '-C', str(root), *args], capture_output=True,
                                  text=True, check=True, timeout=15).stdout.strip()
        if (git('rev-parse', 'HEAD') != UPSTREAM_COMMIT or
                Path(git('rev-parse', '--show-toplevel')).resolve() != root or
                git('status', '--porcelain', '--untracked-files=all')):
            raise ValueError('SapiensID source must be a clean checkout at ' + UPSTREAM_COMMIT)
        git('ls-files', '--error-unmatch', '--', *required)
    except (OSError, subprocess.SubprocessError) as error:
        raise ValueError('Cannot verify the pinned SapiensID source checkout.') from error
    if any(not (root / path).is_file() for path in required):
        raise ValueError('SapiensID source checkout is incomplete.')
    return root


@contextmanager
def source_imports(root):
    # Official source uses these absolute package names. Never reuse a package
    # from a different checkout (including a previous incomplete import).
    paths = {'src': root / 'tasks/sapiensID/src', 'general_utils': root / 'general_utils'}
    for name, expected in paths.items():
        for key, module in tuple(sys.modules.items()):
            if key != name and not key.startswith(name + '.'):
                continue
            locations = getattr(module, '__path__', None)
            file = getattr(module, '__file__', None)
            if locations is not None:
                valid = all(Path(p).resolve().is_relative_to(expected) for p in locations)
            else:
                valid = file is not None and Path(file).resolve().is_relative_to(expected)
            if not valid:
                raise RuntimeError(f'SapiensID cannot reuse foreign Python package {key}.')
    previous = list(sys.path)
    sys.path[:0] = [str(root / 'tasks/sapiensID'), str(root)]
    try:
        yield
    finally:
        sys.path[:] = previous


def load_encoder(config):
    settings = config.data['sapiensid']['model']
    from .sapiensid_assets import encoder_assets
    root, directory = encoder_assets(config)
    raw = checked_bytes(directory / 'model.yaml', settings['model_yaml_sha256'], limit=131072)
    cfg = parse_yaml(raw, context='SapiensID model')
    if (not isinstance(cfg, dict) or cfg.get('name') != 'mrm' or
            cfg.get('input_size') != [384, 384] or cfg.get('output_dim') != 4096 or
            cfg.get('color_space') != 'RGB' or not cfg.get('square_pad') or
            cfg.get('rgb_mean') != [.5, .5, .5] or cfg.get('rgb_std') != [.5, .5, .5] or
            cfg.get('kp_preprocessor', {}).get('name') != 'kp19'):
        raise ValueError('Unsupported SapiensID v1 wb4m model.yaml architecture/preprocessing.')
    from .sapiensid_v1 import verify_wb4m_checkpoint
    verify_wb4m_checkpoint(directory / 'model.pth')
    try:
        with source_imports(root):
            from omegaconf import OmegaConf
            model_type = importlib.import_module('src.models.sapiens_id').SapiensModel
            cfg.update(start_from='', pretrained_path='', freeze=True)
            model = model_type.from_config(OmegaConf.create(cfg))
        load_verified_weights(model, directory / 'model.pth')
        return SapiensIDV1Encoder(model, None, device=config.data['models']['device'],
            max_batch_size=config.data['sapiensid']['execution']['batch_size'])
    except ImportError as error:
        raise RuntimeError('Install requirements-sapiensid.txt in the existing CPU/GPU environment.') from error


def load_pose(config):
    from .sapiensid_assets import pose_asset
    path, checksum = pose_asset(config)
    data = checked_bytes(path, checksum, limit=134217728)
    # Ultralytics accepts a filename and may unpickle it. Load only the exact
    # hash-verified local bytes, with no Ultralytics path-to-download fallback.
    from ultralytics import YOLO
    with tempfile.TemporaryDirectory(prefix='stash-sapiensid-pose-') as temp:
        path = Path(temp) / 'yolov8n-pose.pt'
        path.write_bytes(data)
        model = YOLO(str(path), task='pose')
    return model
