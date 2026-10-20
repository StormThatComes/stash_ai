"""Opt-in, offline bootstrap of upstream SapiensID v1 model and YOLO/DFA aligner.

Nothing here registers a face recognizer, downloads model weights, modifies
recognition SQLite, or publishes to Stash. Upstream Python code is executable
code: only use a trusted, clean, revision-pinned local checkout.

The existing JoyTag torch/torchvision installation is reused. Additional
upstream libraries are optional and must be installed explicitly.
"""
from __future__ import annotations

from contextlib import contextmanager
import hashlib
from importlib import import_module
import io
from pathlib import Path
import os
import shutil
import stat
import subprocess
import sys
import tempfile

from .sapiensid_v1 import WB4M_BYTES, WB4M_SHA256
from .sapiensid_inference import SapiensIDV1Encoder, load_verified_weights

# Audited upstream reference checkout. Do not accept mutable main/HEAD aliases.
UPSTREAM_COMMIT = 'a8d8011117db23f20c7ade3bc38e8075810ac886'
_IMAGE_BYTES_MAX = 128 * 1024
_POSE_BYTES_MAX = 128 * 1024 * 1024
_DFA_BYTES_MAX = 512 * 1024 * 1024


def _sha256(value, name):
    if (type(value) is not str or len(value) != 64 or
            any(char not in '0123456789abcdef' for char in value)):
        raise ValueError(f'{name}: expected a lowercase 64-character SHA256.')
    return value


def _read_checked(path, expected_sha256, *, max_bytes, name, destination=None):
    """Hash exactly what is read; optionally copy that stream for a local loader.

    Reject symlinks, special files, oversized assets and changing descriptors.
    YAML is returned as bounded bytes. Large pose weights are copied directly
    to a private path so a path-based external loader cannot reopen a changed
    operator-provided input.
    """
    expected_sha256 = _sha256(expected_sha256, name)
    path = Path(path)
    if path.is_symlink():
        raise ValueError(f'{name}: symbolic links are not accepted.')
    with path.open('rb') as source:
        before = os.fstat(source.fileno())
        if not stat.S_ISREG(before.st_mode) or before.st_size < 1 or before.st_size > max_bytes:
            raise ValueError(f'{name}: invalid asset type/size.')
        hash_value = hashlib.sha256()
        buffer = bytearray() if destination is None else None
        size = 0
        with (open(destination, 'xb') if destination is not None else _nullsink()) as output:
            while True:
                block = source.read(min(1024 * 1024, max_bytes - size + 1))
                if not block:
                    break
                size += len(block)
                if size > max_bytes:
                    raise ValueError(f'{name}: exceeded maximum asset size.')
                hash_value.update(block)
                if buffer is None:
                    output.write(block)
                else:
                    buffer.extend(block)
        after = os.fstat(source.fileno())
        if (size != before.st_size or size != after.st_size
                or (before.st_dev, before.st_ino) != (after.st_dev, after.st_ino)
                or hash_value.hexdigest() != expected_sha256):
            raise ValueError(f'{name}: changed asset or SHA256 mismatch.')
    return bytes(buffer) if buffer is not None else None


@contextmanager
def _nullsink():
    yield None


def _trusted_checkout(checkout):
    root = Path(checkout).expanduser().resolve(strict=True)
    required = (
        'tasks/sapiensID/src/models/sapiens_id/__init__.py',
        'tasks/sapiensID/src/aligners/keypoint_predictor/__init__.py',
        'tasks/sapiensID/src/aligners/keypoint_predictor/face_detector/dfa/__init__.py',
    )
    if not root.is_dir() or any(not (root / rel).is_file() for rel in required):
        raise ValueError('Expected a complete local SapiensID v1 source checkout.')
    try:
        head = subprocess.run(
            ['git', '-C', str(root), 'rev-parse', 'HEAD'], timeout=10,
            capture_output=True, text=True, check=True).stdout.strip()
        git_root = subprocess.run(
            ['git', '-C', str(root), 'rev-parse', '--show-toplevel'],
            timeout=10, capture_output=True, text=True, check=True).stdout.strip()
        # A pinned parent Git repository is not enough: the exact upstream
        # files must actually be tracked in that checkout, not ignored copies.
        subprocess.run(
            ['git', '-C', str(root), 'ls-files', '--error-unmatch', '--', *required],
            timeout=10, capture_output=True, text=True, check=True)
        status = subprocess.run(
            ['git', '-C', str(root), 'status', '--porcelain', '--untracked-files=all'],
            timeout=10, capture_output=True, text=True, check=True).stdout
    except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired) as error:
        raise ValueError('Cannot verify local SapiensID source revision/cleanliness.') from error
    if head != UPSTREAM_COMMIT or Path(git_root).resolve() != root or status.strip():
        raise ValueError('SapiensID source checkout must be pinned to the audited commit and clean.')
    return root


@contextmanager
def _source_imports(root):
    """Support upstream absolute 'src' imports without replacing another src.

    This is *not* a security sandbox: checkout must be trusted Python source.
    """
    task_root = root / 'tasks' / 'sapiensID'
    src_root = task_root / 'src'
    existing = sys.modules.get('src')
    if existing is not None:
        paths = list(getattr(existing, '__path__', ()))
        if len(paths) != 1 or Path(paths[0]).resolve() != src_root.resolve():
            raise RuntimeError("The Python module 'src' already belongs to another project.")
    previous = list(sys.path)
    sys.path[:0] = [str(task_root), str(root)]
    try:
        yield
    finally:
        sys.path[:] = previous


def _upstream_modules(root):
    """Import official constructors from an explicitly trusted checkout."""
    with _source_imports(root):
        try:
            models = import_module('src.models.sapiens_id')
            aligners = import_module('src.aligners.keypoint_predictor')
            dfa = import_module('src.aligners.keypoint_predictor.face_detector.dfa')
            from ultralytics import YOLO
            from omegaconf import OmegaConf
        except ImportError as error:
            raise RuntimeError(
                'SapiensID optional upstream dependencies are unavailable; '
                'use a compatible isolated environment. No packages were installed.'
            ) from error
    return models.SapiensModel, aligners.KeypointPredictor, dfa, YOLO, OmegaConf


def _load_dfa_weights(dfa_module, path, *, sha256):
    """Construct upstream DFA network without calling its unsafe torch.load."""
    import torch
    _sha256(sha256, 'dfa_sha256')
    net, prior = dfa_module.get_landmark_predictor(
        network='mobilenetv4', use_aggregator=True, input_size=160)
    preprocessor = dfa_module.get_preprocessor(output_size=160, padding=0, padding_val='zero')
    if Path(path).is_symlink():
        raise ValueError('DFA checkpoint symlinks are not accepted.')
    with Path(path).open('rb') as stream:
        before = os.fstat(stream.fileno())
        if not stat.S_ISREG(before.st_mode) or not 0 < before.st_size <= _DFA_BYTES_MAX:
            raise ValueError('DFA checkpoint has invalid size/type.')
        sha = hashlib.sha256()
        size = 0
        for block in iter(lambda: stream.read(1048576), b''):
            sha.update(block)
            size += len(block)
        after = os.fstat(stream.fileno())
        if (size != before.st_size or size != after.st_size or sha.hexdigest() != sha256):
            raise ValueError('DFA checkpoint SHA256 mismatch.')
        stream.seek(0)
        try:
            state = torch.load(stream, map_location='cpu', weights_only=True)
        except Exception as error:
            raise ValueError('DFA checkpoint requires unsafe deserialization; rejected.') from error
    if not isinstance(state, dict) or not state or any(
            type(k) is not str or not isinstance(v, torch.Tensor) for k, v in state.items()):
        raise ValueError('DFA checkpoint must be a state dict of named tensors.')
    net.load_state_dict(state, strict=True)
    net.requires_grad_(False)
    return dfa_module.FaceKPPredictor(net, prior, preprocessor)


def load_local_encoder(*, checkout, checkpoint_dir, model_yaml_sha256,
                       yolo_pose_path, yolo_pose_sha256, dfa_path, dfa_sha256,
                       device='auto', max_batch_size=4,
                       model_size=WB4M_BYTES, model_sha256=WB4M_SHA256):
    """Load official v1 PyTorch objects from pinned local source and weights.

    The model checkpoint must match its model.yaml. This function cannot infer
    people directly from a VTT sheet: use one already isolated person crop.
    The optional upstream packages are NOT pulled into the JoyTag requirements.

    YOLO's own loader may use pickle-based PyTorch serialization; only supply
    YOLO weights from a trusted publisher with independently checked SHA256.
    """
    root = _trusted_checkout(checkout)
    directory = Path(checkpoint_dir).expanduser().resolve(strict=True)
    _sha256(model_sha256, 'model_sha256')
    _sha256(model_yaml_sha256, 'model_yaml_sha256')
    _sha256(yolo_pose_sha256, 'yolo_pose_sha256')
    _sha256(dfa_sha256, 'dfa_sha256')
    if not directory.is_dir():
        raise ValueError('SapiensID checkpoint_dir must be a directory.')
    model_path = directory / 'model.pth'
    model_yaml = directory / 'model.yaml'
    # Mandatory preflight before importing any third-party upstream code.
    yaml_bytes = _read_checked(model_yaml, model_yaml_sha256,
                               max_bytes=_IMAGE_BYTES_MAX, name='model.yaml')
    # The original model can be large; the offline adapter rechecks + loads
    # its source descriptor with torch.load(weights_only=True).
    from .sapiensid_v1 import verify_wb4m_checkpoint
    verify_wb4m_checkpoint(model_path, expected_size=model_size, expected_sha256=model_sha256)
    _read_checked(dfa_path, dfa_sha256, max_bytes=_DFA_BYTES_MAX, name='DFA checkpoint')

    with tempfile.TemporaryDirectory(prefix='stash-sapiensid-v1-') as temp:
        # Only the verified temporary copy is ever opened by Ultralytics.
        pose_copy = Path(temp) / 'yolov8n-pose.pt'
        _read_checked(yolo_pose_path, yolo_pose_sha256,
                      max_bytes=_POSE_BYTES_MAX, name='YOLO pose',
                      destination=pose_copy)
        Model, Aligner, dfa, YOLO, OmegaConf = _upstream_modules(root)
        try:
            cfg = OmegaConf.load(io.StringIO(yaml_bytes.decode('utf-8-sig')))
        except (UnicodeError, ValueError) as error:
            raise ValueError('Invalid local SapiensID model.yaml.') from error
        if (cfg.name != 'mrm' or list(cfg.input_size) != [384, 384]
                or cfg.output_dim != 4096 or cfg.color_space != 'RGB'
                or cfg.kp_preprocessor.name != 'kp19'):
            raise ValueError('Unsupported SapiensID v1 model.yaml architecture.')
        # Override any startup weights, disabling hidden eager model loading.
        cfg.start_from = ''
        cfg.pretrained_path = ''
        cfg.freeze = False
        model = Model.from_config(cfg)
        load_verified_weights(model, model_path,
                              expected_size=model_size, expected_sha256=model_sha256)
        aligner_config = OmegaConf.create({
            'name': 'keypoint_predictor',
            'start_from': '', 'freeze': True,
            'rgb_mean': [0.5, 0.5, 0.5], 'rgb_std': [0.5, 0.5, 0.5],
            'use_yolo': False, 'use_openpose': False, 'face_detector': None,
            'faceness_threshold': 0.5, 'use_aligned_face_template': False,
            'use_cache': False,
        })
        # Prevent KeypointPredictor.__init__ invoking its torch.load(path)
        # without weights_only. Inject the safely loaded DFA model ourselves.
        aligner = Aligner.from_config(aligner_config)
        aligner.face_predictor = _load_dfa_weights(dfa, dfa_path, sha256=dfa_sha256)
        aligner.faceness_threshold = 0.5
        aligner.yolo_net = YOLO(str(pose_copy))
        aligner.config.face_detector = 'dfa_mobilenetv4'
        aligner.config.use_yolo = True
        aligner.eval()
        return SapiensIDV1Encoder(model, aligner, device=device, max_batch_size=max_batch_size)


def main(argv=None):
    """Explicit local one-person smoke test; never invokes Stash APIs."""
    import argparse
    import json
    from PIL import Image

    parser = argparse.ArgumentParser(
        description='Run SapiensID v1 on a single already-cropped person image (offline).')
    parser.add_argument('--checkout', required=True,
                        help='Clean official source checkout at the audited commit.')
    parser.add_argument('--checkpoint-dir', required=True,
                        help='Directory containing wb4m model.yaml and model.pth.')
    parser.add_argument('--model-yaml-sha256', required=True)
    parser.add_argument('--yolo-pose-path', required=True)
    parser.add_argument('--yolo-pose-sha256', required=True)
    parser.add_argument('--dfa-path', required=True)
    parser.add_argument('--dfa-sha256', required=True)
    parser.add_argument('--image', required=True,
                        help='Local RGB image containing a single person (no detection done).')
    parser.add_argument('--device', choices=('auto','cpu','cuda'), default='auto')
    parser.add_argument('--batch-size', type=int, default=1)
    args = parser.parse_args(argv)

    path = Path(args.image)
    if not path.is_file() or path.is_symlink() or path.stat().st_size > 50 * 1024 * 1024:
        parser.error('Image must be a local regular file of at most 50 MiB.')
    encoder = load_local_encoder(
        checkout=args.checkout, checkpoint_dir=args.checkpoint_dir,
        model_yaml_sha256=args.model_yaml_sha256,
        yolo_pose_path=args.yolo_pose_path, yolo_pose_sha256=args.yolo_pose_sha256,
        dfa_path=args.dfa_path, dfa_sha256=args.dfa_sha256,
        device=args.device, max_batch_size=args.batch_size)
    try:
        with Image.open(path) as source:
            if source.width * source.height > 16 * 1024 * 1024:
                raise ValueError('SapiensID smoke test image exceeds 16 megapixels.')
            feature = encoder.embed(source)
        import numpy as np
        print(json.dumps({
            'model': 'sapiensid_v1', 'embedding_dimensions': int(feature.shape[0]),
            'unit_norm': bool(abs(float(np.linalg.norm(feature.astype(np.float64))) - 1.) < 0.001),
            'device': str(encoder.device), 'published_to_stash': False,
        }, sort_keys=True))
    finally:
        encoder.close()


if __name__ == '__main__':
    main()
