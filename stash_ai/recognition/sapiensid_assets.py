"""Prepare missing assets in configured folders, only on inference paths."""
import errno
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import tempfile
import zipfile

from . import sapiensid_registry as registry
from .sapiensid_downloads import install

MARKER = '.stash-sapiensid-source'


def source_digest(root):
    rows = []
    for path in sorted(root.rglob('*')):
        relative = path.relative_to(root)
        if path.is_symlink():
            raise ValueError('SapiensID source contains a symbolic link.')
        if '__pycache__' in relative.parts or relative.as_posix() == MARKER:
            continue
        if path.is_file():
            rows.append((relative.as_posix(), hashlib.sha256(path.read_bytes()).hexdigest()))
    return hashlib.sha256(json.dumps(sorted(rows), separators=(',', ':')).encode()).hexdigest()


def verify_source(root):
    if (root / MARKER).read_text() != registry.UPSTREAM_COMMIT or source_digest(root) != registry.SOURCE_TREE_SHA256:
        raise ValueError('Downloaded SapiensID source has changed; restore the pinned source directory.')
    return root


def _require_download(config, path):
    if not config.data['models']['download_missing']:
        raise FileNotFoundError(f'Missing SapiensID asset {path}; models.download_missing is disabled.')


def _path(config, value, fallback):
    from .sapiensid_loading import local_path
    return local_path(config, value) if value.strip() else config.model_dir / 'sapiensid_v1' / fallback


def ensure_source(config, path):
    from .sapiensid_loading import trusted_checkout
    if path.exists() and (not path.is_dir() or any(path.iterdir())):
        return trusted_checkout(path)
    _require_download(config, path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='.sapiensid-source-', dir=path.parent) as temporary:
        temporary = Path(temporary)
        archive, staging = temporary / 'source.zip', temporary / 'source'
        staging.mkdir()
        install(registry.SOURCE_URL, archive, registry.SOURCE_BYTES, registry.SOURCE_SHA256)
        with zipfile.ZipFile(archive) as zipped:
            for info in zipped.infolist():
                parts = PurePosixPath(info.filename).parts
                if (not parts or parts[0] != 'sapiensid-'+registry.UPSTREAM_COMMIT
                        or any(p in ('.','..') or '\\' in p or ':' in p for p in parts)
                        or (info.external_attr >> 16) & 0o170000 == 0o120000):
                    raise ValueError('Unsafe SapiensID source archive entry.')
                target = staging.joinpath(*parts[1:])
                if info.is_dir():
                    target.mkdir(parents=True, exist_ok=True)
                else:
                    target.parent.mkdir(parents=True, exist_ok=True)
                    with zipped.open(info) as source, target.open('xb') as output:
                        shutil.copyfileobj(source, output, 1048576)
        if source_digest(staging) != registry.SOURCE_TREE_SHA256:
            raise ValueError('SapiensID source tree checksum mismatch.')
        (staging / MARKER).write_text(registry.UPSTREAM_COMMIT)
        if path.exists():
            if any(path.iterdir()):
                return trusted_checkout(path)  # Another installer completed first.
            path.rmdir()  # Only an empty destination may be replaced.
        try:
            os.rename(staging, path)
        except OSError as error:
            if error.errno not in (errno.EEXIST, errno.ENOTEMPTY):
                raise
            return trusted_checkout(path)
    return trusted_checkout(path)


def encoder_assets(config):
    from .sapiensid_v1 import WB4M_BYTES, WB4M_SHA256
    settings = config.data['sapiensid']['model']
    source = _path(config, settings['source_directory'], 'source')
    directory = _path(config, settings['checkpoint_directory'], 'wb4m')
    source = ensure_source(config, source)
    for name, size, checksum in (('model.yaml', 1198, settings['model_yaml_sha256']),
                                  ('model.pth', WB4M_BYTES, WB4M_SHA256)):
        path = directory / name
        if not path.is_file():
            _require_download(config, path)
            if name == 'model.yaml' and checksum != registry.MODEL_YAML_SHA256:
                raise FileNotFoundError('Custom SapiensID model.yaml hash requires an existing matching local file.')
            install(registry.CHECKPOINT_URL, path, size, checksum,
                    byte_range=registry.CHECKPOINT_RANGES[name], archive_size=registry.CHECKPOINT_BYTES)
    return source, directory


def pose_asset(config):
    settings = config.data['sapiensid']['pose']
    path = _path(config, settings['path'], 'yolov8n-pose.pt')
    checksum = settings['sha256'] or registry.POSE_SHA256
    if not path.is_file():
        _require_download(config, path)
        if checksum != registry.POSE_SHA256:
            raise FileNotFoundError('Custom SapiensID pose hash requires an existing matching local checkpoint.')
        install(registry.POSE_URL, path, registry.POSE_BYTES, checksum)
    return path, checksum
