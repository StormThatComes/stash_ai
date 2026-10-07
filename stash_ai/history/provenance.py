"""Code and package provenance without importing inference backends."""
import hashlib
from importlib import metadata
from pathlib import Path
import subprocess

PACKAGES = ('numpy', 'Pillow', 'requests', 'torch', 'torchvision', 'timm',
            'transformers', 'onnxruntime', 'onnxruntime-gpu')


def capture(root):
    root = Path(root)
    code = hashlib.sha256()
    files = sorted((root / 'stash_ai').rglob('*.py'))
    for path in files:
        code.update(path.relative_to(root).as_posix().encode())
        with path.open('rb') as stream:
            code.update(hashlib.file_digest(stream, 'sha256').digest())
    revision = None
    if (root / '.git').exists():
        try:
            result = subprocess.run(['git', '-C', str(root), 'rev-parse', 'HEAD'],
                                    capture_output=True, text=True, timeout=1)
            if result.returncode == 0:
                revision = result.stdout.strip()
        except (OSError, subprocess.TimeoutExpired):
            pass
    packages = {}
    for name in PACKAGES:
        try:
            packages[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            packages[name] = None
    return dict(code_sha256=code.hexdigest() if files else None, revision=revision,
                packages=packages, package_versions_observed=True)
