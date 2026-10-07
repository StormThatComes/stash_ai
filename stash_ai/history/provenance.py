"""Code and package provenance without backends or external processes."""
import hashlib
from importlib import metadata
from pathlib import Path
import re

PACKAGES = ('numpy', 'Pillow', 'requests', 'torch', 'torchvision', 'timm',
            'transformers', 'onnxruntime', 'onnxruntime-gpu')


def git_revision(root):
    git = Path(root) / '.git'
    if git.is_file():
        pointer = git.read_text().strip()
        if not pointer.startswith('gitdir: '):
            return None
        git = (Path(root) / pointer[8:]).resolve()
    if not git.is_dir():
        return None
    head = (git / 'HEAD').read_text().strip()
    if head.startswith('ref: '):
        ref = head[5:]
        common = git
        if (git / 'commondir').is_file():
            common = (git / (git / 'commondir').read_text().strip()).resolve()
        for directory in dict.fromkeys((git, common)):
            path = directory / ref
            if path.is_file():
                head = path.read_text().strip()
                break
            packed = directory / 'packed-refs'
            if packed.is_file():
                match = next((line.split()[0] for line in packed.read_text().splitlines()
                              if len(line.split()) == 2 and line.split()[1] == ref), None)
                if match:
                    head = match
                    break
    return head if re.fullmatch(r'[0-9a-fA-F]{40}|[0-9a-fA-F]{64}', head) else None


def capture(root):
    root = Path(root)
    code = hashlib.sha256()
    files = sorted((root / 'stash_ai').rglob('*.py'))
    for path in files:
        code.update(path.relative_to(root).as_posix().encode())
        with path.open('rb') as stream:
            code.update(hashlib.file_digest(stream, 'sha256').digest())
    try:
        revision = git_revision(root)
    except (OSError, ValueError):
        revision = None
    packages = {}
    for name in PACKAGES:
        try:
            packages[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            packages[name] = None
    return dict(code_sha256=code.hexdigest() if files else None, revision=revision,
                packages=packages, package_versions_observed=True)
