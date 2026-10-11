"""Atomic installation from the existing pinned Hugging Face file cache."""
import hashlib
import os
from pathlib import Path
import tempfile


def install_hf_asset(repo, filename, revision, destination, *, expected_sha256=None, max_bytes=None):
    from huggingface_hub import hf_hub_download
    cached = Path(hf_hub_download(repo_id=repo, filename=filename, revision=revision))
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=destination.name + '.', suffix='.part', dir=destination.parent)
    try:
        with os.fdopen(fd, 'wb') as target, cached.open('rb') as source:
            digest, size = hashlib.sha256(), 0
            while chunk := source.read(1024 * 1024):
                size += len(chunk)
                if max_bytes is not None and size > max_bytes:
                    raise ValueError('Hugging Face asset exceeds its pinned size.')
                target.write(chunk)
                digest.update(chunk)
            if expected_sha256 is not None and digest.hexdigest() != expected_sha256:
                raise ValueError('Hugging Face asset checksum does not match the pinned registry.')
            target.flush()
            os.fsync(target.fileno())
        os.replace(temporary, destination)
    finally:
        Path(temporary).unlink(missing_ok=True)
