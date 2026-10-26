"""Streaming, hash-verified installation of fixed upstream assets."""
import hashlib
import os
from pathlib import Path
import tempfile
import zlib

from .progress import work


def _selected(chunks, skip, length, progress):
    remaining = length
    for chunk in chunks:
        progress.advance(len(chunk))
        if skip:
            count = min(skip, len(chunk))
            skip -= count
            chunk = chunk[count:]
        if chunk:
            part = chunk[:remaining]
            remaining -= len(part)
            yield part
        if not skip and not remaining:
            return
    raise ValueError('Incomplete SapiensID download.')


def _inflate(chunks):
    decoder = zlib.decompressobj(-15)
    for chunk in chunks:
        while chunk:
            yield decoder.decompress(chunk, 1048576)
            chunk = decoder.unconsumed_tail
        if decoder.unused_data:
            raise ValueError('Unexpected data after SapiensID ZIP entry.')
    if not decoder.eof:
        raise ValueError('Incomplete SapiensID ZIP entry.')


def install(url, destination, size, checksum, *, byte_range=None, archive_size=None):
    """Never expose partial or unverified bytes at the configured destination."""
    import requests
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    headers = {'Accept-Encoding': 'identity'}
    offset, length = byte_range or (0, size)
    if byte_range:
        headers['Range'] = f'bytes={offset}-{offset+length-1}'
    with requests.get(url, headers=headers, stream=True, timeout=(15, 60)) as response:
        response.raise_for_status()
        if response.status_code not in (200, 206):
            raise ValueError('Unexpected SapiensID download response.')
        skip = offset if byte_range and response.status_code == 200 else 0
        if response.status_code == 206:
            expected = f'bytes {offset}-{offset+length-1}/{archive_size or size}'
            if response.headers.get('Content-Range') != expected:
                raise ValueError('SapiensID server returned a different byte range.')
        # Some proxies ignore Range: stream past the prefix without retaining it.
        # Hash validation still applies, and the connection closes after this entry.
        with work('Download SapiensID '+destination.name, total=skip+length, unit='byte') as progress:
            chunks = _selected(response.iter_content(1048576), skip, length, progress)
            if byte_range:
                chunks = _inflate(chunks)
            descriptor, temporary = tempfile.mkstemp(prefix=destination.name+'.', suffix='.part', dir=destination.parent)
            try:
                digest, consumed = hashlib.sha256(), 0
                with os.fdopen(descriptor, 'wb') as target:
                    for chunk in chunks:
                        consumed += len(chunk)
                        if consumed > size:
                            raise ValueError('SapiensID asset exceeds its pinned size.')
                        digest.update(chunk)
                        target.write(chunk)
                    if consumed != size or digest.hexdigest() != checksum:
                        raise ValueError('SapiensID download size/SHA256 mismatch.')
                    target.flush()
                    os.fsync(target.fileno())
                os.replace(temporary, destination)
            finally:
                Path(temporary).unlink(missing_ok=True)
