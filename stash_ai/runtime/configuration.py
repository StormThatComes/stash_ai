"""One immutable file-byte snapshot and its hashes for a command invocation."""
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
import hashlib
from pathlib import Path
from types import MappingProxyType

_current = ContextVar('effective_configuration', default=None)
FILES = ('model_config', 'temporal_policy', 'hierarchy', 'marker_core_exclusions', 'scene_tag_exclusions')


@dataclass(frozen=True)
class EffectiveConfiguration:
    files: object
    hashes: object
    parsed: dict = field(default_factory=dict, compare=False, repr=False)

    @classmethod
    def capture(cls, args):
        files, hashes = {}, {}
        for name in FILES:
            path = Path(getattr(args, name)).expanduser().resolve()
            if path not in files:
                try:
                    files[path] = path.read_bytes()
                except OSError as error:
                    files[path] = error
            value = files[path]
            hashes[name] = hashlib.sha256(value).hexdigest() if isinstance(value, bytes) else None
        return cls(MappingProxyType(files), MappingProxyType(hashes))

    @contextmanager
    def use(self):
        token = _current.set(self)
        try:
            yield self
        finally:
            _current.reset(token)


def read_bytes(path):
    path = Path(path).expanduser().resolve()
    snapshot = _current.get()
    if snapshot is not None and path in snapshot.files:
        value = snapshot.files[path]
        if isinstance(value, OSError):
            raise value
        return value
    return path.read_bytes()


def is_file(path):
    snapshot = _current.get()
    resolved = Path(path).expanduser().resolve()
    return isinstance(snapshot.files[resolved], bytes) if snapshot is not None and resolved in snapshot.files else resolved.is_file()


def cached_yaml(path, context, load):
    from copy import deepcopy
    snapshot = _current.get()
    if snapshot is None:
        return load()
    key = (Path(path).expanduser().resolve(), context)
    if key not in snapshot.parsed:
        snapshot.parsed[key] = load()
    return deepcopy(snapshot.parsed[key])
