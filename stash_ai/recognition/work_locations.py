"""Filesystem locations owned by one recognition invocation and its children."""
from contextlib import contextmanager
import os
from pathlib import Path
import sys
import tempfile
from threading import RLock

_lock = RLock()


def configured_path(config, value):
    base = config.source_path.parent if getattr(config, 'source_path', None) else Path.cwd()
    return (base / Path(value).expanduser()).resolve()


def work_root(config):
    value = config.data['execution'].get('rebuild_scratch_directory')
    root = configured_path(config, value) if value else Path(config.database).parent / 'recognition_work'
    root.mkdir(parents=True, exist_ok=True)
    return root.resolve()


def validate_outputs(config, paths):
    roots = {work_root(config), Path(config.database).parent.resolve()}
    output = config.data['storage'].get('output_directory')
    if output:
        roots.add(configured_path(config, output))
    for value in filter(None, paths):
        path = Path(value).expanduser().resolve()
        if not any(path.is_relative_to(root) for root in roots):
            raise ValueError(f'Solver output {path} is outside configured storage directories: {sorted(map(str, roots))}')


@contextmanager
def invocation(config):
    """Set native temp/cache paths before SQLite and model runtimes start.

    Python caches its temp path. SQLite and spawned runtimes consult the
    environment independently, including TMP/TEMP on Windows.
    """
    from .disk_budget import scope as disk_scope, temporary
    from .scratch_space import scope as scratch_scope
    with _lock, disk_scope(config), scratch_scope(config) as budget:
        work = Path(budget.directory)
        cache = work_root(config) / 'runtime-cache'
        cache.mkdir(exist_ok=True)
        paths = dict.fromkeys(('TMPDIR', 'TMP', 'TEMP', 'SQLITE_TMPDIR'), str(work))
        paths.update(RECOGNITION_WORK_ROOT=str(work), XDG_CACHE_HOME=str(cache),
                     HF_HOME=str(cache/'huggingface'), TORCH_HOME=str(cache/'torch'),
                     YOLO_CONFIG_DIR=str(cache/'ultralytics'), CUDA_CACHE_PATH=str(cache/'cuda'),
                     TRITON_CACHE_DIR=str(cache/'triton'), TORCHINDUCTOR_CACHE_DIR=str(cache/'inductor'),
                     MPLCONFIGDIR=str(cache/'matplotlib'), PYTHONDONTWRITEBYTECODE='1')
        previous = {name: os.environ.get(name) for name in paths}
        previous_temp, previous_bytecode = tempfile.tempdir, sys.dont_write_bytecode
        try:
            os.environ.update(paths)
            tempfile.tempdir = str(work)
            sys.dont_write_bytecode = True
            with temporary(cache):
                yield
        finally:
            tempfile.tempdir, sys.dont_write_bytecode = previous_temp, previous_bytecode
            for name, value in previous.items():
                if value is None:
                    os.environ.pop(name, None)
                else:
                    os.environ[name] = value
