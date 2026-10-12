"""One application ORT distribution and shared CUDA library/provider policy."""
from importlib import import_module, metadata
import os
import shlex
import sys
from types import SimpleNamespace

from stash_ai.backends.onnx_session import _execution_providers


def check_dependencies(config):
    """Explain missing recognition packages before downloading weights/media."""
    required = [('cv2', 'opencv-python-headless>=4.10,<5')]
    if config.data['models']['device'] != 'cpu':
        required.append(('onnx', 'onnx>=1.21,<2'))
    missing = []
    for module, requirement in required:
        try:
            import_module(module)
        except ModuleNotFoundError as error:
            if error.name != module:
                raise  # A broken package's internal import is a different failure.
            missing.append((module, requirement))
    if missing:
        arguments = [sys.executable, '-m', 'pip', 'install', *(item[1] for item in missing)]
        if os.name == 'nt':
            # PowerShell needs the call operator for a quoted executable path.
            command = '& ' + ' '.join("'" + item.replace("'", "''") + "'" for item in arguments)
        else:
            command = shlex.join(arguments)
        modules = ', '.join(item[0] for item in missing)
        raise ValueError(f'Recognition dependencies missing: {modules}. '
                         'The code updater does not install Python packages. '
                         f'Install into the Python running this command: {command}')


def runtime(config):
    check_dependencies(config)
    variants = []
    for name in ('onnxruntime', 'onnxruntime-gpu'):
        try:
            metadata.version(name)
            variants.append(name)
        except metadata.PackageNotFoundError:
            pass
    if len(variants) != 1:
        raise ValueError('Recognition requires exactly one application ONNX Runtime variant: CPU or GPU; do not install both.')
    import onnxruntime as ort
    available = ort.get_available_providers()
    if config.data['models']['device'] == 'cuda' and 'CUDAExecutionProvider' not in available:
        raise ValueError('Recognition requested CUDA but this runtime has no CUDA provider; install the GPU profile or set models.device: cpu.')
    if 'CPUExecutionProvider' not in available:
        raise ValueError('Recognition requires the supported ONNX Runtime CPU fallback.')
    providers, _ = _execution_providers(ort, SimpleNamespace(device=config.data['models']['device']))
    return ort, providers


def session(config, model, *, runtime_info=None):
    ort, providers = runtime_info or runtime(config)
    options = ort.SessionOptions()
    options.intra_op_num_threads = options.inter_op_num_threads = 1
    try:
        result = ort.InferenceSession(model, sess_options=options, providers=providers)
    except Exception:
        if config.data['models']['device'] != 'auto' or 'CUDAExecutionProvider' not in providers:
            raise
        result = ort.InferenceSession(model, sess_options=options, providers=['CPUExecutionProvider'])
    active = result.get_providers()
    device = 'cuda' if active and active[0] == 'CUDAExecutionProvider' else 'cpu'
    if config.data['models']['device'] == 'cuda' and device != 'cuda':
        raise ValueError('Recognition CUDA initialization fell back to CPU; requested GPU was not met. Check CUDA/cuDNN libraries.')
    return result, device
