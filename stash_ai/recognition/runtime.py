"""One application ORT distribution and shared CUDA library/provider policy."""
from importlib import metadata
from types import SimpleNamespace

from stash_ai.backends.onnx_session import _execution_providers


def runtime(config):
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
    try:
        providers, _ = _execution_providers(ort, SimpleNamespace(device=config.data['models']['device']))
    except Exception:
        if config.data['models']['device'] != 'auto' or 'CUDAExecutionProvider' not in available:
            raise
        providers = ['CPUExecutionProvider']
    return ort, providers


def session(config, model, *, runtime_info=None, cpu_fallback=True):
    ort, providers = runtime_info or runtime(config)
    options = ort.SessionOptions()
    options.intra_op_num_threads = options.inter_op_num_threads = 1
    if config.data['models']['device'] == 'cuda' and not cpu_fallback:
        # Expression models contain CUDA-supported float operators. Refuse a
        # partially CPU-assigned graph as well as a CPU-only provider fallback.
        options.add_session_config_entry('session.disable_cpu_ep_fallback', '1')
        providers = [item for item in providers if item != 'CPUExecutionProvider']
    try:
        result = ort.InferenceSession(model, sess_options=options, providers=providers)
    except Exception:
        if config.data['models']['device'] != 'auto' or 'CUDAExecutionProvider' not in providers:
            raise
        result = ort.InferenceSession(model, sess_options=options, providers=['CPUExecutionProvider'])
    active = result.get_providers()
    device = 'cuda' if active and active[0] == 'CUDAExecutionProvider' else 'cpu'
    if config.data['models']['device'] == 'cuda' and device != 'cuda':
        result = None
        raise ValueError('Recognition CUDA initialization fell back to CPU; requested GPU was not met. Check CUDA/cuDNN libraries.')
    return result, device
