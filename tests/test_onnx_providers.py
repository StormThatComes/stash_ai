"""Provider selection and CUDA library setup without models or a GPU."""
import sys
from types import SimpleNamespace

import pytest

from stash_ai.onnx_tagger import _execution_providers


@pytest.mark.parametrize('device,available,use_cuda', [
    ('auto', ['CUDAExecutionProvider', 'CPUExecutionProvider'], True),
    ('cuda', ['CUDAExecutionProvider', 'CPUExecutionProvider'], True),
    ('cpu', ['CUDAExecutionProvider', 'CPUExecutionProvider'], False),
    ('auto', ['CPUExecutionProvider'], False),
    ('cpu', ['CPUExecutionProvider'], False),
])
def test_preload_libraries_only_when_cuda_is_selected(monkeypatch, device, available, use_cuda):
    monkeypatch.setitem(sys.modules, 'torch', None)
    events = []
    ort = SimpleNamespace(get_available_providers=lambda: available,
                          preload_dlls=lambda: events.append('preload'))

    providers, selected_cuda = _execution_providers(ort, SimpleNamespace(device=device))

    assert selected_cuda is use_cuda
    assert providers == (['CUDAExecutionProvider', 'CPUExecutionProvider']
                         if use_cuda else ['CPUExecutionProvider'])
    assert events == (['preload'] if use_cuda else [])


def test_explicit_cuda_request_fails_before_preloading_when_provider_is_absent():
    ort = SimpleNamespace(get_available_providers=lambda: ['CPUExecutionProvider'],
                          preload_dlls=lambda: pytest.fail('Unavailable CUDA was preloaded'))

    with pytest.raises(RuntimeError, match='CUDA provider unavailable'):
        _execution_providers(ort, SimpleNamespace(device='cuda'))


def test_cuda_preload_error_is_not_silently_ignored(monkeypatch):
    monkeypatch.setitem(sys.modules, 'torch', None)

    def preload():
        raise OSError('CUDA library could not be loaded')

    ort = SimpleNamespace(get_available_providers=lambda: ['CUDAExecutionProvider'],
                          preload_dlls=preload)

    with pytest.raises(OSError, match='CUDA library could not be loaded'):
        _execution_providers(ort, SimpleNamespace(device='auto'))
