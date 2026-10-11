import hashlib
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from stash_ai.recognition.config import load
from stash_ai.recognition.models import Asset, checked_detection, normalized_embedding, verify, namespace


def test_face_boundaries_reject_nonfinite_shapes():
    with pytest.raises(ValueError):
        checked_detection([1] * 14)
    with pytest.raises(ValueError):
        checked_detection([float('nan')] * 15)
    for bad in (np.zeros(128), np.ones(127), [float('inf')] * 128):
        with pytest.raises(ValueError):
            normalized_embedding(bad, 128)
    assert np.linalg.norm(normalized_embedding(np.ones((1, 128)), 128)) == pytest.approx(1)


def test_pinned_cache_installs_atomically_and_keeps_custom_files(tmp_path, monkeypatch):
    from stash_ai.backends.downloads import install_hf_asset
    source = tmp_path / 'cache'
    source.write_bytes(b'weights')
    calls = []
    def download(**kwargs):
        calls.append(kwargs)
        return str(source)
    monkeypatch.setitem(sys.modules, 'huggingface_hub', SimpleNamespace(hf_hub_download=download))
    target = tmp_path / 'models' / 'model.onnx'
    sha = hashlib.sha256(b'weights').hexdigest()
    install_hf_asset('owner/model', 'weights.onnx', 'a' * 40, target, expected_sha256=sha, max_bytes=7)
    assert target.read_bytes() == b'weights'
    assert calls == [dict(repo_id='owner/model', filename='weights.onnx', revision='a' * 40)]
    with pytest.raises(ValueError):
        install_hf_asset('owner/model', 'weights.onnx', 'a' * 40, target, expected_sha256='0' * 64)
    assert target.read_bytes() == b'weights'
    assert not list(target.parent.glob('*.part'))
    verify(target, Asset('r', 's', 'model.onnx', sha, 7, 'MIT'))
    target.write_bytes(b'custom!')
    with pytest.raises(ValueError):
        verify(target, Asset('r', 's', 'model.onnx', sha, 7, 'MIT'))
    assert target.read_bytes() == b'custom!'


def test_model_namespace_ignores_tuning_and_asset_location(tmp_path):
    one = load(root=tmp_path)
    p = tmp_path / 'config.yml'
    p.write_text('version: 1\nrecognition: {match_threshold: 0.8}\nmodels: {directory: other}\n')
    assert namespace(one) == namespace(load(p))
