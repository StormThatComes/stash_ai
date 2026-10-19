"""Integrity verification covers loader inputs, including a replaced source path."""
from dataclasses import replace
import hashlib
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from stash_ai.recognition import models
from stash_ai.recognition.config import load


def asset(path, data):
    path.write_bytes(data)
    return models.Asset('fixture', 'revision', path.name, hashlib.sha256(data).hexdigest(), len(data), 'MIT')


def test_verified_bytes_are_immutable_after_same_size_path_replacement(tmp_path):
    path = tmp_path/'weights'
    expected = asset(path, b'pinned weights')
    blob = models.verified_bytes(path, expected)
    alternate = tmp_path/'alternate'
    alternate.write_bytes(b'custom weights')
    alternate.replace(path)
    assert blob == b'pinned weights'
    with pytest.raises(ValueError, match='checksum'): models.verified_bytes(path, expected)
    assert path.read_bytes() == b'custom weights'


def explicit_yunet(tmp_path):
    from stash_ai.recognition.config import load
    yaml = tmp_path / 'yunet.yml'
    yaml.write_text('version: 1\\nmodels: {detector: yunet, device: cpu}\\n')
    return load(yaml)


def test_cpu_loader_consumes_verified_buffer_and_not_the_replaced_path(tmp_path, monkeypatch):
    path = tmp_path/'yunet'
    monkeypatch.setitem(models.ASSETS, 'yunet', asset(path, b'pinned'))
    inputs = []
    def create(framework, buffer, config, *args):
        path.write_bytes(b'custom')
        inputs.append((framework, buffer.tobytes()))
        return SimpleNamespace()
    cv = SimpleNamespace(__version__='fixture', FaceDetectorYN=SimpleNamespace(create=create))
    monkeypatch.setitem(sys.modules, 'cv2', cv)
    backend = models.YuNetDetector(explicit_yunet(tmp_path), {'yunet':path})
    backend.close()
    assert inputs == [('onnx', b'pinned')] and path.read_bytes() == b'custom'


def test_swap_after_path_verification_cannot_reach_cpu_loader(tmp_path, monkeypatch):
    path = tmp_path/'yunet'
    expected = asset(path, b'pinned')
    monkeypatch.setitem(models.ASSETS, 'yunet', expected)
    original = models.verify
    def swapped(path, expected):
        original(path, expected)
        path.write_bytes(b'custom')
    monkeypatch.setattr(models, 'verify', swapped)
    monkeypatch.setitem(sys.modules, 'cv2', SimpleNamespace(
        FaceDetectorYN=SimpleNamespace(create=lambda *args: pytest.fail('unverified bytes reached loader'))))
    with pytest.raises(ValueError, match='checksum'):
        models.YuNetDetector(load(root=tmp_path, device='cpu'), {'yunet':path})


def test_onnx_declaration_adaptation_parses_exact_verified_bytes(tmp_path, monkeypatch):
    import onnx
    from stash_ai.recognition import onnx_models
    model = onnx.helper.make_model(onnx.helper.make_graph([], 'fixture', [], []))
    data = model.SerializeToString()
    path = tmp_path/'sface'
    monkeypatch.setitem(models.ASSETS, 'sface', asset(path, data))
    original = onnx.load_model_from_string
    inputs = []
    def parse(blob):
        path.write_bytes(b'X'*len(data))
        inputs.append(blob)
        return original(blob)
    monkeypatch.setattr(onnx, 'load_model_from_string', parse)
    prepared, sha = onnx_models.prepared_model(path)
    assert inputs == [data] and hashlib.sha256(prepared).hexdigest() == sha
    assert path.read_bytes() == b'X'*len(data)


@pytest.mark.parametrize('dimensions', [128,512])
def test_large_finite_embedding_normalizes_without_float32_norm_overflow(dimensions):
    vector = models.normalized_embedding(np.full(dimensions, 1e20, dtype=np.float32), dimensions)
    assert vector.dtype == np.float32 and np.isfinite(vector).all()
    assert np.linalg.norm(vector.astype(np.float64)) == pytest.approx(1, abs=1e-6)
    # The normalized result must pass the same retained-evidence boundary.
    from stash_ai.recognition.clustering import vector as retained
    assert np.array_equal(retained({'dimensions':dimensions, 'embedding':vector.tobytes()}), vector)


def test_embedding_cast_overflow_is_rejected_before_persistence():
    with np.errstate(over='ignore'), pytest.raises(ValueError, match='non-finite'):
        models.normalized_embedding(np.full(128, 1e300), 128)
