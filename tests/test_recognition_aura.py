from dataclasses import replace
import json
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from stash_ai.recognition.aura import input_blob, runtime, similarity_matrix
from stash_ai.recognition.config import load
from stash_ai.recognition.database import FaceDatabase
from stash_ai.recognition.models import ASSETS, ensure_assets, namespace
from tests.test_recognition_run import setup, run


def test_alignment_similarity_matches_known_affine_and_rejects_degeneracy():
    reference = np.array([[38.2946,51.6963],[73.5318,51.5014],[56.0252,71.7366],
                          [41.5493,92.3655],[70.7299,92.2041]])
    angle = np.deg2rad(17)
    rotation = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    source = (reference @ rotation.T) * 2 + [17, 23]
    matrix = similarity_matrix(source)
    assert np.allclose(source @ matrix[:, :2].T + matrix[:, 2], reference, atol=1e-4)
    with pytest.raises(ValueError, match='degenerate'):
        similarity_matrix(np.zeros((5, 2)))


def test_rgb_normalization_nchw_is_exact_and_has_no_upstream_normalization_twice():
    bgr = np.empty((112,112,3), dtype='uint8')
    bgr[:] = [0, 127, 255]
    blob = input_blob(bgr)
    assert blob.shape == (1,3,112,112) and blob.dtype == np.float32 and blob.flags.c_contiguous
    assert blob[0,:,0,0] == pytest.approx([1, (127-127.5)/127.5, -1])
    with pytest.raises(ValueError):
        input_blob(np.zeros((128,128,3), dtype='uint8'))


def test_only_recognition_weight_and_detector_are_downloaded(tmp_path, monkeypatch):
    import stash_ai.backends.downloads as downloads
    import stash_ai.recognition.aura as aura
    import stash_ai.recognition.models as models
    path = tmp_path / 'face.yml'
    path.write_text('version: 1\nmodels: {recognizer: auraface}\n')
    calls = []
    monkeypatch.setattr(aura, 'runtime', lambda config: None)
    monkeypatch.setattr(models, 'verify', lambda *args: None)
    monkeypatch.setattr(downloads, 'install_hf_asset', lambda repo, filename, revision, destination, **kwargs:
                        calls.append((repo, filename, revision)))
    ensure_assets(load(path), write=lambda text: None)
    assert [item[1] for item in calls] == [ASSETS['yunet'].filename, 'glintr100.onnx']
    assert all('gender' not in item[1] and '1k3' not in item[1] for item in calls)


def test_runtime_rejects_duplicate_packages_and_missing_cuda(tmp_path, monkeypatch):
    import stash_ai.recognition.aura as aura
    monkeypatch.setattr(aura.metadata, 'version', lambda name: '1.26.0')
    with pytest.raises(ValueError, match='exactly one'):
        runtime(load(root=tmp_path))
    def version(name):
        if name == 'onnxruntime-gpu':
            raise aura.metadata.PackageNotFoundError(name)
        return '1.26.0'
    monkeypatch.setattr(aura.metadata, 'version', version)
    monkeypatch.setitem(sys.modules, 'onnxruntime', SimpleNamespace(get_available_providers=lambda: ['CPUExecutionProvider']))
    path = tmp_path / 'face.yml'
    path.write_text('version: 1\nmodels: {recognizer: auraface, device: cuda}')
    with pytest.raises(ValueError, match='no CUDA'):
        runtime(load(path))


def test_model_switch_retains_evidence_and_profiles_without_duplicate_publication(tmp_path, monkeypatch):
    config, client, services, _ = setup(tmp_path, monkeypatch)
    assert run(config, services, '1') == 0
    writes = list(client.writes)
    path = tmp_path / 'aura.yml'
    path.write_text('version: 1\nmodels: {recognizer: auraface, device: cpu}\n'
                    'detection: {min_sharpness: 0}\nportraits: {enabled: false}\n')
    aura = replace(load(path), database=config.database)
    assert namespace(config) != namespace(aura)
    assert run(aura, services, '1') == 0
    assert client.writes == writes
    with FaceDatabase(config.database) as db:
        assert db.connection.execute('SELECT COUNT(*) FROM scans').fetchone()[0] == 2
        assert db.connection.execute('SELECT COUNT(DISTINCT namespace) FROM models').fetchone()[0] == 2
        assert db.connection.execute('SELECT COUNT(*) FROM people WHERE performer IS NOT NULL').fetchone()[0] == 1
        row = db.connection.execute("SELECT payload,status FROM publications WHERE kind='namespace'").fetchone()
        assert row['status'] == 'withheld' and json.loads(row['payload'])['namespace'] == namespace(aura)
        assert all('/1.' in url for url in client.media_calls)


def test_local_pilot_uses_same_crops_without_rewriting_published_membership(tmp_path, monkeypatch):
    from stash_ai.recognition.pilot import evaluate_observations
    from tests.helpers.recognition import Backend
    config, client, services, _ = setup(tmp_path, monkeypatch)
    assert run(config, services, '1') == 0
    with FaceDatabase(config.database) as db:
        before = [tuple(row) for row in db.connection.execute('SELECT * FROM members')]
        row = db.connection.execute('SELECT id,scope FROM runs LIMIT 1').fetchone()
        labels = {r[0]: 'fixture-A' for r in db.connection.execute('SELECT id FROM observations')}
        results = evaluate_observations(db, row['id'], {'sface': config}, labels,
                                        assets=lambda config: {}, backend=lambda *args: Backend())
        assert results['sface']['pair_false_merge'] == results['sface']['pair_false_miss_including_withheld'] == 0
        assert results['sface']['accepted_coverage'] == 3
        assert [tuple(row) for row in db.connection.execute('SELECT * FROM members')] == before
        assert db.connection.execute("SELECT data FROM progress WHERE phase='pilot_comparison'").fetchone()
