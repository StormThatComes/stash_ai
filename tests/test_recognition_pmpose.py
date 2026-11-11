"""PMPose contracts, bounded batches, probability semantics and safe loading."""
from dataclasses import replace
import hashlib
from types import SimpleNamespace

import numpy as np
from PIL import Image
import pytest

from stash_ai.recognition.config import defaults, load, plain, validate
from stash_ai.recognition import pmpose_pose, pmpose_registry as registry


def configuration(tmp_path):
    path = tmp_path / 'recognition.yml'
    path.write_text('version: 1\nmodels: {device: cpu}\nsapiensid:\n  pose: {backend: pmpose}\n')
    config = load(path)
    return replace(config, data=plain(config.data))


def fingerprints(data):
    from stash_ai.recognition.person_signal import contract
    from stash_ai.recognition.body_observations import contract as body_contract
    from stash_ai.recognition.sapiensid_backend import identity_contract
    config = SimpleNamespace(data=data)
    return contract(config), body_contract(config), identity_contract(config)


@pytest.mark.parametrize('backend', ['yolo', 'vitpose_plus_large', 'sapiens'])
def test_pmpose_options_do_not_change_existing_evidence(backend):
    data = defaults()
    data['sapiensid']['pose']['backend'] = backend
    before = fingerprints(data)
    for key in list(data['sapiensid']['pose']):
        if key.startswith('pmpose_'):
            data['sapiensid']['pose'].pop(key)
    assert fingerprints(data) == before


def test_pmpose_contract_separates_model_and_probability_thresholds():
    data = defaults()
    old = fingerprints(data)
    pose = data['sapiensid']['pose']
    pose['backend'] = 'pmpose'
    expected = fingerprints(data)
    assert all(a != b for a, b in zip(old, expected))
    pose.update(pmpose_directory='another/place', pmpose_batch_size=8)
    assert fingerprints(data) == expected
    for name in ('pmpose_presence_threshold', 'pmpose_visibility_threshold'):
        original = pose[name]
        pose[name] = .25
        assert all(a != b for a, b in zip(expected, fingerprints(data)))
        pose[name] = original


@pytest.mark.parametrize('field,value', [
    ('pmpose_batch_size', 0), ('pmpose_batch_size', 17), ('pmpose_batch_size', True),
    ('pmpose_batch_size', 1.5), ('pmpose_directory', None), ('pmpose_directory', '\0'),
    ('pmpose_directory', ' '), ('pmpose_presence_threshold', -.1),
    ('pmpose_visibility_threshold', 1.1), ('pmpose_visibility_threshold', float('nan')),
    ('pmpose_visibility_threshold', True),
])
def test_invalid_options(field, value):
    data = defaults()
    data['sapiensid']['pose'][field] = value
    with pytest.raises(ValueError, match='sapiensid.pose'):
        validate(data)


def test_crop_rgb_normalization_padding_and_translation():
    from stash_ai.recognition.pmpose_geometry import prepare, decode
    with Image.new('RGB', (300, 300), (10, 20, 30)) as image:
        crop, center, scale = prepare(image, [10, 20, 110, 220])
        _, other_center, other_scale = prepare(image, [160, 50, 260, 250])
    assert crop.shape == (3, 256, 192)
    np.testing.assert_allclose(crop[:, 128, 96], (np.array([10, 20, 30]) - registry.MEAN) / registry.STD, atol=1e-6)
    np.testing.assert_allclose(center, [60, 120])
    np.testing.assert_allclose(scale, [187.5, 250])
    maps = np.zeros((17, 64, 48), dtype=np.float32)
    maps[:, 32, 24] = 1.
    points = decode(maps, center, scale)
    translated = decode(maps, other_center, other_scale)
    np.testing.assert_allclose(translated - points, np.tile([150, 30], (17, 1)), atol=1e-4)
    np.testing.assert_allclose(points[0], np.array([24 / 47, 32 / 63]) * scale + center - scale / 2, atol=1e-4)
    assert (decode(np.zeros_like(maps), center, scale) == -1).all()


def test_batches_restore_flip_labels_and_frame_order_with_empty_frames():
    torch = pytest.importorskip('torch')
    calls = []
    class Model:
        def __call__(self, inputs):
            assert torch.is_inference_mode_enabled()
            flip = len(calls) % 2
            calls.append(len(inputs))
            maps = torch.zeros(len(inputs), 17, 64, 48)
            maps[:, :, 32, 23 if flip else 24] = 1
            scores = torch.arange(17).expand(len(inputs), 17).float() / 20
            if flip:
                scores = scores[:, registry.FLIP]
            return maps, scores, torch.full_like(scores, .75), torch.full_like(scores, .25)
    refiner = pmpose_pose.PMPose(Model(), device='cpu', batch_size=2, fused_flip=False)
    with Image.new('RGB', (300, 300)) as image:
        rows = refiner.predict_many([(image, [[10, 20, 110, 220]]), (image, []),
                                    (image, [[160, 50, 260, 250], [10, 20, 110, 220]])])
    assert calls == [2, 2, 1, 1]
    assert rows[1][0].shape == (0, 17, 2) and rows[1][2]['presence'].shape == (0, 17)
    np.testing.assert_allclose(rows[2][0][0] - rows[0][0][0], np.tile([150, 30], (17, 1)), atol=1e-4)
    np.testing.assert_allclose(rows[2][0][1], rows[0][0][0])
    np.testing.assert_allclose(rows[0][1][0], np.arange(17) / 20)
    np.testing.assert_allclose(rows[0][2]['presence'], .75)
    np.testing.assert_allclose(rows[0][2]['visibility'], .25)


@pytest.mark.parametrize('failure', ['nan', 'range', 'shape', 'missing'])
def test_invalid_model_outputs_fail_closed(failure):
    torch = pytest.importorskip('torch')
    def model(inputs):
        values = [torch.zeros(len(inputs), 17, 64, 48)] + [torch.ones(len(inputs), 17) for _ in range(3)]
        if failure == 'nan': values[0][0, 0, 0, 0] = float('nan')
        if failure == 'range': values[2][0, 0] = 1.1
        if failure == 'shape': values[3] = torch.ones(1, 23)
        if failure == 'missing': values.pop()
        return values
    with Image.new('RGB', (100, 200)) as image, pytest.raises(ValueError, match='PMPose returned'):
        pmpose_pose.PMPose(model, device='cpu', batch_size=1).predict(image, [[0, 0, 100, 200]])


def test_presence_visibility_and_oks_filter_independently_and_retain_audit_scores():
    from stash_ai.recognition.sapiensid_backend import SapiensIDFaces
    detector = object.__new__(SapiensIDFaces)
    detector.config = SimpleNamespace(data=defaults())
    settings = detector.config.data['sapiensid']['pose']
    settings.update(backend='pmpose', pmpose_visibility_threshold=.5)
    points = np.full((1, 17, 2), 30., dtype=np.float32)
    scores = np.full((1, 17), .9, dtype=np.float32)
    probabilities = {key: np.full((1, 17), .9, dtype=np.float32) for key in ('presence', 'visibility')}
    scores[0, 0] = .1
    probabilities['visibility'][0, 1] = .1  # A confident localization can still be occluded.
    probabilities['presence'][0, 2] = .1
    points[0, 3] = [100, 50]  # Preserve the original detector-box exclusion.
    person = detector._people([[0, 0, 100, 200]], [.99], points, scores, probabilities)[0]
    assert person['keypoints'][:4] == [[-1., -1.]] * 4
    assert person['keypoints'][4] == [30., 30.]
    assert person['keypoint_score_kind'] == 'expected_oks'
    assert person['keypoint_scores'][1] == pytest.approx(.9)
    assert person['keypoint_visibility'][1] == pytest.approx(.1)
    assert person['keypoint_presence'][2] == pytest.approx(.1)
    settings['pmpose_visibility_threshold'] = 0
    person = detector._people([[0, 0, 100, 200]], [.99], points, scores, probabilities)[0]
    assert person['keypoints'][1] == [30., 30.]
    with pytest.raises(ValueError, match='separate finite presence'):
        detector._people([[0, 0, 100, 200]], [.99], points, scores)


def test_loading_checks_hash_strict_weights_and_offline_policy(tmp_path, monkeypatch):
    torch = pytest.importorskip('torch')
    from stash_ai.recognition import pmpose_network, sapiensid_downloads
    config = configuration(tmp_path)
    config.data['sapiensid']['pose']['pmpose_directory'] = 'local/pose'
    config.data['models']['download_missing'] = False
    monkeypatch.setattr(pmpose_network, 'PMPoseB', lambda: torch.nn.Linear(2, 3))
    with pytest.raises(FileNotFoundError, match='download_missing'):
        pmpose_pose.load(config, 'cpu')
    config.data['models']['download_missing'] = True
    seed = tmp_path / 'seed.pth'
    torch.save({'state_dict': torch.nn.Linear(2, 3).state_dict()}, seed)
    raw = seed.read_bytes()
    monkeypatch.setattr(registry, 'SIZE', len(raw))
    monkeypatch.setattr(registry, 'SHA256', hashlib.sha256(raw).hexdigest())
    calls = []
    def install(url, path, size, checksum):
        calls.append(url)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(raw)
    monkeypatch.setattr(sapiensid_downloads, 'install', install)
    refiner = pmpose_pose.load(config, 'cpu')
    assert calls == [registry.URL] and not refiner.model.training
    path = tmp_path / 'local/pose' / registry.NAME
    path.write_bytes(b'x' * len(raw))
    with pytest.raises(ValueError, match='SHA256'):
        pmpose_pose.load(config, 'cpu')
    assert len(calls) == 1
    path.write_bytes(raw)
    monkeypatch.setattr(pmpose_network, 'PMPoseB', lambda: torch.nn.Linear(2, 4))
    with pytest.raises(RuntimeError, match='size mismatch'):
        pmpose_pose.load(config, 'cpu')


def test_training_metadata_cannot_resolve_arbitrary_attributes():
    assert pmpose_pose._history_attribute(pmpose_pose._TrainingHistory, 'mean') is None
    for obj, name in ((pmpose_pose._TrainingHistory, '__dict__'), (np, 'load')):
        with pytest.raises(ValueError, match='metadata attribute'):
            pmpose_pose._history_attribute(obj, name)
