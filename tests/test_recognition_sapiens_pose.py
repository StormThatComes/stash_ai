"""Sapiens pose downloads, COCO mapping, geometry and backend integration."""
import copy
import hashlib
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image

from stash_ai.recognition import sapiens_pose as pose, sapiens_pose_registry as registry
from stash_ai.recognition.config import defaults, load, plain, validate


def configuration(tmp_path, **options):
    path = tmp_path / 'recognition.yml'
    path.write_text('version: 1\nmodels: {device: cpu}\nsapiensid:\n  pose: {backend: sapiens}\n')
    config = load(path)
    data = plain(config.data)
    data['sapiensid']['pose'].update(options)
    return replace(config, data=data)


@pytest.mark.parametrize('key,value', [
    ('sapiens_variant', None), ('sapiens_variant', []), ('sapiens_variant', '2b'),
    ('sapiens_variant', .3), ('sapiens_batch_size', 0), ('sapiens_batch_size', 17),
    ('sapiens_batch_size', True), ('sapiens_batch_size', 1.5),
    ('sapiens_directory', None), ('sapiens_directory', ' '), ('sapiens_directory', '\0'),
])
def test_invalid_options(key, value):
    data = defaults()
    data['sapiensid']['pose'][key] = value
    with pytest.raises(ValueError, match='sapiensid.pose'):
        validate(data)


@pytest.mark.parametrize('backend', ['yolo', 'vitpose_plus_large'])
def test_existing_and_historical_contracts_ignore_unused_sapiens_settings(tmp_path, backend):
    from stash_ai.recognition.person_signal import contract
    from stash_ai.recognition.body_observations import contract as body_contract
    from stash_ai.recognition.sapiensid_backend import identity_contract
    config = configuration(tmp_path, backend=backend)
    before = contract(config), body_contract(config), identity_contract(config)
    config.data['sapiensid']['pose'].update(sapiens_variant='1b', sapiens_batch_size=16,
                                          sapiens_directory='another/folder')
    assert (contract(config), body_contract(config), identity_contract(config)) == before
    for field in ('sapiens_variant', 'sapiens_batch_size', 'sapiens_directory'):
        config.data['sapiensid']['pose'].pop(field)
    assert (contract(config), body_contract(config), identity_contract(config)) == before


def test_selected_model_and_preprocessing_separate_evidence(tmp_path, monkeypatch):
    from stash_ai.recognition.person_signal import contract
    from stash_ai.recognition.body_observations import contract as body_contract
    from stash_ai.recognition.sapiensid_backend import identity_contract
    config = configuration(tmp_path)
    fingerprints = lambda: (contract(config), body_contract(config), identity_contract(config))
    expected = fingerprints()
    config.data['sapiensid']['pose'].update(sapiens_directory='other', sapiens_batch_size=2, batch_size=8)
    assert fingerprints() == expected
    for backend in ('yolo', 'vitpose_plus_large'):
        config.data['sapiensid']['pose']['backend'] = backend
        assert all(a != b for a, b in zip(fingerprints(), expected))
    config.data['sapiensid']['pose']['backend'] = 'sapiens'
    for variant in ('0.6b', '1b'):
        config.data['sapiensid']['pose']['sapiens_variant'] = variant
        assert all(a != b for a, b in zip(fingerprints(), expected))
    config.data['sapiensid']['pose']['sapiens_variant'] = '0.3b'
    monkeypatch.setattr(registry, 'COCO17_INDICES', tuple(reversed(registry.COCO17_INDICES)))
    assert all(a != b for a, b in zip(fingerprints(), expected))


@pytest.mark.parametrize('variant', ['0.3b', '0.6b', '1b'])
def test_selected_asset_download_and_corruption_detection(tmp_path, monkeypatch, variant):
    from stash_ai.recognition import sapiensid_downloads
    config = configuration(tmp_path, sapiens_variant=variant)
    models = copy.deepcopy(registry.MODELS)
    raw = b'verified model fixture'
    models[variant].update(size=len(raw), sha256=hashlib.sha256(raw).hexdigest())
    monkeypatch.setattr(registry, 'MODELS', models)
    calls = []
    def install(url, path, size, checksum):
        calls.append(url)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(raw)
    monkeypatch.setattr(sapiensid_downloads, 'install', install)
    path = pose.asset(config)
    model = models[variant]
    assert path == config.model_dir / 'sapiens_pose' / variant / model['revision'] / model['filename']
    assert calls == [f"https://huggingface.co/{model['repository']}/resolve/{model['revision']}/{model['filename']}"]
    assert pose.asset(config) == path and len(calls) == 1
    path.write_bytes(b'x' * len(raw))
    with pytest.raises(ValueError, match='SHA256'):
        pose.asset(config)
    assert len(calls) == 1


def test_offline_relative_path_and_corrupt_model_never_loaded(tmp_path, monkeypatch):
    torch = pytest.importorskip('torch')
    config = configuration(tmp_path, sapiens_directory='custom/pose')
    config.data['models']['download_missing'] = False
    def unexpected(*args, **kwargs):pytest.fail('unverified model loaded')
    monkeypatch.setattr(torch.jit, 'load', unexpected)
    with pytest.raises(FileNotFoundError, match='download_missing'):
        pose.load(config, 'cpu')
    model = registry.MODELS['0.3b']
    path = tmp_path / 'custom/pose' / model['filename']
    path.parent.mkdir(parents=True)
    path.write_bytes(b'incomplete model')
    with pytest.raises(ValueError, match='size mismatch'):
        pose.load(config, 'cpu')
    assert path.read_bytes() == b'incomplete model'


def gaussian(x=80.25, y=120.75):
    width, height = registry.HEATMAP_SIZE
    yy, xx = np.mgrid[:height, :width]
    return np.exp(-((xx - x) ** 2 + (yy - y) ** 2) / 72.).astype(np.float32)


def test_rgb_normalization_and_udp_crop_preserve_source_geometry():
    from stash_ai.recognition.sapiens_pose_geometry import prepare
    with Image.new('RGB', (500, 500), (240, 40, 10)) as image:
        crop, center, scale = prepare(image, [100, 50, 200, 250])
    assert crop.shape == (3, 1024, 768) and crop.dtype == np.float32 and crop.flags.c_contiguous
    np.testing.assert_array_equal(center, [150, 150])
    np.testing.assert_array_equal(scale, [187.5, 250])
    np.testing.assert_allclose(crop[:, 512, 384], (np.array([240, 40, 10]) - registry.MEAN) / registry.STD, atol=1e-6)


def test_udp_subpixel_decode_translates_and_keeps_missing_points():
    from stash_ai.recognition.sapiens_pose_geometry import decode
    maps = np.repeat(gaussian()[None], 17, axis=0)
    maps[0] = 0; maps[1] = -1
    original = maps.copy()
    center, scale = np.array([200., 300.]), np.array([300., 400.])
    points, scores = decode(maps, center, scale)
    expected = np.array([80.25 / 191, 120.75 / 255]) * scale + center - scale * .5
    np.testing.assert_allclose(points[2:], np.tile(expected, (15, 1)), atol=.03)
    assert (points[:2] == -1).all() and (scores[:2] == 0).all()
    translated, _ = decode(maps, center + [55, -20], scale)
    np.testing.assert_allclose(translated[2:] - points[2:], np.tile([55, -20], (15, 1)), atol=1e-4)
    np.testing.assert_array_equal(maps, original)


@pytest.mark.parametrize('location', [(0, 0), (191, 255), (0, 255), (191, 0)])
def test_udp_border_and_flat_heatmaps_are_finite(location):
    from stash_ai.recognition.sapiens_pose_geometry import decode
    maps = np.repeat(gaussian(*location)[None], 17, axis=0)
    maps[0] = 1
    points, scores = decode(maps, np.array([150, 200]), np.array([300, 400]))
    assert np.isfinite(points).all() and np.isfinite(scores).all()


def test_mapping_and_batches_across_frames_preserve_people_order():
    torch = pytest.importorskip('torch')
    calls = []
    # Distinct scores on all 308 channels catch wrist/hip ordering mistakes.
    heatmaps = torch.from_numpy(gaussian())[None] * (torch.arange(308) + 1)[:, None, None] / 308
    class Model:
        def __call__(self, inputs):
            assert torch.is_inference_mode_enabled() and inputs.dtype == torch.float32
            assert inputs.shape[1:] == (3, 1024, 768)
            calls.append(len(inputs))
            return heatmaps[None].expand(len(inputs), -1, -1, -1)
    model = pose.SapiensPose(Model(), device='cpu', batch_size=2)
    boxes = [[0, 0, 100, 200], [150, 30, 250, 230]]
    with Image.new('RGB', (400, 400)) as image:
        result = model.predict_many([(image, boxes[:1]), (image, []), (image, boxes[1:] * 2)])
    assert calls == [2, 1] and result[1][0].shape == (0, 17, 2)
    expected_indices = [0, 1, 2, 3, 4, 5, 6, 7, 8, 62, 41, 9, 10, 11, 12, 13, 14]
    np.testing.assert_allclose(result[0][1][0], (np.array(expected_indices) + 1) / 308 * gaussian().max(), rtol=1e-6)
    np.testing.assert_allclose(result[2][0][0] - result[0][0][0], np.tile([150, 30], (17, 1)), atol=1e-4)
    np.testing.assert_array_equal(result[2][0][0], result[2][0][1])
    assert boxes == [[0, 0, 100, 200], [150, 30, 250, 230]]


@pytest.mark.parametrize('bounds', [[[0, 0, 0, 10]], [[0, 0, 10]], [[0, 0, float('nan'), 10]]])
def test_bad_boxes_are_rejected_before_inference(bounds):
    pytest.importorskip('torch')
    with pytest.raises(ValueError, match='person boxes'):
        pose.SapiensPose(None, device='cpu', batch_size=1).predict(None, bounds)


def test_empty_frames_do_not_run_model_or_preprocessor():
    pytest.importorskip('torch')
    model = pose.SapiensPose(None, device='cpu', batch_size=1)
    assert model.predict_many([]) == []
    points, scores = model.predict(None, [])
    assert points.shape == (0, 17, 2) and scores.shape == (0, 17)


@pytest.mark.parametrize('failure', ['channels', 'size', 'count', 'type', 'nan'])
def test_bad_outputs_are_rejected(failure):
    torch = pytest.importorskip('torch')
    shape = [1, 308, 256, 192]
    if failure == 'channels':shape[1] = 17
    if failure == 'size':shape[2] = 64
    if failure == 'count':shape[0] = 2
    value = torch.full((1,), float('nan') if failure == 'nan' else 1.).expand(*shape)
    model = pose.SapiensPose(lambda _: None if failure == 'type' else value, device='cpu', batch_size=1)
    with Image.new('RGB', (100, 200)) as image, pytest.raises(ValueError, match='Sapiens Pose returned'):
        model.predict(image, [[0, 0, 100, 200]])


def test_verified_torchscript_load_and_real_inference(tmp_path, monkeypatch):
    torch = pytest.importorskip('torch')
    class Tiny(torch.nn.Module):
        def forward(self, images):
            return images.square().mean(dim=(1, 2, 3))[:, None, None, None].expand(-1, 308, 256, 192)
    config = configuration(tmp_path, sapiens_directory='local')
    model = copy.deepcopy(registry.MODELS['0.3b'])
    path = tmp_path / 'local' / model['filename']
    path.parent.mkdir()
    torch.jit.script(Tiny()).save(str(path))
    raw = path.read_bytes()
    model.update(size=len(raw), sha256=hashlib.sha256(raw).hexdigest())
    monkeypatch.setitem(registry.MODELS, '0.3b', model)
    refiner = pose.load(config, 'cpu')
    assert not refiner.model.training and refiner.batch_size == 1
    with Image.new('RGB', (100, 200), 'white') as image:
        points, scores = refiner.predict(image, [[0, 0, 100, 200]])
    assert points.shape == (1, 17, 2) and np.isfinite(points).all()
    assert scores.shape == (1, 17) and (scores > 0).all()


def test_exported_yaml_preserves_relative_sapiens_location(tmp_path):
    import yaml
    from stash_ai.recognition.merge_goal_output import configuration_document
    from stash_ai.recognition.merge_report_policy import proposal
    config = configuration(tmp_path, sapiens_directory='custom/sapiens')
    exported = yaml.safe_load(configuration_document(config, proposal(config, {}), {}))
    assert exported['sapiensid']['pose']['sapiens_directory'] == str((tmp_path / 'custom/sapiens').resolve())
