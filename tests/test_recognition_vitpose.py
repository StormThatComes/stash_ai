"""ViTPose contracts, integrity, top-down geometry and production adapter regressions."""
import hashlib
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image

from stash_ai.recognition.config import defaults, load, validate
from stash_ai.recognition.person_signal import contract
from stash_ai.recognition import vitpose_pose, vitpose_registry


def configuration(tmp_path):
    path = tmp_path / 'recognition.yml'
    path.write_text('version: 1\nmodels: {device: cpu}\nsapiensid:\n'
                    '  pose: {backend: vitpose_plus_large}\n')
    return load(path)


@pytest.mark.parametrize('field,value', [
    ('backend', 'vitpose'), ('backend', None), ('backend', []),
    ('batch_size', 0), ('batch_size', 17), ('batch_size', True), ('batch_size', 1.5),
    ('vitpose_directory', None), ('vitpose_directory', ' '), ('vitpose_directory', '\0'),
])
def test_invalid_pose_options_rejected(field, value):
    data = defaults()
    data['sapiensid']['pose'][field] = value
    with pytest.raises(ValueError, match='sapiensid.pose'):
        validate(data)


def test_old_yaml_and_yolo_contract_are_unchanged(tmp_path):
    from stash_ai.recognition.config import digest
    from stash_ai.recognition.sapiensid_registry import MODELS, UPSTREAM_COMMIT, POSE_INFERENCE
    path = tmp_path / 'old.yml'
    path.write_text('version: 1\nsapiensid: {enabled: true}\n')
    config = load(path)
    s = config.data['sapiensid']
    assert s['pose']['backend'] == 'yolo'
    # Reconstruct the previous released format, without using the new contract builder.
    before = dict(format='sapiensid-person-v4', model_yaml=s['model']['model_yaml_sha256'],
                  pose={k: s['pose'][k] for k in ('sha256', 'score_threshold', 'keypoint_score_threshold',
                                                 'nms_threshold', 'max_people_per_frame')},
                  pose_inference=POSE_INFERENCE, variant='wb12m',
                  model_sha256=MODELS['wb12m']['sha256'], source_revision=UPSTREAM_COMMIT)
    assert contract(config) == digest(before)


def test_historical_yolo_snapshot_without_new_keys_keeps_its_fingerprint():
    from stash_ai.recognition.body_observations import contract as body_contract
    from stash_ai.recognition.sapiensid_backend import identity_contract
    data = defaults()
    config = SimpleNamespace(data=data)
    expected = contract(config), body_contract(config), identity_contract(config)
    for name in ('backend', 'batch_size', 'vitpose_directory'):
        data['sapiensid']['pose'].pop(name)
    assert (contract(config), body_contract(config), identity_contract(config)) == expected


def test_backend_and_model_changes_separate_evidence_but_paths_and_batches_do_not(monkeypatch):
    from stash_ai.recognition.body_observations import contract as body_contract
    from stash_ai.recognition.sapiensid_backend import identity_contract, input_contract
    data = defaults()
    config = SimpleNamespace(data=data)
    yolo = contract(config), body_contract(config), identity_contract(config)
    data['sapiensid']['pose']['backend'] = 'vitpose_plus_large'
    vitpose = contract(config), body_contract(config), identity_contract(config)
    assert all(a != b for a, b in zip(yolo, vitpose))
    assert 'vitpose-plus-large' in input_contract(config)
    data['sapiensid']['pose'].update(path='another/yolo.pt', vitpose_directory='another/vitpose', batch_size=1)
    assert (contract(config), body_contract(config), identity_contract(config)) == vitpose
    monkeypatch.setattr(vitpose_registry, 'REVISION', '1' * 40)
    assert contract(config) != vitpose[0]


def fake_assets(monkeypatch):
    payloads = {name: name.encode() for name in vitpose_registry.ASSETS}
    monkeypatch.setattr(vitpose_registry, 'ASSETS', {
        name: (len(raw), hashlib.sha256(raw).hexdigest()) for name, raw in payloads.items()})
    return payloads


def test_assets_download_pinned_files_and_verify_cached_bytes(tmp_path, monkeypatch):
    from stash_ai.recognition import sapiensid_downloads
    config = configuration(tmp_path)
    payloads = fake_assets(monkeypatch)
    calls = []
    def install(url, path, size, checksum):
        calls.append(url)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payloads[path.name])
    monkeypatch.setattr(sapiensid_downloads, 'install', install)
    directory = vitpose_pose.assets(config)
    assert directory == config.model_dir / 'vitpose_plus_large' / vitpose_registry.REVISION
    assert calls == [vitpose_registry.BASE_URL + name for name in payloads]
    assert vitpose_pose.assets(config) == directory and len(calls) == 3
    path = directory / 'model.safetensors'
    path.write_bytes(b'x' * len(payloads[path.name]))
    with pytest.raises(ValueError, match='SHA256'):
        vitpose_pose.assets(config)
    assert len(calls) == 3  # Corrupt existing files are never replaced.


def test_offline_and_relative_asset_directory(tmp_path, monkeypatch):
    from dataclasses import replace
    from stash_ai.recognition.config import plain
    config = configuration(tmp_path)
    data = plain(config.data)
    data['models']['download_missing'] = False
    data['sapiensid']['pose']['vitpose_directory'] = 'local/pose'
    config = replace(config, data=data)
    payloads = fake_assets(monkeypatch)
    with pytest.raises(FileNotFoundError, match='download_missing'):
        vitpose_pose.assets(config)
    directory = tmp_path / 'local/pose'
    directory.mkdir(parents=True)
    for name, raw in payloads.items():
        (directory / name).write_bytes(raw)
    assert vitpose_pose.assets(config) == directory


@pytest.mark.parametrize('missing_module', ['scipy.linalg', 'scipy.ndimage'])
def test_missing_scipy_fails_before_downloading_models(tmp_path, monkeypatch, missing_module):
    import builtins
    original_import = builtins.__import__
    def without_scipy(name, *args, **kwargs):
        if name == missing_module:
            raise ModuleNotFoundError(f"No module named '{name}'", name=name)
        return original_import(name, *args, **kwargs)
    def unexpected_download(_):
        pytest.fail('Missing dependencies must be reported before model downloads')
    monkeypatch.setattr(builtins, '__import__', without_scipy)
    monkeypatch.setattr(vitpose_pose, 'assets', unexpected_download)
    with pytest.raises(RuntimeError, match='requires SciPy.*install requirements-sapiensid.txt') as caught:
        vitpose_pose.load(configuration(tmp_path), 'cpu')
    assert isinstance(caught.value.__cause__, ModuleNotFoundError)


def test_refiner_batches_coco_expert_and_preserves_labels_and_image_coordinates():
    torch = pytest.importorskip('torch')
    from transformers import BatchFeature
    calls = []
    class Processor:
        def __call__(self, *, images, boxes, return_tensors):
            calls.append(boxes[0].copy())
            return BatchFeature({'pixel_values': torch.zeros(len(boxes[0]), 3, 8, 8)})
        def post_process_pose_estimation(self, outputs, *, boxes, threshold):
            assert threshold is None
            labels = torch.arange(16, -1, -1)
            return [[dict(keypoints=torch.tensor([[x + 2, y + 3]] * 17), labels=labels,
                          scores=torch.linspace(-.1, 1.1, 17).flip(0)) for x, y, w, h in boxes[0]]]
    class Model:
        def __call__(self, *, pixel_values, dataset_index):
            assert torch.is_inference_mode_enabled()
            assert dataset_index.dtype == torch.long and dataset_index.tolist() == [0] * len(pixel_values)
            return object()
    refiner = vitpose_pose.ViTPoseLarge(Model(), Processor(), device='cpu', batch_size=2)
    bounds = [[10, 20, 50, 100], [70, 30, 110, 110], [120, 50, 180, 130]]
    with Image.new('RGB', (200, 150)) as image:
        points, scores = refiner.predict(image, bounds)
    assert [len(batch) for batch in calls] == [2, 1]
    np.testing.assert_array_equal(np.concatenate(calls), [[10, 20, 40, 80], [70, 30, 40, 80], [120, 50, 60, 80]])
    np.testing.assert_array_equal(points[:, 0], [[12, 23], [72, 33], [122, 53]])
    assert scores.shape == (3, 17) and np.all(scores[:, 0] == 0) and np.all(scores[:, -1] == 1)
    assert bounds[0] == [10, 20, 50, 100]


def test_empty_frame_never_runs_processor_or_model():
    pytest.importorskip('torch')
    refiner = vitpose_pose.ViTPoseLarge(None, None, device='cpu', batch_size=4)
    points, scores = refiner.predict(None, [])
    assert points.shape == (0, 17, 2) and scores.shape == (0, 17)


@pytest.mark.parametrize('failure', ['count', 'labels', 'nan'])
def test_invalid_refiner_output_is_rejected(failure):
    torch = pytest.importorskip('torch')
    from transformers import BatchFeature
    class Processor:
        def __call__(self, **kwargs):
            return BatchFeature({'pixel_values': torch.zeros(1, 3, 8, 8)})
        def post_process_pose_estimation(self, *args, **kwargs):
            if failure == 'count': return [[]]
            return [[dict(keypoints=torch.full((17, 2), float('nan') if failure == 'nan' else 5.),
                          scores=torch.ones(17), labels=torch.zeros(17) if failure == 'labels' else torch.arange(17))]]
    refiner = vitpose_pose.ViTPoseLarge(lambda **kw: None, Processor(), device='cpu', batch_size=4)
    with pytest.raises(ValueError, match='ViTPose returned'):
        refiner.predict(None, [[0, 0, 100, 100]])


@pytest.mark.parametrize('selected_backend', ['vitpose_plus_large', 'sapiens'])
def test_rear_view_refines_without_face_and_ignores_yolo_keypoints(tmp_path, monkeypatch, selected_backend):
    torch = pytest.importorskip('torch')
    from stash_ai.recognition import sapiensid_backend as backend
    from stash_ai.recognition.sapiensid_v1 import prepare_person_input
    config = configuration(tmp_path)
    from dataclasses import replace
    from stash_ai.recognition.config import plain
    data = plain(config.data)
    data['sapiensid']['pose']['backend'] = selected_backend
    config = replace(config, data=data)
    # No YOLO keypoints at all: only its person boxes are needed by this backend.
    result = SimpleNamespace(boxes=SimpleNamespace(
        xyxy=torch.tensor([[10., 20., 90., 140.], [0., 0., 50., 50.]]), conf=torch.tensor([.99, .1])))
    monkeypatch.setattr(backend, 'load_pose', lambda _: SimpleNamespace(predict=lambda **kw: [result]))
    def predict(image, bounds):
        assert bounds == [[10, 20, 90, 140]]
        points = np.full((1, 17, 2), 30., dtype=np.float32)
        scores = np.ones((1, 17), dtype=np.float32)
        scores[:, :5] = .1  # Rear view: no usable facial pose points.
        points[0, 16] = [90, 100]  # Exclusive crop boundary must be omitted too.
        return points, scores
    from stash_ai.recognition import sapiens_pose
    selected = sapiens_pose if selected_backend == 'sapiens' else vitpose_pose
    other = vitpose_pose if selected_backend == 'sapiens' else sapiens_pose
    monkeypatch.setattr(selected, 'load', lambda *args: SimpleNamespace(predict=predict))
    monkeypatch.setattr(other, 'load', lambda *args: pytest.fail('unselected backend loaded'))
    detector = backend.SapiensIDFaces(config, {}, load_identity=False, load_detector=False)
    try:
        with Image.new('RGB', (160, 180)) as image:
            detector.detect(image, faces=[], force_pose=True)
            assert len(detector.frame_people) == 1
            person = detector.frame_people[0]
            assert person['keypoints'][:5] == [[-1., -1.]] * 5
            assert person['keypoints'][16] == [-1., -1.]
            assert person['keypoints'][5] == [30., 30.]
            bounds, points = detector.person_geometry(image, person)
            assert len(points) == 22 and points[17:] == [[-1., -1.]] * 5
            with image.crop(bounds) as crop:
                _, normalized, _ = prepare_person_input(crop, points)
            assert (normalized[0] >= 0).all(dim=1).sum().item() == 11
    finally:
        detector.close()
    assert detector.pose_refiner is None


def test_real_processor_restores_translated_boxes_and_batches():
    torch = pytest.importorskip('torch')
    transformers = pytest.importorskip('transformers')
    processor = transformers.VitPoseImageProcessor()
    class Model:
        def __call__(self, *, pixel_values, dataset_index):
            # Identical heatmaps in two translated person crops must translate equally.
            y, x = torch.meshgrid(torch.arange(64), torch.arange(48), indexing='ij')
            heatmap = torch.exp(-((x - 24.) ** 2 + (y - 32.) ** 2) / 8.)
            return SimpleNamespace(heatmaps=heatmap.expand(len(pixel_values), 17, 64, 48).clone())
    refiner = vitpose_pose.ViTPoseLarge(Model(), processor, device='cpu', batch_size=1)
    with Image.new('RGB', (300, 300)) as image:
        points, scores = refiner.predict(image, [[10, 20, 110, 220], [160, 50, 260, 250]])
    np.testing.assert_allclose(points[1] - points[0], np.tile([150, 30], (17, 1)), atol=1e-3)
    assert np.isfinite(points).all() and (scores > .5).all()


@pytest.mark.parametrize('missing_weight', [False, True])
def test_real_transformer_loads_verified_safetensors_and_runs_coco_expert(tmp_path, monkeypatch, missing_weight):
    torch = pytest.importorskip('torch')
    transformers = pytest.importorskip('transformers')
    config = configuration(tmp_path)
    directory = config.model_dir / 'vitpose_plus_large' / vitpose_registry.REVISION
    backbone = transformers.VitPoseBackboneConfig(hidden_size=32, num_hidden_layers=1,
        num_attention_heads=4, num_experts=6, part_features=8, out_indices=[1])
    model_config = transformers.VitPoseConfig(backbone_config=backbone, num_labels=17, use_simple_decoder=False)
    model = transformers.VitPoseForPoseEstimation(model_config).eval()
    model.save_pretrained(directory, safe_serialization=True)
    transformers.VitPoseImageProcessor().save_pretrained(directory)
    if missing_weight:
        from safetensors.torch import load_file, save_file
        weights = load_file(directory / 'model.safetensors')
        weights.pop(next(iter(weights)))
        save_file(weights, directory / 'model.safetensors', metadata={'format': 'pt'})
    monkeypatch.setattr(vitpose_registry, 'ASSETS', {name: (len(raw), hashlib.sha256(raw).hexdigest())
        for name in vitpose_registry.ASSETS for raw in [(directory / name).read_bytes()]})
    if missing_weight:
        with pytest.raises(ValueError, match='does not match'):
            vitpose_pose.load(config, 'cpu')
        return
    refiner = vitpose_pose.load(config, 'cpu')
    assert not refiner.model.training and next(refiner.model.parameters()).dtype == torch.float32
    with Image.new('RGB', (100, 200)) as image:
        points, scores = refiner.predict(image, [[0, 0, 100, 200]])
    assert points.shape == (1, 17, 2) and scores.shape == (1, 17)
    assert np.isfinite(points).all() and np.isfinite(scores).all()
