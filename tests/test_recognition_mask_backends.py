"""Selectable mask models: contracts, shared pipeline, assets and prompt ownership."""
from dataclasses import replace
import hashlib
from types import SimpleNamespace

import numpy as np
from PIL import Image
import pytest

from stash_ai.recognition.config import defaults, load, plain
from stash_ai.recognition.pose_refinement_config import contract, validate
from stash_ai.recognition.mask_backends import BACKENDS
from tests.test_recognition_pmpose import configuration


@pytest.mark.parametrize('backend', BACKENDS)
def test_public_yaml_round_trip_and_tune_selection(tmp_path, backend):
    from stash_ai.recognition.config_layout import v2_document
    from stash_ai.recognition.tune_editor import Settings, CHOICES
    path = tmp_path / 'masks.yml'
    path.write_text('version: 2\npose:\n  estimator: {backend: pmpose}\n'
                    '  refinement: {enabled: true, backend: ' + backend + '}\n')
    config = load(path)
    assert v2_document(config.data)['pose']['mask']['backend'] == backend
    editor = Settings(config)
    assert backend in CHOICES['pose.mask.backend']
    editor.edit('pose.refinement.backend', 'hq_sam2_large')
    assert editor.effective().data['sapiensid']['pose']['refinement']['backend'] == 'hq_sam2_large'


def test_default_fingerprint_is_preserved_and_alternatives_invalidate():
    settings = defaults()['sapiensid']['pose']['refinement']
    before = contract(settings)
    old = dict(settings); old.pop('backend')
    assert before == contract(old)
    contracts = [contract(dict(settings, backend=b)) for b in BACKENDS]
    assert len({repr(c) for c in contracts}) == len(BACKENDS)
    for backend in BACKENDS:
        assert contract(dict(settings, backend=backend, directory='relocated')) == contract(dict(settings, backend=backend))


def test_invalid_backend_and_disabled_options():
    pose = defaults()['sapiensid']['pose']
    pose['backend'] = 'pmpose'
    pose['refinement'].update(enabled=True, backend='not_a_model')
    with pytest.raises(ValueError, match='backend'):
        validate(pose)
    pose['refinement']['enabled'] = False
    validate(pose)


@pytest.mark.parametrize('backend,module', [('sam3','sam3_masks'),('hq_sam2_large','hq_sam2_masks'),
                                            ('rfdetr_seg_large','rfdetr_masks')])
def test_existing_entry_point_dispatches_without_loading_default_assets(tmp_path, monkeypatch, backend, module):
    import importlib
    from stash_ai.recognition import sam_pose2seg
    config = configuration(tmp_path)
    config.data['sapiensid']['pose']['refinement']['backend'] = backend
    token = object()
    monkeypatch.setattr(importlib.import_module('stash_ai.recognition.' + module), 'load', lambda c,d: token)
    monkeypatch.setattr(sam_pose2seg, 'assets', lambda *a: pytest.fail('default model downloaded'))
    assert sam_pose2seg.load(config, 'cpu') is token


def test_sam3_download_policy_revision_hash_and_cached_offline_use(tmp_path, monkeypatch):
    from stash_ai.recognition import sam3_masks
    import huggingface_hub
    config = configuration(tmp_path)
    raw = b'configuration'; size = len(raw)
    checksum = hashlib.sha1(f'blob {size}\0'.encode() + raw).hexdigest()
    monkeypatch.setattr(sam3_masks, 'FILES', {'config.json': (size, checksum)})
    calls = []
    def download(repo, name, **kwargs):
        calls.append((repo, kwargs['revision']))
        p = kwargs['local_dir'] / name; p.parent.mkdir(parents=True); p.write_bytes(raw)
    monkeypatch.setattr(huggingface_hub, 'hf_hub_download', download)
    config.data['models']['download_missing'] = False
    with pytest.raises(FileNotFoundError, match='download_missing'):
        sam3_masks.assets(config)
    assert not calls
    config.data['models']['download_missing'] = True
    directory = sam3_masks.assets(config)
    assert calls == [('facebook/sam3', sam3_masks.REVISION)]
    config.data['models']['download_missing'] = False
    assert sam3_masks.assets(config) == directory
    (directory/'config.json').write_bytes(b'x'*size)
    with pytest.raises(ValueError, match='integrity'):
        sam3_masks.assets(config)
    assert len(calls) == 1


def test_sam3_gated_download_error(tmp_path, monkeypatch):
    from stash_ai.recognition import sam3_masks
    import huggingface_hub
    from huggingface_hub.errors import GatedRepoError
    config = configuration(tmp_path)
    def denied(*a, **kw):
        error = GatedRepoError.__new__(GatedRepoError)
        Exception.__init__(error, 'Access denied')
        raise error
    monkeypatch.setattr(huggingface_hub, 'hf_hub_download', denied)
    with pytest.raises(RuntimeError, match='HF_TOKEN'):
        sam3_masks.assets(config)


def test_rfdetr_selects_whole_instances_and_rejects_unmatched_pose():
    from stash_ai.recognition.rfdetr_masks import RFDETRMasks
    model = RFDETRMasks(None, 'cpu', 'fp32', .5)
    model.size = (30,20)
    a = np.zeros((20,30), bool); a[2:18,2:14] = True
    b = np.zeros_like(a); b[2:18,12:28] = True
    model.masks = [np.packbits(a), np.packbits(b)]
    model.scores = np.array([.95,.90])
    mask, score = model.predict([[20,5]], [[5,5]])
    assert np.array_equal(mask,b) and score == .9
    # A pose spanning two people must never produce the union of both masks.
    assert not model.predict([[5,5],[20,5]])[0].any()
    assert not model.predict([[20,5]], [[22,5]])[0].any()
    model.clear()
    with pytest.raises(RuntimeError, match='active'):
        model.predict([[20,5]])


def test_sam3_real_processor_coordinates_reuse_and_cleanup():
    torch = pytest.importorskip('torch')
    from transformers import Sam3TrackerProcessor, Sam2ImageProcessor
    from stash_ai.recognition.sam3_masks import SAM3Masks
    class Model:
        calls = 0
        def get_image_embeddings(self, pixels):
            self.calls += 1
            return [pixels]
        def __call__(self, **kwargs):
            assert torch.is_inference_mode_enabled()
            assert kwargs['input_points'].shape == (1,1,3,2)
            assert kwargs['input_labels'].tolist() == [[[1,1,0]]]
            count = 3 if kwargs['multimask_output'] else 1
            return SimpleNamespace(pred_masks=torch.ones(1,1,count,8,8),
                                   iou_scores=torch.full((1,1,count),.9))
    model = Model()
    processor = Sam3TrackerProcessor(Sam2ImageProcessor(size={'height':32,'width':32}))
    segmenter = SAM3Masks(model, processor, 'cpu', 'fp32')
    positive,negative = [[5,5],[10,10]], [[20,10]]
    image = Image.new('RGB',(30,20))
    def process(prompts, initial, decoder):
        assert initial[0][0].shape == (20,30)
        return decoder(positive,negative,True)
    from stash_ai.recognition.tune_mask_generation import MaskReplay
    with MaskReplay(SimpleNamespace(image=image)).bind(image):
        initial = segmenter.refine_frames([(image,[(positive,negative)])], process)
        result = segmenter.refine_frames([(image,[(positive,negative)])], process)
        assert np.array_equal(initial[0][0][0], result[0][0][0])
    assert len(result[0]) == 3 and model.calls == 1
    assert segmenter.embeddings is None and segmenter.size is None
    with pytest.raises(RuntimeError, match='active'):
        segmenter.predict(positive,negative)
    with pytest.raises(ValueError, match='failure'):
        with segmenter.frame(image):
            raise ValueError('failure')
    assert segmenter.embeddings is None


def test_hq_manifest_does_not_include_initializers():
    from stash_ai.recognition import hq_sam2_registry as r
    assert 'LICENSE' in r.SOURCE_FILES and r.CONFIG in r.SOURCE_FILES
    assert not any(name.endswith('__init__.py') for name in r.SOURCE_FILES)
    assert all(len(item['sha256']) == 64 and item['size'] > 0 for item in r.SOURCE_FILES.values())


@pytest.mark.parametrize('backend', ['hq_sam2_large','rfdetr_seg_large'])
def test_public_checkpoint_download_is_verified_and_respects_offline(tmp_path, monkeypatch, backend):
    from stash_ai.recognition.mask_backends import checkpoint
    from stash_ai.recognition import sapiensid_downloads
    config = configuration(tmp_path)
    raw = b'pinned-weights'; checksum = hashlib.sha256(raw).hexdigest(); calls = []
    def install(url,path,size,digest):
        calls.append(url)
        path.parent.mkdir(parents=True, exist_ok=True); path.write_bytes(raw)
    monkeypatch.setattr(sapiensid_downloads,'install',install)
    args = (config,backend,'revision','weights.pt','https://example.com/weights.pt',len(raw),checksum)
    config.data['models']['download_missing'] = False
    with pytest.raises(FileNotFoundError,match='download_missing'):
        checkpoint(*args)
    assert not calls
    config.data['models']['download_missing'] = True
    path = checkpoint(*args)
    config.data['models']['download_missing'] = False
    assert checkpoint(*args) == path and len(calls)==1
    path.write_bytes(b'x'*len(raw))
    with pytest.raises(ValueError,match='SHA256'):
        checkpoint(*args)
