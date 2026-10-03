"""Offline contracts for fully managed Hugging Face JoyTag installations."""

import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

from stash_ai.inference.identity import model_identity
from stash_ai import cli as app
from stash_ai import joytag
from stash_ai.models import JOYTAG, WD, ModelSettings, Vocabulary
from stash_ai.taggers import JoyTagger
from stash_ai.vocabulary import load_vocabulary

pytestmark = pytest.mark.usefixtures("reset_multimodel")

LOADER = b'''class ViT:
    def __init__(self, **config):
        self.image_size = config['image_size']
        self.n_tags = config['n_tags']
    def load(self, weights):
        self.weights = weights
    def eval(self):
        return self
    def to(self, device):
        return self
'''


@pytest.fixture
def hf_assets(tmp_path, monkeypatch):
    model_dir = tmp_path / "models" / "joytag"
    cache = tmp_path / "hf-cache"
    cache.mkdir()
    assets = {"top_tags.txt": b"blue_hair\nwatermark\n",
              "config.json": b'{"class":"ViT","image_size":448,"n_tags":2}',
              "model.safetensors": b"test safetensors"}
    for name, data in {**assets, "Models.py": LOADER}.items():
        (cache / name).write_bytes(data)
    monkeypatch.setattr(joytag, "LOADER_SHA256", hashlib.sha256(LOADER).hexdigest())
    requests = []
    def download(**kwargs):
        requests.append(kwargs)
        assert kwargs['repo_id'] == 'fancyfeast/joytag'
        if kwargs.get('repo_type') == 'space':
            assert kwargs['filename'] == 'Models.py' and kwargs['revision'] == joytag.LOADER_REVISION
        else:
            assert kwargs['revision'] == JOYTAG.revision
        path = cache / kwargs['filename']
        if not path.is_file():
            raise FileNotFoundError(f"Missing cached {kwargs['filename']}")
        return str(path)
    monkeypatch.setitem(sys.modules, "huggingface_hub", SimpleNamespace(hf_hub_download=download))
    monkeypatch.delenv("JOYTAG_MODEL_PATH", raising=False)
    config = tmp_path / "models.yml"
    config.write_text("version: 1\njoytag:\n  model_dir: models/joytag\n")
    monkeypatch.setattr(app, "create_stash_client", lambda _: pytest.fail("model download accessed Stash"))
    return SimpleNamespace(model_dir=model_dir, cache=cache, config=config, assets=assets, requests=requests)


def command(f):
    return ["--models", "joytag", "--model-config", str(f.config), "--download-models"]


def test_download_prepares_weights_and_loader_without_importing_them(hf_assets, monkeypatch):
    f = hf_assets
    monkeypatch.setitem(sys.modules, 'torch', None)
    assert app.main(command(f)) == 0
    assert {r['filename'] for r in f.requests} == {*f.assets, 'Models.py'}
    before = {n: ((f.model_dir/n).read_bytes(), (f.model_dir/n).stat().st_mtime_ns) for n in f.assets}
    f.requests.clear()
    assert app.main(command(f) + ['--offline']) == 0
    assert len(f.requests) == 1 and f.requests[0]['local_files_only'] is True
    assert before == {n: ((f.model_dir/n).read_bytes(), (f.model_dir/n).stat().st_mtime_ns) for n in f.assets}


@pytest.mark.parametrize('missing', ['top_tags.txt', 'config.json', 'model.safetensors', 'Models.py'])
def test_offline_requires_cached_assets_and_loader(hf_assets, missing):
    f = hf_assets
    f.model_dir.mkdir(parents=True)
    for name, data in f.assets.items():
        if name != missing:
            (f.model_dir/name).write_bytes(data)
    if missing == 'Models.py':
        (f.cache/missing).unlink()
    with pytest.raises(FileNotFoundError, match=missing):
        app.main(command(f) + ['--offline'])
    assert all(r.get('local_files_only') for r in f.requests)


@pytest.mark.parametrize('offline', [False, True])
def test_model_pt_never_substitutes_for_safetensors(hf_assets, offline):
    f = hf_assets
    f.model_dir.mkdir(parents=True)
    old = f.model_dir/'model.pt'
    old.write_bytes(b'old weights')
    if offline:
        for name in ('config.json', 'top_tags.txt'):
            (f.model_dir/name).write_bytes(f.assets[name])
        with pytest.raises(FileNotFoundError, match='model.safetensors'):
            app.main(command(f) + ['--offline'])
    else:
        assert app.main(command(f)) == 0
        assert (f.model_dir/'model.safetensors').read_bytes() == f.assets['model.safetensors']
    assert old.read_bytes() == b'old weights'


def test_inference_uses_verified_hf_loader_and_safetensors_only(hf_assets, monkeypatch):
    f = hf_assets
    settings = ModelSettings(JOYTAG, f.model_dir, device='cpu')
    vocabulary = load_vocabulary(settings)
    (f.model_dir/'Models.py').write_text("raise AssertionError('local code was executed')")
    monkeypatch.setitem(sys.modules, 'Models', SimpleNamespace())
    reads = []
    def load_file(path, device):
        assert Path(path).read_bytes() == f.assets['model.safetensors'] and device == 'cpu'
        reads.append(path)
        return {'weights': 'loaded'}
    monkeypatch.setitem(sys.modules, 'safetensors.torch', SimpleNamespace(load_file=load_file))
    monkeypatch.setitem(sys.modules, 'torch', SimpleNamespace(
        device=lambda name: SimpleNamespace(type=name), cuda=SimpleNamespace(is_available=lambda: False)))
    old_path = list(sys.path)
    tagger = JoyTagger(settings, vocabulary, lambda image, size: image)
    assert tagger.model.weights == {'weights': 'loaded'} and tagger.model.n_tags == len(vocabulary.labels)
    assert reads == [str(f.model_dir/'model.safetensors')] and sys.path == old_path
    tagger.close()


def test_corrupt_loader_is_rejected_before_execution(hf_assets, monkeypatch):
    f = hf_assets
    (f.cache/'Models.py').write_text("raise AssertionError('corrupt code was executed')")
    monkeypatch.setitem(sys.modules, 'safetensors.torch', SimpleNamespace())
    with pytest.raises(ValueError, match='checksum'):
        joytag.loader_path()


@pytest.mark.parametrize('values', [{'class': 'Other', 'n_tags': 2}, {'class': 'ViT', 'n_tags': 3}])
def test_config_must_match_official_architecture_and_vocabulary(hf_assets, monkeypatch, values):
    f = hf_assets
    f.model_dir.mkdir(parents=True)
    (f.model_dir/'config.json').write_text(json.dumps(values))
    monkeypatch.setitem(sys.modules, 'safetensors.torch', SimpleNamespace(
        load_file=lambda *a, **k: pytest.fail('invalid config reached weights')))
    with pytest.raises(ValueError, match='config'):
        joytag.load_model(f.model_dir, 2)


def test_joytag_defaults_to_managed_models_directory_and_rejects_code_dir(tmp_path, monkeypatch):
    monkeypatch.delenv('JOYTAG_MODEL_PATH', raising=False)
    monkeypatch.setenv('JOYTAG_CODE_PATH', str(tmp_path/'unused-checkout'))
    config = tmp_path/'models.yml'
    config.write_text('version: 1\n')
    settings = app.load_model_settings(config, ['joytag'], None)[0]
    assert settings.model_dir == tmp_path/'models/joytag'
    config.write_text('version: 1\njoytag:\n  code_dir: old-checkout\n')
    with pytest.raises(ValueError, match='Invalid joytag settings'):
        app.load_model_settings(config, ['joytag'], None)


def test_loader_revision_changes_only_joytag_inference_identity(tmp_path, monkeypatch):
    vocabulary = Vocabulary(['tag'], {})
    configs = [ModelSettings(s, tmp_path/s.key) for s in (JOYTAG, WD)]
    before = [model_identity(app, c, vocabulary) for c in configs]
    monkeypatch.setattr(joytag, 'LOADER_REVISION', 'a'*40)
    after = [model_identity(app, c, vocabulary) for c in configs]
    assert before[0] != after[0] and before[1] == after[1]
