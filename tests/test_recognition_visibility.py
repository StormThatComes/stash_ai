import json
import numpy as np
from PIL import Image
import pytest

from stash_ai.recognition.config import load
from stash_ai.recognition.database import FaceDatabase
from stash_ai.recognition.models import checked_detection, namespace
from stash_ai.recognition.portraits import policy_identity
from stash_ai.recognition.visibility import CONTRACT_ID, FaceVisibility, LazyVisibility, measurement, prepare
from tests.helpers.recognition import Visibility
from tests.test_recognition_metadata import setup as video_setup, Decoder
from tests.test_recognition_vtt_portraits import setup, run, write_config


def cues(stash, count):
    def time(seconds):
        return f'00:{seconds // 60:02d}:{seconds % 60:02d}.000'
    stash.vtt = ('WEBVTT\n\n' + '\n\n'.join(
        f'{time(i)} --> {time(i + 1)}\nsprite.jpg#xywh=0,0,500,350' for i in range(count))).encode()


def winner(config):
    with FaceDatabase(config.database) as db:
        row = db.connection.execute("SELECT * FROM portraits WHERE status='published'").fetchone()
        return dict(row), json.loads(row['metadata'])


def test_vtt_selects_uncovered_late_observation_beyond_old_shortlist_and_pool(tmp_path, monkeypatch):
    config, stash, services, backends = setup(tmp_path, monkeypatch, enabled=False, max_candidates=1)
    cues(stash, 101)
    assert run(config, services, '1') == 0
    with FaceDatabase(config.database) as db:
        for row in db.connection.execute('SELECT id,cue,quality FROM observations').fetchall():
            quality = json.loads(row['quality'])
            quality.update(sharpness=500 if row['cue'] < 100 else 100, visible_face_fraction=.1 if row['cue'] < 100 else 1.,
                           visibility={'contract': CONTRACT_ID, 'device': 'cpu'})
            db.connection.execute('UPDATE observations SET quality=? WHERE id=?', (json.dumps(quality), row['id']))
        db.connection.commit()
    write_config(config.source_path, max_candidates=1)
    config = load(config.source_path)
    services.visibility = lambda *args: pytest.fail('retained visibility must be reused')
    assert run(config, services, '1') == 0
    _, metadata = winner(config)
    assert metadata['cue'] == 100 and metadata['quality']['visible_face_fraction'] == 1
    assert metadata['eligible_observations'] == 101
    assert len(backends) == 1 and backends[0].calls == 101
    with FaceDatabase(config.database) as db:
        assert db.connection.execute('SELECT COUNT(*) FROM portraits WHERE image IS NOT NULL').fetchone()[0] == 1
        assert sum('face_occluded' in json.loads(r[0]).get('quality_shortfalls', [])
                   for r in db.connection.execute('SELECT metadata FROM portraits')) == 100
    writes = list(stash.writes)
    write_config(config.source_path, max_candidates=64, time_offsets=[2], max_video_bytes=1048576)
    changed = load(config.source_path)
    assert policy_identity(config) == policy_identity(changed)
    assert run(changed, services, '1') == 0 and stash.writes == writes


def test_vtt_encodes_only_first_usable_winner_and_falls_back_beyond_eight(tmp_path, monkeypatch):
    import stash_ai.recognition.vtt_portraits as vtt
    config, stash, services, _ = setup(tmp_path, monkeypatch, max_candidates=1)
    cues(stash, 9)
    original, attempts = vtt.portrait_bytes, []
    def prepare_candidate(*args, **kwargs):
        attempts.append(1)
        if len(attempts) <= 8:
            raise ValueError('this crop cannot satisfy framing')
        return original(*args, **kwargs)
    monkeypatch.setattr(vtt, 'portrait_bytes', prepare_candidate)
    assert run(config, services, '1') == 0
    assert len(attempts) == 9 and winner(config)[1]['cue'] == 8
    with FaceDatabase(config.database) as db:
        assert db.connection.execute('SELECT COUNT(*) FROM portraits WHERE image IS NOT NULL').fetchone()[0] == 1
    assert stash.media_calls.count('http://stash/sprite/1.jpg') == 2  # scan and one shared portrait source


def test_visibility_weight_prefers_more_visible_face_and_threshold_reuses_measurements(tmp_path, monkeypatch):
    config, stash, services, _ = setup(tmp_path, monkeypatch)
    models = []
    class Measured(Visibility):
        def __init__(self, *args):
            super().__init__(*args)
            models.append(self)
        def measure(self, image, face):
            result = super().measure(image, face)
            result['visible_face_fraction'] = [.85, .95, .83][self.calls - 1]
            return result
    services.visibility = Measured
    assert run(config, services, '1') == 0
    before, metadata = winner(config)
    assert metadata['cue'] == 1 and metadata['quality']['visible_face_fraction'] == .95
    assert len(models) == 1 and models[0].calls == 3 and models[0].closed
    services.visibility = lambda *args: pytest.fail('changing policy must not repeat visibility inference')
    services.backend = lambda *args: pytest.fail('changing portrait policy must not repeat recognition')
    write_config(config.source_path, margin=.5, min_visible_face_fraction=.94)
    changed = load(config.source_path)
    assert namespace(config) == namespace(changed)
    assert run(changed, services, '1') == 0
    with FaceDatabase(config.database) as db:
        current = db.connection.execute("SELECT * FROM portraits WHERE status='published' AND id<>?", (before['id'],)).fetchone()
        assert json.loads(current['metadata'])['quality']['visible_face_fraction'] == .95
        assert json.loads(current['metadata'])['cue'] == 1
        assert db.connection.execute('SELECT COUNT(*) FROM scans').fetchone()[0] == 1
        assert all(json.loads(r[0])['visibility']['contract'] == CONTRACT_ID
                   for r in db.connection.execute('SELECT quality FROM observations'))


def test_visibility_disabled_skips_model_and_measurements(tmp_path, monkeypatch):
    config, stash, services, _ = setup(tmp_path, monkeypatch, min_visible_face_fraction=0, weights={'visibility': 0})
    services.visibility = lambda *args: pytest.fail('visibility is fully disabled')
    assert run(config, services, '1') == 0
    assert 'visible_face_fraction' not in winner(config)[1]['quality']


@pytest.mark.parametrize('source', ['vtt', 'video'])
def test_visibility_model_failure_ranks_other_quality_and_closes_session(tmp_path, monkeypatch, source, capsys):
    config, stash, services = setup(tmp_path, monkeypatch)[:3] if source == 'vtt' else video_setup(tmp_path, monkeypatch)
    model = Visibility()
    model.measure = lambda *args: {'visible_face_fraction': float('nan'), 'visibility': {'contract': CONTRACT_ID}}
    services.visibility = lambda *args: model
    assert run(config, services, '1') == 0
    assert stash.images and model.closed
    assert 'invalid or incompatible measurement' in capsys.readouterr().out


def test_video_ranks_occluded_actual_video_face_instead_of_leaving_profile_blank(tmp_path, monkeypatch):
    config, stash, services = video_setup(tmp_path, monkeypatch)
    model = Visibility()
    def occluded(*args):
        return {'visible_face_fraction': .5, 'visibility': {'contract': CONTRACT_ID, 'device': 'cpu'}}
    model.measure = occluded
    services.visibility = lambda *args: model
    assert run(config, services, '1') == 0
    assert stash.images and model.closed
    with FaceDatabase(config.database) as db:
        assert all('face_occluded' in json.loads(r[0])['quality_shortfalls']
                   for r in db.connection.execute('SELECT metadata FROM portraits'))


def test_video_candidate_budget_still_applies(tmp_path, monkeypatch):
    config, stash, services = video_setup(tmp_path, monkeypatch)
    write_config(config.source_path, source='video', margin=.1, max_candidates=1, time_offsets=[0, .5, 1])
    assert run(load(config.source_path), services, '1') == 0
    assert sum(len(d.calls) for d in Decoder.instances) == 3


def test_visibility_crop_contract_normalization_and_ratio_ignore_portrait_padding():
    face = checked_detection([64, 64, 128, 128, 96, 108, 160, 108, 128, 136, 105, 160, 151, 160, .99])
    with Image.new('RGB', (256, 256), (255, 128, 0)) as image:
        blob, region = prepare(image, face)
    assert blob.shape == (1, 3, 256, 256) and blob.dtype == np.float32 and blob.flags.c_contiguous
    assert blob[0, :, 128, 128] == pytest.approx([(1-.485)/.229, (128/255-.456)/.224, -.406/.225])
    logits = np.ones((1, 1, 256, 256), dtype=np.float32)
    assert measurement(logits, region, 'cpu')['visible_face_fraction'] == 1
    logits[:, :, 128:] = -1
    result = measurement(logits, region, 'cpu')
    assert result['visible_face_fraction'] == .5 and result['visibility']['face_pixels'] == int(region.sum())
    with pytest.raises(ValueError):
        measurement(np.full_like(logits, np.nan), region, 'cpu')
    with pytest.raises(ValueError):
        measurement(np.ones((256, 256)), region, 'cpu')


@pytest.mark.parametrize('value', ['-.1', '1.1', '.nan', 'true'])
def test_visibility_fraction_rejects_invalid_yaml_before_work(tmp_path, value):
    path = tmp_path / 'bad.yml'
    path.write_text(f'version: 1\nportraits: {{min_visible_face_fraction: {value}}}')
    with pytest.raises(ValueError):
        load(path)
    assert not (tmp_path / 'faces_state.sqlite3').exists()


def test_unknown_visibility_key_still_fails(tmp_path):
    path = tmp_path / 'bad.yml'
    path.write_text('version: 1\nportraits: {max_occlusion: .2}')
    with pytest.raises(ValueError, match='unknown keys'):
        load(path)


def test_changed_visibility_contract_is_recomputed_without_recognition(tmp_path, monkeypatch):
    config, stash, services, backends = setup(tmp_path, monkeypatch)
    assert run(config, services, '1') == 0
    with FaceDatabase(config.database) as db:
        for row in db.connection.execute('SELECT id,quality FROM observations').fetchall():
            quality = json.loads(row['quality'])
            quality['visibility']['contract'] = 'older-model-contract'
            db.connection.execute('UPDATE observations SET quality=? WHERE id=?', (json.dumps(quality), row['id']))
        db.connection.commit()
    model = Visibility()
    services.visibility = lambda *args: model
    write_config(config.source_path, margin=.5)
    assert run(load(config.source_path), services, '1') == 0
    assert model.calls == 3 and model.closed and len(backends) == 1


def test_zero_threshold_keeps_visibility_preference_active(tmp_path, monkeypatch):
    config, stash, services, _ = setup(tmp_path, monkeypatch, min_visible_face_fraction=0)
    model = Visibility()
    services.visibility = lambda *args: model
    assert run(config, services, '1') == 0
    assert model.calls == 3 and model.closed


def test_disabled_visibility_does_not_download_faceocc(tmp_path, monkeypatch):
    import stash_ai.backends.downloads as downloads
    import stash_ai.recognition.models as models
    import stash_ai.recognition.runtime as runtime
    config, _, _, _ = setup(tmp_path, monkeypatch, weights={'visibility': 0})
    downloaded = []
    monkeypatch.setattr(runtime, 'runtime', lambda *args: None)
    monkeypatch.setattr(models, 'verify', lambda *args: None)
    monkeypatch.setattr(downloads, 'install_hf_asset', lambda repo, filename, *args, **kwargs: downloaded.append(filename))
    assert set(models.ensure_assets(config)) == {'scrfd_10g', 'sface'}
    assert 'model.safetensors' not in downloaded


@pytest.mark.parametrize('requested,available,fail_cuda,expected', [
    ('cpu', True, False, 'cpu'), ('auto', False, False, 'cpu'), ('auto', True, True, 'cpu'),
    ('cuda', False, False, None), ('cuda', True, True, None), ('cuda', True, False, 'cuda')])
def test_visibility_device_policy_has_no_implicit_cuda_fallback(tmp_path, monkeypatch, requested, available, fail_cuda, expected):
    import torch
    import safetensors.torch
    import stash_ai.recognition.faceocc as faceocc
    import stash_ai.recognition.visibility as visibility
    calls = []
    class Model:
        def load_state_dict(self, value, strict): assert strict
        def to(self, device):
            calls.append(device)
            if device == 'cuda' and fail_cuda:
                raise RuntimeError('CUDA init failed')
            return self
        def eval(self): return self
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: available)
    monkeypatch.setattr(visibility, 'verified_bytes', lambda *args: b'weights')
    monkeypatch.setattr(safetensors.torch, 'load', lambda *args, **kwargs: {})
    monkeypatch.setattr(faceocc, 'build_model', Model)
    config = load(root=tmp_path, device=requested)
    if expected is None:
        with pytest.raises((RuntimeError, ValueError)):
            FaceVisibility(config, {'faceocc': 'test.safetensors'})
    else:
        model = FaceVisibility(config, {'faceocc': 'test.safetensors'})
        assert model.device == expected
        model.close()


def test_lazy_visibility_shares_and_closes_one_model():
    models = []
    def factory(*args):
        value = Visibility(); models.append(value); return value
    lazy = LazyVisibility(None, {}, factory)
    for _ in range(3):
        lazy.measure(None, None)
    lazy.close()
    assert len(models) == 1 and models[0].calls == 3 and models[0].closed
