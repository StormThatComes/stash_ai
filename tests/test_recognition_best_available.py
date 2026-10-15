"""Best available portraits illustrate accepted identities without ideal-quality vetoes."""
from io import BytesIO
import json
import math

import numpy as np
from PIL import Image
import pytest

from stash_ai.recognition.config import defaults, load
from stash_ai.recognition.database import FaceDatabase
from stash_ai.recognition.models import checked_detection, namespace
from stash_ai.recognition.quality import portrait_bytes, portrait_score, rejection
from stash_ai.recognition.scanning import scan_scene
from stash_ai.recognition.clustering import cluster_scene
from stash_ai.recognition.publication import publish_person
from stash_ai.recognition.run import instance_scope
from stash_ai.recognition.media import SpriteInput
from stash_ai.recognition.visibility import CONTRACT_ID
from stash_ai.runtime.report import RunReport
from tests.helpers.recognition import Backend, MediaClient
from tests.test_recognition_vtt_portraits import setup, write_config, run, image_writes, image_size
from tests.test_recognition_portrait_removal import change, last_counts
from tests.test_recognition_image_retry import image_operation
from tests.test_recognition_metadata import setup as video_setup, Decoder


def published(config):
    operation = image_operation(config)
    with FaceDatabase(config.database) as db:
        row = db.connection.execute('SELECT * FROM portraits WHERE id=?',
                                     (json.loads(operation['payload'])['portrait'],)).fetchone()
        return dict(row), json.loads(row['metadata'])


def test_every_candidate_can_miss_every_quality_goal_and_still_supply_a_portrait(tmp_path, monkeypatch):
    config, stash, services, backends = setup(tmp_path, monkeypatch, enabled=False)
    assert run(config, services, '1') == 0
    with FaceDatabase(config.database) as db:
        for row in db.connection.execute('SELECT id,quality FROM observations').fetchall():
            quality = json.loads(row['quality'])
            quality.update(pixels=32, sharpness=1., pose_deviation=.9, exposure=250., clipped_fraction=.8,
                           complete=False, landmarks_inside=False, visible_face_fraction=.2,
                           visibility={'contract': CONTRACT_ID})
            db.connection.execute('UPDATE observations SET quality=? WHERE id=?', (json.dumps(quality), row['id']))
        db.connection.commit()
    write_config(config.source_path, min_face_pixels=200, min_sharpness=100, max_pose_deviation=.1,
                 max_exposure=200, max_clipped_fraction=.1, min_visible_face_fraction=.95)
    services.backend = services.visibility = lambda *a: pytest.fail('reuse recognition and quality evidence')
    assert run(load(config.source_path), services, '1') == 0
    row, metadata = published(config)
    assert row['image'] and row['image'] == stash.images[stash.links['1'][-1]]
    assert set(metadata['quality_shortfalls']) == {'face_too_small', 'face_blurred', 'face_incomplete',
        'face_pose', 'face_exposure', 'face_clipped_exposure', 'face_occluded'}
    assert math.isfinite(metadata['score']) and 0 < metadata['score'] < 1
    assert len(backends) == 1 and len(image_writes(stash)) == 1
    assert last_counts(config)['portraits_quality_limited'] == 1
    assert run(load(config.source_path), services, '1') == 0 and len(image_writes(stash)) == 1


def test_quality_goal_changes_reuse_thumbnail_bytes_without_sources_or_models(tmp_path, monkeypatch):
    config, stash, services, _ = setup(tmp_path, monkeypatch)
    assert run(config, services, '1') == 0
    original, before = published(config)
    changed = change(config, min_face_pixels=400, min_sharpness=100000, max_pose_deviation=0)
    original_source, calls = services.source, []
    def source(client, scene, configuration):
        calls.append(scene['id'])
        assert len(calls) == 1  # the scan checks its current manifest; portrait uses a cached JPEG
        return original_source(client, scene, configuration)
    services.source = source
    services.backend = services.visibility = lambda *a: pytest.fail('no inference for soft goal changes')
    assert run(changed, services, '1') == 0
    current, metadata = published(config)
    assert current['image'] == original['image'] and metadata['reused_from'] == original['id']
    assert metadata['score'] < before['score'] and metadata['quality_shortfalls']
    assert len(image_writes(stash)) == 1
    assert last_counts(config)['portraits_quality_limited'] == 1


@pytest.mark.parametrize('feature', ['visibility', 'expression'])
def test_optional_gpu_quality_failure_does_not_leave_an_identity_blank_or_retry_init_per_face(tmp_path, monkeypatch, feature):
    config, stash, services, _ = setup(tmp_path, monkeypatch, weights={'closed_mouth': 2})
    calls = []
    def failed(configuration, paths):
        calls.append(configuration.data['models']['device'])
        raise RuntimeError('quality CUDA initialization failed')
    setattr(services, feature, failed)
    assert run(config, services, '1') == 0
    _, metadata = published(config)
    assert calls == ['cuda'] and stash.images
    assert feature in metadata['quality_errors']
    assert ('face_visibility_unavailable' if feature == 'visibility' else 'face_expression_unavailable') in metadata['quality_shortfalls']
    assert last_counts(config).get('metadata_failed', 0) == 0


@pytest.mark.parametrize('asset,feature', [('faceocc', 'visibility'), ('expression_mesh', 'expression'),
                                         ('expression_blendshapes', 'expression')])
def test_optional_asset_download_failure_does_not_block_recognition_or_portraits(tmp_path, monkeypatch, asset, feature, capsys):
    import stash_ai.backends.downloads as downloads
    import stash_ai.recognition.models as models
    import stash_ai.recognition.runtime as runtime
    config, stash, services, _ = setup(tmp_path, monkeypatch, weights={'closed_mouth': 2})
    attempts = []
    monkeypatch.setattr(runtime, 'runtime', lambda *args: None)
    monkeypatch.setattr(models, 'verify', lambda *args: None)
    def download(repo, filename, *args, **kwargs):
        attempts.append(filename)
        if filename == models.ASSETS[asset].filename:
            raise OSError('quality download unavailable')
    monkeypatch.setattr(downloads, 'install_hf_asset', download)
    services.assets = models.ensure_assets
    def failed(configuration, paths):
        assert paths[asset] == config.model_dir / asset / models.ASSETS[asset].filename
        raise FileNotFoundError('quality asset is missing')
    setattr(services, feature, failed)
    assert run(config, services, '1') == 0 and stash.images
    assert attempts.count(models.ASSETS[asset].filename) == 1
    assert feature in published(config)[1]['quality_errors']
    assert f'Portrait quality asset {asset} unavailable' in capsys.readouterr().out
    assert last_counts(config).get('metadata_failed', 0) == 0


def test_identity_asset_download_failure_remains_a_hard_error(tmp_path, monkeypatch):
    import stash_ai.backends.downloads as downloads
    import stash_ai.recognition.models as models
    import stash_ai.recognition.runtime as runtime
    monkeypatch.setattr(runtime, 'runtime', lambda *args: None)
    def failed(*args, **kwargs):
        raise OSError('identity download unavailable')
    monkeypatch.setattr(downloads, 'install_hf_asset', failed)
    with pytest.raises(OSError, match='identity download unavailable'):
        models.ensure_assets(load(root=tmp_path), write=lambda _: None)


def test_invalid_optional_asset_is_preserved_and_never_loaded(tmp_path, monkeypatch):
    import stash_ai.backends.downloads as downloads
    import stash_ai.recognition.models as models
    import stash_ai.recognition.runtime as runtime
    from stash_ai.recognition.visibility import FaceVisibility
    config, stash, services, _ = setup(tmp_path, monkeypatch)
    path = config.model_dir / 'faceocc' / models.ASSETS['faceocc'].filename
    path.parent.mkdir(parents=True)
    path.write_bytes(b'invalid custom model')
    verify = models.verify
    monkeypatch.setattr(runtime, 'runtime', lambda *args: None)
    monkeypatch.setattr(models, 'verify', lambda current, asset: verify(current, asset) if current == path else None)
    def download(repo, filename, *args, **kwargs):
        assert filename != models.ASSETS['faceocc'].filename
    monkeypatch.setattr(downloads, 'install_hf_asset', download)
    services.assets, services.visibility = models.ensure_assets, FaceVisibility
    assert run(config, services, '1') == 0 and stash.images
    assert path.read_bytes() == b'invalid custom model'
    assert 'wrong asset size' in published(config)[1]['quality_errors']['visibility']


def test_zero_weights_still_choose_best_general_quality_without_optional_models(tmp_path, monkeypatch):
    policy = defaults()['portraits']
    config, stash, services, _ = setup(tmp_path, monkeypatch, enabled=False,
                                     min_face_pixels=0, weights=dict.fromkeys(policy['weights'], 0))
    assert run(config, services, '1') == 0
    with FaceDatabase(config.database) as db:
        for row in db.connection.execute('SELECT id,cue,quality FROM observations').fetchall():
            quality = json.loads(row['quality'])
            quality['sharpness'] = [0., 20., 500.][row['cue']]
            db.connection.execute('UPDATE observations SET quality=? WHERE id=?', (json.dumps(quality), row['id']))
        db.connection.commit()
    config = change(config, enabled=True)
    services.visibility = services.expression = lambda *a: pytest.fail('zero weights disable optional measurements')
    assert run(config, services, '1') == 0
    assert published(config)[1]['cue'] == 2


def test_existing_good_portrait_is_not_replaced_by_a_worse_new_scene(tmp_path, monkeypatch):
    config, stash, services, _ = setup(tmp_path, monkeypatch)
    assert run(config, services, '1') == 0
    original = dict(stash.images)
    with Image.new('RGB', (500, 350), (128, 128, 128)) as image:
        output = BytesIO(); image.save(output, 'PNG'); stash.sprite = output.getvalue()
    stash.media_calls.clear()
    assert run(config, services, '2') == 0
    assert stash.images == original and len(image_writes(stash)) == 1
    assert all('/1.' not in url for url in stash.media_calls)


def test_better_new_scene_upgrades_an_imperfect_portrait_and_repeat_is_idempotent(tmp_path, monkeypatch):
    config, stash, services, _ = setup(tmp_path, monkeypatch, min_sharpness=25)
    better = stash.sprite
    with Image.new('RGB', (500, 350), (128, 128, 128)) as image:
        output = BytesIO(); image.save(output, 'PNG'); stash.sprite = output.getvalue()
    assert run(config, services, '1') == 0 and stash.images
    original = dict(stash.images)
    assert 'face_blurred' in published(config)[1]['quality_shortfalls']
    stash.sprite = better
    assert run(config, services, '2') == 0
    assert stash.images != original and len(image_writes(stash)) == 2
    assert published(config)[1]['scene'] == '2'
    operation = image_operation(config)
    assert run(config, services, '2') == 0
    assert len(image_writes(stash)) == 2 and image_operation(config) == operation


def test_retained_crop_is_upgraded_to_full_thumbnail_when_source_recovers(tmp_path, monkeypatch):
    config, stash, services, _ = setup(tmp_path, monkeypatch)
    original, calls = services.source, []
    def unavailable(client, scene, configuration):
        calls.append(scene['id'])
        if len(calls) > 1:
            raise TimeoutError('thumbnail unavailable after scan')
        return original(client, scene, configuration)
    services.source = unavailable
    assert run(config, services, '1') == 0
    assert published(config)[1]['source_type'] == 'recognition_crop'
    assert image_size(next(iter(stash.images.values()))) == (130, 130)
    services.source = original
    assert run(config, services, '1') == 0
    assert published(config)[1]['source_type'] == 'vtt'
    assert image_size(next(iter(stash.images.values()))) == (170, 170)
    assert len(image_writes(stash)) == 2
    assert run(config, services, '1') == 0 and len(image_writes(stash)) == 2


def test_repeated_source_failure_keeps_the_retained_portrait_idempotent(tmp_path, monkeypatch):
    config, stash, services, _ = setup(tmp_path, monkeypatch)
    original = services.source
    def unavailable(client, scene, configuration):
        source = original(client, scene, configuration)
        def thumbnail(index):
            raise TimeoutError('thumbnail unavailable after scan')
        source.thumbnail = thumbnail
        return source
    services.source = unavailable
    assert run(config, services, '1') == 0 and stash.images
    operation = image_operation(config)
    assert run(config, services, '1') == 0
    assert published(config)[1]['source_type'] == 'recognition_crop'
    assert len(image_writes(stash)) == 1 and image_operation(config) == operation


def test_video_backend_inference_failure_uses_the_accepted_thumbnail(tmp_path, monkeypatch):
    config, stash, services = video_setup(tmp_path, monkeypatch)
    calls, backends = [], []
    def backend(*args):
        calls.append(1)
        model = Backend(fail_at=4 if not backends else 1)
        backends.append(model)
        return model
    services.backend = backend
    assert run(config, services, '1') == 0
    _, metadata = published(config)
    assert metadata['source_type'] == 'vtt' and metadata['requested_source'] == 'video'
    assert metadata['fallback']['reason'] == 'no_usable_identity_verified_video_portrait'
    assert len(calls) == 2 and Decoder.instances and all(model.closed for model in backends)
    assert last_counts(config).get('metadata_failed', 0) == 0


def test_optional_quality_recovers_from_cached_video_without_seeking_or_rewriting(tmp_path, monkeypatch):
    config, stash, services = video_setup(tmp_path, monkeypatch)
    def failed(*args):
        raise RuntimeError('visibility CUDA initialization failed')
    services.visibility = failed
    assert run(config, services, '1') == 0
    original, before = published(config)
    assert 'face_visibility_unavailable' in before['quality_shortfalls']
    writes, decoders = list(stash.writes), list(Decoder.instances)
    from tests.helpers.recognition import Visibility
    services.visibility = Visibility
    services.backend = services.decoder = lambda *a: pytest.fail('cached pixels need no face inference or video seek')
    assert run(config, services, '1') == 0
    current, metadata = published(config)
    assert current['image'] == original['image'] and metadata['score'] > before['score']
    assert metadata['quality_source'] == 'cached_portrait' and metadata['quality_errors'] == {}
    assert 'face_visibility_unavailable' not in metadata['quality_shortfalls']
    assert stash.writes == writes and Decoder.instances == decoders


def test_cold_replay_portrait_uses_the_face_assigned_to_current_cluster(tmp_path, monkeypatch):
    config, stash, services, _ = setup(tmp_path, monkeypatch)
    scope = instance_scope(stash.url)
    media = MediaClient('99', count=32)
    with FaceDatabase(config.database) as db:
        with SpriteInput(media, media.scene, config) as source:
            scan, _ = scan_scene(db, scope, media.scene, config, namespace(config), source, Backend)
        person = cluster_scene(db, scan, config)[0]['person']
        performer = publish_person(db, stash, RunReport(write=lambda _: None), scope, '99', person)
    query = np.array([.68, .64, math.sqrt(1 - .68**2 - .64**2), 0.])
    values = iter([query] * 3 + [np.array([0.,1.,0.,0.])] * 3)
    model = Backend()
    model.embed = lambda *args: next(values)
    services.backend = lambda *a: model
    assert run(config, services, '2', '3') == 0
    assert performer in stash.images and len(stash.images) == 1
    with FaceDatabase(config.database) as db:
        photo = db.connection.execute('SELECT metadata FROM portraits WHERE person=? AND status=?', (person, 'published')).fetchone()
        metadata = json.loads(photo[0])
        assert metadata['scene'] == '2'
        assert db.connection.execute("SELECT COUNT(*) FROM candidates c JOIN scans s ON s.id=c.scan WHERE s.scene='3' AND c.status='withheld'").fetchone()[0] == 3


def test_impossible_ratio_and_byte_budget_adapt_framing_without_upscaling():
    policy = defaults()['portraits']
    policy.update(aspect_ratio=.1, jpeg_quality=100)
    pixels = np.random.default_rng(23).integers(0, 256, (500, 500, 3), dtype='uint8')
    face = checked_detection([50,50,400,400,150,150,350,150,250,250,170,350,330,350,.99])
    details = {}
    with Image.fromarray(pixels) as image:
        data, bounds = portrait_bytes(image, face, policy, limit=1024, details=details)
    assert len(data) <= 1024 and details['aspect_ratio_adjusted'] and details['downscaled']
    assert image_size(data) == tuple(details['pixels'])
    assert image_size(data)[0] <= bounds[2]-bounds[0] and image_size(data)[1] <= bounds[3]-bounds[1]


def test_detection_thresholds_stay_hard_and_portrait_goals_are_only_preferences():
    data = defaults()
    poor = {'pixels': 16, 'sharpness': 1., 'pose_deviation': 1.5, 'exposure': 250., 'clipped_fraction': .9,
            'complete': False, 'landmarks_inside': False}
    assert rejection(poor, data['detection']) == 'face_too_small'
    assert math.isfinite(portrait_score(poor, data['portraits']))
    assert 0 < portrait_score(poor, data['portraits']) < 1
