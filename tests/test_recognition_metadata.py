from dataclasses import replace
import json
from types import SimpleNamespace

import numpy as np
import pytest

from stash_ai.recognition.config import load
from stash_ai.recognition.database import FaceDatabase
from stash_ai.recognition.names import load_names, plan_names, sync_name
from stash_ai.recognition.portraits import missing_image
from stash_ai.recognition.publication import PendingRecognition
from stash_ai.recognition.run import Services, execute, instance_scope
from tests.helpers.recognition import Backend, Stash, Visibility, textured_image


class Decoder:
    instances = []
    url = 'http://stash/scene/1/stream?apikey=secret'

    def __init__(self, client, scene, config):
        self.scene = scene['id']
        self.calls, self.closed = [], False
        self.instances.append(self)

    def __enter__(self):
        return self

    def frame(self, seconds):
        self.calls.append(seconds)
        return textured_image(), {'requested_seconds': seconds, 'actual_seconds': seconds + .04}

    def close(self):
        self.closed = True


def setup(tmp_path, monkeypatch, *, names=False):
    Decoder.instances.clear()
    monkeypatch.setenv('STASH_URL', 'http://stash/')
    path = tmp_path / 'facial_recognition.yml'
    path.write_text('version: 1\ndetection: {min_sharpness: 0}\n'
                    'portraits: {source: video, min_sharpness: 0, margin: 0.1, max_candidates: 2, time_offsets: [0]}\n'
                    + ('names: {file: performer_names.yml}\n' if names else ''))
    config = load(path)
    stash = Stash()
    services = Services(client=lambda **kwargs: stash, assets=lambda *a, **k: {},
                        backend=lambda *a: Backend(), decoder=Decoder, visibility=Visibility)
    return config, stash, services


def run(config, services, *ids):
    return execute(SimpleNamespace(scene_id=list(ids), all_scenes=False, progress_style='off'), config, services=services)


def test_automatic_video_portrait_is_scoped_stored_and_rerun_preserves_image(tmp_path, monkeypatch):
    config, stash, services = setup(tmp_path, monkeypatch)
    assert run(config, services, '1') == 0
    performer = stash.links['1'][-1]
    assert performer in stash.images
    assert all(d.closed and d.scene == '1' for d in Decoder.instances)
    with FaceDatabase(config.database) as db:
        rows = db.connection.execute('SELECT * FROM portraits').fetchall()
        assert len(rows) == 2
        assert sum(row['status'] == 'published' for row in rows) == 1
        metadata = json.loads(rows[0]['metadata'])
        assert metadata['actual_seconds'] == metadata['requested_seconds'] + .04
        assert metadata['source'] == 'http://stash/scene/1/stream'
        assert metadata['quality']['visibility_method'].startswith('five-landmark')
        assert rows[0]['image'] and rows[0]['digest']
    writes = list(stash.writes)
    assert run(config, services, '2') == 0
    assert len(Decoder.instances) == 1  # existing portrait prevents other video fetches
    assert sum(w[0] == 'metadata' for w in stash.writes) == 1


def test_wrong_identity_video_is_skipped_without_breaking_links(tmp_path, monkeypatch):
    config, stash, services = setup(tmp_path, monkeypatch)
    calls = 0
    def backend(*args):
        nonlocal calls
        calls += 1
        return Backend(vector=(1, 0, 0, 0) if calls == 1 else (0, 1, 0, 0))
    services.backend = backend
    assert run(config, services, '1') == 0
    assert len(stash.links['1']) == 2 and not stash.images
    with FaceDatabase(config.database) as db:
        metadata = json.loads(db.connection.execute('SELECT metadata FROM portraits').fetchone()[0])
        assert metadata['exclusions'][0]['reason'] == 'portrait_identity_mismatch'


def test_image_supplied_during_extraction_is_preserved(tmp_path, monkeypatch):
    config, stash, services = setup(tmp_path, monkeypatch)
    class Concurrent(Decoder):
        def frame(self, seconds):
            identifier = stash.links['1'][-1]
            stash.profiles[identifier]['image_path'] = 'http://stash/manual-image.jpg'
            return super().frame(seconds)
    services.decoder = Concurrent
    assert run(config, services, '1') == 0
    assert not any(w[0] == 'metadata' for w in stash.writes)


def test_missing_placeholder_contract_only_accepts_correct_origin_and_id():
    client = Stash()
    assert missing_image({'image_path': 'http://stash/performer/123/image?t=1&default=true'}, client, '123')
    assert not missing_image({'image_path': 'http://stash/performer/123/image?t=1'}, client, '123')
    assert not missing_image({'image_path': 'http://else/performer/123/image?default=true'}, client, '123')
    assert not missing_image({'image_path': 'http://stash/performer/456/image?default=true'}, client, '123')


def test_names_snapshot_and_selected_id_intersection(tmp_path, monkeypatch):
    config, stash, services = setup(tmp_path, monkeypatch)
    assert run(config, services, '1') == 0
    performer = stash.links['1'][-1]
    names = tmp_path / 'performer_names.yml'
    names.write_text(f'version: 1\nstash_scope: http://stash\nperformers:\n  "{performer}": "Żółć 漢字"\n  "77": Other\n', encoding='utf-8')
    config = replace(config, names_path=names)
    assert run(config, services, '1') == 0
    assert stash.profiles[performer]['name'] == 'Żółć 漢字'
    assert stash.profiles['77']['name'] == 'Manual'
    with FaceDatabase(config.database) as db:
        row = db.connection.execute('SELECT config FROM runs ORDER BY started DESC LIMIT 1').fetchone()
        assert json.loads(row[0])['names']['performers'][performer] == 'Żółć 漢字'
    writes = len(stash.writes)
    assert run(config, services, '1') == 0
    assert len(stash.writes) == writes


@pytest.mark.parametrize('value', [
    'version: true\nstash_scope: http://stash\nperformers: {}',
    'version: 1\nstash_scope: http://other\nperformers: {}',
    'version: 1\nstash_scope: http://stash\nperformers: {123: Foo}',
    'version: 1\nstash_scope: http://stash\nperformers: {"123": false}',
    'version: 1\nstash_scope: http://stash\nperformers: {"123": ""}',
    'version: 1\nstash_scope: http://stash\nperformers: {"123": Foo, "123": Bar}',
    'version: 1\nstash_scope: http://stash\nperformers: {"123": "A\\nB"}',
])
def test_bad_names_fail_before_stash_database_and_models(tmp_path, monkeypatch, value):
    config, stash, services = setup(tmp_path, monkeypatch, names=True)
    config.names_path.write_text(value)
    services.client = lambda **kwargs: pytest.fail('no Stash before valid mapping')
    with pytest.raises(ValueError):
        run(config, services, '1')
    assert not config.database.exists()


def test_name_concurrent_edit_and_unknown_write_are_not_overwritten(tmp_path, monkeypatch):
    config, stash, services = setup(tmp_path, monkeypatch)
    assert run(config, services, '1') == 0
    performer = stash.links['1'][-1]
    mapping = tmp_path / 'performer_names.yml'
    mapping.write_text(f'version: 1\nstash_scope: http://stash\nperformers: {{"{performer}": Requested}}')
    config = replace(config, names_path=mapping)
    scope = instance_scope(stash.url)
    with FaceDatabase(config.database) as db:
        person = db.connection.execute('SELECT id FROM people').fetchone()[0]
        target = {'person': person, 'performer': performer}
        plan = plan_names(db, stash, None, config, load_names(config, scope), scope, [target])
        stash.profiles[performer]['name'] = 'Concurrent'
        with pytest.raises(PendingRecognition, match='Concurrent'):
            sync_name(db, stash, None, scope, target, plan)
        plan = plan_names(db, stash, None, config, load_names(config, scope), scope, [target])
        update = stash.update_person
        def lost(*args, **kwargs):
            update(*args, **kwargs)
            raise TimeoutError('lost name response')
        stash.update_person = lost
        with pytest.raises(TimeoutError):
            sync_name(db, stash, None, scope, target, plan)
        assert sync_name(db, stash, None, scope, target, plan) == {'names_unchanged': 1}


def test_portrait_quality_ranks_complete_frontal_sharp_face():
    from stash_ai.recognition.quality import portrait_score, rejection
    from stash_ai.recognition.config import defaults
    policy = defaults()['portraits']
    good = dict(complete=True, landmarks_inside=True, pixels=128, sharpness=500,
                exposure=128, clipped_fraction=0, pose_deviation=.05)
    for changed, reason in [({'complete': False}, 'face_incomplete'),
                           ({'landmarks_inside': False}, 'face_incomplete'),
                           ({'sharpness': 1}, 'face_blurred'), ({'pixels': 32}, 'face_too_small'),
                           ({'pose_deviation': .8}, 'face_pose'), ({'exposure': 250}, 'face_exposure')]:
        assert rejection({**good, **changed}, policy, portrait=True) == reason
    assert portrait_score(good, policy) > portrait_score({**good, 'pose_deviation': .3, 'sharpness': 30}, policy)


def test_lost_image_acknowledgement_is_reconciled_without_retry(tmp_path, monkeypatch):
    config, stash, services = setup(tmp_path, monkeypatch)
    update = stash.update_person
    def lost(*args, **kwargs):
        update(*args, **kwargs)
        raise TimeoutError('lost image response')
    stash.update_person = lost
    assert run(config, services, '1') == 1
    assert run(config, services, '1') == 0
    assert sum(w[0] == 'metadata' for w in stash.writes) == 1
    with FaceDatabase(config.database) as db:
        assert db.connection.execute("SELECT status FROM publications WHERE kind='image'").fetchone()[0] == 'observed'


def test_multiscene_portrait_shortlist_respects_selected_scene_diversity(tmp_path, monkeypatch):
    config, stash, services = setup(tmp_path, monkeypatch)
    assert run(config, services, '1', '2') == 0
    assert {d.scene for d in Decoder.instances} == {'1', '2'}
    assert sum(w[0] == 'metadata' for w in stash.writes) == 1


def test_missing_target_in_name_preflight_prevents_partial_name_batch(tmp_path, monkeypatch):
    config, stash, services = setup(tmp_path, monkeypatch)
    assert run(config, services, '1', '2') == 0
    performer = stash.links['1'][-1]
    path = tmp_path / 'performer_names.yml'
    path.write_text(f'version: 1\nstash_scope: http://stash\nperformers: {{"{performer}": First, "999": Missing}}')
    config = replace(config, names_path=path)
    scope = instance_scope(stash.url)
    with FaceDatabase(config.database) as db:
        person = db.connection.execute('SELECT id FROM people').fetchone()[0]
        with pytest.raises(KeyError):
            plan_names(db, stash, None, config, load_names(config, scope), scope,
                       [{'person': person, 'performer': performer}, {'person': person, 'performer': '999'}])
    assert stash.profiles[performer]['name'].startswith('AI Person ')
