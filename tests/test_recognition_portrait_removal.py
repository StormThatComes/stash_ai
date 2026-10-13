import hashlib
import json

import pytest
import yaml

from stash_ai.recognition.config import load
from stash_ai.recognition.database import FaceDatabase
from stash_ai.recognition.portraits import missing_image
from tests.test_recognition_image_retry import image_operation, run
from tests.test_recognition_metadata import setup as video_setup
from tests.test_recognition_vtt_portraits import setup as vtt_setup, image_writes


def setup(tmp_path, monkeypatch, source):
    return vtt_setup(tmp_path, monkeypatch)[:3] if source == 'vtt' else video_setup(tmp_path, monkeypatch)


def change(config, **values):
    data = yaml.safe_load(config.source_path.read_text())
    data['portraits'].update(values)
    config.source_path.write_text(yaml.safe_dump(data))
    return load(config.source_path)


def last_counts(config):
    with FaceDatabase(config.database) as db:
        return json.loads(db.connection.execute('SELECT summary FROM runs ORDER BY started DESC LIMIT 1').fetchone()[0])['counts']


@pytest.mark.parametrize('source', ['vtt', 'video'])
@pytest.mark.parametrize('restriction', ['quality', 'visibility', 'framing', 'disabled'])
def test_policy_rejection_removes_only_owned_photo_once_and_restoring_policy_refills(tmp_path, monkeypatch, source, restriction):
    config, stash, services = setup(tmp_path, monkeypatch, source)
    assert run(config, services, '1') == 0
    performer = stash.links['1'][-1]
    original_image = stash.images[performer]
    original_config = config.source_path.read_text()
    original_operation = image_operation(config)
    if restriction == 'quality':
        changed = change(config, min_face_pixels=200)
    elif restriction == 'visibility':
        changed = change(config, min_visible_face_fraction=.9, margin=.1)
        from stash_ai.recognition.visibility import CONTRACT_ID
        with FaceDatabase(config.database) as db:
            for row in db.connection.execute('SELECT id,quality FROM observations').fetchall():
                quality = json.loads(row['quality'])
                quality.update(visible_face_fraction=.5, visibility={'contract': CONTRACT_ID})
                db.connection.execute('UPDATE observations SET quality=? WHERE id=?', (json.dumps(quality), row['id']))
            db.connection.commit()
        if source == 'video':
            model = services.visibility()
            model.measure = lambda *args: {'visible_face_fraction': .5, 'visibility': {'contract': CONTRACT_ID}}
            services.visibility = lambda *args: model
    elif restriction == 'framing':
        changed = change(config, aspect_ratio=.1)
    else:
        changed = change(config, enabled=False)
    assert run(changed, services, '1') == 0
    assert performer not in stash.images
    assert missing_image(stash.performer(performer), stash, performer)
    assert stash.links['1'] == ['77', performer]
    assert last_counts(config)['portraits_removed'] == 1
    operation = image_operation(config)
    payload = json.loads(operation['payload'])
    assert operation['status'] == 'confirmed' and payload['action'] == 'remove' and payload['sha256'] is None
    assert json.loads(operation['precondition'])['image_sha256'] == hashlib.sha256(original_image).hexdigest()
    with FaceDatabase(config.database) as db:
        row = db.connection.execute('SELECT * FROM portraits WHERE id=?',
                                    (json.loads(original_operation['payload'])['portrait'],)).fetchone()
        assert row['status'] == 'removed' and row['image'] == original_image  # keep evidence, clear Stash only
        assert db.connection.execute('SELECT COUNT(*) FROM scans').fetchone()[0] == 1
    writes = list(stash.writes)
    assert run(changed, services, '1') == 0
    assert stash.writes == writes
    assert image_operation(config) == operation
    if restriction == 'visibility':
        # Restore cached evidence/model for the same original policy.
        with FaceDatabase(config.database) as db:
            for row in db.connection.execute('SELECT id,quality FROM observations').fetchall():
                quality = json.loads(row['quality'])
                quality['visible_face_fraction'] = 1.
                db.connection.execute('UPDATE observations SET quality=? WHERE id=?', (json.dumps(quality), row['id']))
            db.connection.commit()
        from tests.helpers.recognition import Visibility
        services.visibility = Visibility
    config.source_path.write_text(original_config)
    assert run(load(config.source_path), services, '1') == 0
    assert stash.images[performer] == original_image
    assert len(image_writes(stash)) == 3


@pytest.mark.parametrize('disabled', [False, True])
def test_manual_replacement_at_same_url_is_preserved_when_policy_rejects(tmp_path, monkeypatch, disabled):
    config, stash, services = setup(tmp_path, monkeypatch, 'vtt')
    assert run(config, services, '1') == 0
    performer = stash.links['1'][-1]
    stash.images[performer] = b'manual replacement'
    changed = change(config, enabled=not disabled, min_face_pixels=200)
    assert run(changed, services, '1') == 0
    assert stash.images[performer] == b'manual replacement' and len(image_writes(stash)) == 1


@pytest.mark.parametrize('applied', [False, True])
def test_uncertain_removal_reconciles_or_requires_explicit_retry(tmp_path, monkeypatch, applied):
    config, stash, services = setup(tmp_path, monkeypatch, 'vtt')
    assert run(config, services, '1') == 0
    performer = stash.links['1'][-1]
    changed = change(config, min_face_pixels=200)
    original, calls = stash.update_person, []
    def lost(identifier, **fields):
        assert fields == {'image': ''}
        calls.append(identifier)
        if applied:
            original(identifier, **fields)
        raise TimeoutError('lost image removal response')
    stash.update_person = lost
    assert run(changed, services, '1') == 1
    assert image_operation(config)['status'] == 'unknown'
    assert run(changed, services, '1') == (0 if applied else 1)
    assert len(calls) == 1
    if applied:
        current = image_operation(config)
        assert current['status'] == 'observed' and performer not in stash.images
        with FaceDatabase(config.database) as db:
            assert db.connection.execute("SELECT status FROM portraits WHERE id=?",
                (json.loads(current['payload'])['portrait'],)).fetchone()[0] == 'removed'
    else:
        previous = image_operation(config)
        stash.update_person = original
        assert run(changed, services, '1', retry=True) == 0
        current = image_operation(config)
        assert current['status'] == 'confirmed' and performer not in stash.images
        assert json.loads(current['precondition'])['retry_of']['payload'] == json.loads(previous['payload'])
        assert last_counts(config)['portraits_retried'] == last_counts(config)['portraits_removed'] == 1
        writes = list(stash.writes)
        assert run(changed, services, '1', retry=True) == 0 and stash.writes == writes


@pytest.mark.parametrize('kind', ['image', 'ownership'])
def test_concurrent_manual_change_before_removal_is_preserved(tmp_path, monkeypatch, kind):
    import stash_ai.recognition.portraits as portraits
    config, stash, services = setup(tmp_path, monkeypatch, 'vtt')
    assert run(config, services, '1') == 0
    performer = stash.links['1'][-1]
    changed = change(config, min_face_pixels=200)
    choose = portraits.choose
    def concurrent(*args, **kwargs):
        result = choose(*args, **kwargs)
        if kind == 'image':
            stash.images[performer] = b'concurrent manual replacement'
        else:
            stash.profiles[performer]['details'] = 'ownership changed'
        return result
    monkeypatch.setattr(portraits, 'choose', concurrent)
    assert run(changed, services, '1') == (0 if kind == 'image' else 1)
    assert performer in stash.images and len(image_writes(stash)) == 1


@pytest.mark.parametrize('source', ['vtt', 'video'])
def test_source_failure_after_policy_change_does_not_mean_no_eligible_portrait(tmp_path, monkeypatch, source, capsys):
    config, stash, services = setup(tmp_path, monkeypatch, source)
    assert run(config, services, '1') == 0
    initial = dict(stash.images)
    changed = change(config, margin=.11)
    if source == 'vtt':
        original = services.source
        calls = []
        def fail(client, scene, configuration):
            calls.append(scene['id'])
            if len(calls) > 1:  # let scan evidence be verified first
                raise TimeoutError('VTT unavailable during portrait selection')
            return original(client, scene, configuration)
        services.source = fail
    else:
        services.decoder = lambda *args: (_ for _ in ()).throw(TimeoutError('video unavailable'))
    assert run(changed, services, '1') == 1
    assert stash.images == initial and len(image_writes(stash)) == 1
    assert 'Portrait evaluation is incomplete' in capsys.readouterr().out


def test_removal_only_visits_people_accepted_in_selected_scenes(tmp_path, monkeypatch):
    config, stash, services = setup(tmp_path, monkeypatch, 'vtt')
    assert run(config, services, '1') == 0
    first = stash.links['1'][-1]
    original = services.backend
    def other(*args):
        value = original(*args)
        value.vector = (0, 1, 0, 0)
        return value
    services.backend = other
    assert run(config, services, '2') == 0
    second = stash.links['2'][-1]
    assert first != second
    changed = change(config, min_face_pixels=200)
    assert run(changed, services, '2') == 0
    assert first in stash.images and second not in stash.images
    assert len(image_writes(stash)) == 3


@pytest.mark.parametrize('source', ['vtt', 'video'])
def test_visibility_model_failure_retains_old_photo_instead_of_removing_it(tmp_path, monkeypatch, source):
    config, stash, services = setup(tmp_path, monkeypatch, source)
    assert run(config, services, '1') == 0
    initial = dict(stash.images)
    changed = change(config, margin=.11)
    if source == 'vtt':
        with FaceDatabase(config.database) as db:
            for row in db.connection.execute('SELECT id,quality FROM observations').fetchall():
                quality = json.loads(row['quality'])
                quality['visibility']['contract'] = 'stale-visibility-contract'
                db.connection.execute('UPDATE observations SET quality=? WHERE id=?', (json.dumps(quality), row['id']))
            db.connection.commit()
    from tests.helpers.recognition import Visibility
    model = Visibility()
    model.measure = lambda *args: (_ for _ in ()).throw(RuntimeError('visibility unavailable'))
    services.visibility = lambda *args: model
    assert run(changed, services, '1') == 1
    assert stash.images == initial and len(image_writes(stash)) == 1


def test_manual_replacement_after_uncertain_removal_is_not_removed_on_retry(tmp_path, monkeypatch):
    config, stash, services = setup(tmp_path, monkeypatch, 'vtt')
    assert run(config, services, '1') == 0
    performer = stash.links['1'][-1]
    changed = change(config, min_face_pixels=200)
    def lost(*args, **kwargs):
        raise TimeoutError('removal was not applied')
    original = stash.update_person
    stash.update_person = lost
    assert run(changed, services, '1') == 1
    stash.images[performer] = b'manual replacement after uncertain removal'
    stash.update_person = original
    assert run(changed, services, '1', retry=True) == 0
    assert image_operation(config)['status'] == 'conflict'
    assert stash.images[performer] == b'manual replacement after uncertain removal'
    assert len(image_writes(stash)) == 1
