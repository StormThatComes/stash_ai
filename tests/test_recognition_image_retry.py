import base64
import hashlib
import json

import pytest

from stash_ai.recognition.cli import parser
from stash_ai.recognition.config import load
from stash_ai.recognition.database import FaceDatabase
from stash_ai.recognition.run import execute
from tests.test_recognition_metadata import setup as video_setup
from tests.test_recognition_vtt_portraits import setup, write_config


def run(config, services, *ids, retry=False):
    argv = ['run', '--progress-style', 'off']
    for identifier in ids:
        argv += ['--scene-id', identifier]
    if retry:
        argv.append('--retry-unresolved-images')
    return execute(parser().parse_args(argv), config, services=services)


def image_operation(config):
    with FaceDatabase(config.database) as db:
        return dict(db.connection.execute("SELECT * FROM publications WHERE kind='image'").fetchone())


def lose_images(stash, *, applied=False):
    original, attempts = stash.update_person, []
    def lost(performer, **fields):
        if 'image' not in fields:
            return original(performer, **fields)
        attempts.append((performer, base64.b64decode(fields['image'].split(',', 1)[1])))
        if applied:
            original(performer, **fields)
        raise TimeoutError('lost image response')
    stash.update_person = lost
    return original, attempts


@pytest.mark.parametrize('source', ['vtt', 'video'])
@pytest.mark.parametrize('status', ['pending', 'unknown'])
def test_explicit_retry_reuses_checked_cached_jpeg_without_inference(tmp_path, monkeypatch, source, status):
    config, stash, services = (setup(tmp_path, monkeypatch)[:3] if source == 'vtt'
                               else video_setup(tmp_path, monkeypatch))
    original, attempts = lose_images(stash)
    assert run(config, services, '1') == 1
    previous = image_operation(config)
    with FaceDatabase(config.database) as db:
        db.connection.execute("UPDATE publications SET status=? WHERE kind='image'", (status,))
        db.connection.commit()
    assert run(config, services, '1') == 1
    assert len(attempts) == 1
    stash.update_person = original
    services.backend = lambda *args: pytest.fail('retry must not rerun inference')
    services.decoder = lambda *args: pytest.fail('retry must not decode video again')
    assert run(config, services, '1', retry=True) == 0
    performer = stash.links['1'][-1]
    assert stash.images[performer] == attempts[0][1]
    assert stash.links == {'1': ['77', performer], '2': []}
    with FaceDatabase(config.database) as db:
        current = db.connection.execute("SELECT * FROM publications WHERE kind='image'").fetchone()
        assert current['status'] == 'confirmed'
        assert json.loads(current['payload']) == json.loads(previous['payload'])
        audit = json.loads(current['precondition'])['retry_of']
        assert audit == {'id': previous['id'], 'status': status, 'payload': json.loads(previous['payload']),
                         'image_sha256': None, 'error': previous['error']}
        history = db.connection.execute('SELECT * FROM runs ORDER BY started DESC LIMIT 1').fetchone()
        assert json.loads(history['config'])['invocation'] == {'retry_unresolved_images': True}
        summary = json.loads(history['summary'])
        assert summary['counts']['portraits_retried'] == summary['counts']['portraits_filled'] == 1
        assert summary['mutations']['mutation_confirmed'] == 1
    writes = list(stash.writes)
    assert run(config, services, '1', retry=True) == 0
    assert stash.writes == writes  # opt-in is not a request to rewrite confirmed images


def test_retry_refreshes_old_owned_image_and_keeps_its_digest_in_audit(tmp_path, monkeypatch):
    config, stash, services, _ = setup(tmp_path, monkeypatch)
    assert run(config, services, '1') == 0
    performer = stash.links['1'][-1]
    baseline = hashlib.sha256(stash.images[performer]).hexdigest()
    original, attempts = lose_images(stash)
    write_config(config.source_path, margin=.5)
    config = load(config.source_path)
    assert run(config, services, '1') == 1
    assert run(config, services, '1') == 1
    stash.update_person = original
    assert run(config, services, '1', retry=True) == 0
    assert stash.images[performer] == attempts[0][1]
    current = image_operation(config)
    assert json.loads(current['precondition'])['retry_of']['image_sha256'] == baseline
    with FaceDatabase(config.database) as db:
        summary = json.loads(db.connection.execute('SELECT summary FROM runs ORDER BY started DESC LIMIT 1').fetchone()[0])
        assert summary['counts']['portraits_retried'] == summary['counts']['portraits_updated'] == 1


def test_retry_flag_reconciles_applied_write_without_duplicate_dispatch(tmp_path, monkeypatch):
    config, stash, services, _ = setup(tmp_path, monkeypatch)
    _, attempts = lose_images(stash, applied=True)
    assert run(config, services, '1') == 1
    assert run(config, services, '1', retry=True) == 0
    assert len(attempts) == 1
    assert image_operation(config)['status'] == 'observed'


def test_retry_flag_preserves_manual_replacement_and_changed_ownership(tmp_path, monkeypatch):
    config, stash, services, _ = setup(tmp_path, monkeypatch)
    _, attempts = lose_images(stash)
    assert run(config, services, '1') == 1
    performer = stash.links['1'][-1]
    stash.profiles[performer]['details'] = 'operator changed ownership'
    assert run(config, services, '1', retry=True) == 1
    assert len(attempts) == 1 and image_operation(config)['status'] == 'unknown'
    with FaceDatabase(config.database) as db:
        token = db.connection.execute('SELECT token FROM people').fetchone()[0]
    stash.profiles[performer]['details'] = token
    stash.profiles[performer]['image_path'] = f'http://stash/performer/{performer}/image?t=123'
    stash.images[performer] = b'manual replacement'
    assert run(config, services, '1', retry=True) == 0
    assert stash.images[performer] == b'manual replacement'
    assert len(attempts) == 1 and image_operation(config)['status'] == 'conflict'


@pytest.mark.parametrize('change', ['image', 'ownership'])
def test_retry_rechecks_live_state_after_cached_selection(tmp_path, monkeypatch, change):
    import stash_ai.recognition.portraits as portraits
    config, stash, services, _ = setup(tmp_path, monkeypatch)
    _, attempts = lose_images(stash)
    assert run(config, services, '1') == 1
    performer = stash.links['1'][-1]
    choose = portraits.choose
    def concurrent(*args, **kwargs):
        winner = choose(*args, **kwargs)
        if change == 'image':
            stash.profiles[performer]['image_path'] = f'http://stash/performer/{performer}/image?t=123'
            stash.images[performer] = b'concurrent manual image'
        else:
            stash.profiles[performer]['details'] = 'concurrent ownership change'
        return winner
    monkeypatch.setattr(portraits, 'choose', concurrent)
    assert run(config, services, '1', retry=True) == (0 if change == 'image' else 1)
    assert len(attempts) == 1 and image_operation(config)['status'] == 'unknown'


def test_retry_does_not_publish_portrait_rejected_by_current_quality(tmp_path, monkeypatch, capsys):
    config, stash, services, _ = setup(tmp_path, monkeypatch)
    _, attempts = lose_images(stash)
    assert run(config, services, '1') == 1
    previous = image_operation(config)
    write_config(config.source_path, min_face_pixels=200)
    assert run(load(config.source_path), services, '1', retry=True) == 1
    assert len(attempts) == 1 and image_operation(config) == previous
    assert 'no currently eligible portrait for retry' in capsys.readouterr().out


def test_repeated_unknown_retry_needs_opt_in_again_and_keeps_bounded_audit(tmp_path, monkeypatch):
    config, stash, services, _ = setup(tmp_path, monkeypatch)
    original, attempts = lose_images(stash)
    assert run(config, services, '1') == 1
    assert run(config, services, '1', retry=True) == 1
    assert run(config, services, '1') == 1
    assert len(attempts) == 2
    assert run(config, services, '1', retry=True) == 1
    assert len(attempts) == 3
    previous = image_operation(config)
    stash.update_person = original
    assert run(config, services, '1', retry=True) == 0
    current = image_operation(config)
    assert current['status'] == 'confirmed'
    audit = json.loads(current['precondition'])['retry_of']
    assert audit['payload'] == json.loads(previous['payload'])
    assert 'retry_of' not in audit and 'precondition' not in audit


def test_retry_revalidates_cached_video_identity_instead_of_trusting_old_selection(tmp_path, monkeypatch):
    config, stash, services = video_setup(tmp_path, monkeypatch)
    _, attempts = lose_images(stash)
    assert run(config, services, '1') == 1
    previous = image_operation(config)
    with FaceDatabase(config.database) as db:
        for row in db.connection.execute('SELECT id,metadata FROM portraits').fetchall():
            metadata = json.loads(row['metadata'])
            metadata['identity_embedding']['values'] = [0, 1, 0, 0]
            db.connection.execute('UPDATE portraits SET metadata=? WHERE id=?', (json.dumps(metadata), row['id']))
        db.connection.commit()
    services.backend = lambda *args: pytest.fail('do not infer to replace rejected cached identity')
    services.decoder = lambda *args: pytest.fail('do not decode to replace rejected cached identity')
    assert run(config, services, '1', retry=True) == 1
    assert image_operation(config) == previous and len(attempts) == 1


def test_retry_flag_only_visits_people_accepted_in_selected_scenes(tmp_path, monkeypatch):
    config, stash, services, _ = setup(tmp_path, monkeypatch)
    original, attempts = lose_images(stash)
    assert run(config, services, '1') == 1
    previous = image_operation(config)
    original_backend = services.backend
    def other_identity(*args):
        value = original_backend(*args)
        value.vector = (0, 1, 0, 0)
        return value
    services.backend = other_identity
    stash.update_person = original
    assert run(config, services, '2', retry=True) == 0
    assert len(attempts) == 1
    with FaceDatabase(config.database) as db:
        unchanged = dict(db.connection.execute('SELECT * FROM publications WHERE id=?', (previous['id'],)).fetchone())
        assert unchanged == previous
    assert stash.links['1'][-1] not in stash.images
    assert stash.links['2'][-1] in stash.images


@pytest.mark.parametrize('kind', ['create', 'link', 'name'])
def test_image_retry_never_retries_other_unknown_mutations(tmp_path, monkeypatch, kind):
    from dataclasses import replace
    config, stash, services, _ = setup(tmp_path, monkeypatch)
    if kind == 'create':
        stash.fail_create, stash.create_visible = True, False
    elif kind == 'link':
        def lost_link(*args):
            stash.writes.append(('link', *args))
            raise TimeoutError('lost invisible link')
        stash.add_scene_person = lost_link
    else:
        assert run(config, services, '1') == 0
        performer = stash.links['1'][-1]
        names = tmp_path / 'performer_names.yml'
        names.write_text(f'version: 1\nstash_scope: http://stash\nperformers: {{"{performer}": Requested}}')
        config = replace(config, names_path=names)
        def lost_name(*args, **kwargs):
            stash.writes.append(('name', args, kwargs))
            raise TimeoutError('lost invisible name')
        stash.update_person = lost_name
    assert run(config, services, '1') == 1
    writes = list(stash.writes)
    assert run(config, services, '1', retry=True) == 1
    assert stash.writes == writes


def test_retry_is_an_explicit_cli_option_and_not_a_yaml_setting(tmp_path):
    assert not parser().parse_args(['run', '--scene-id', '1']).retry_unresolved_images
    assert parser().parse_args(['run', '--all-scenes', '--retry-unresolved-images']).retry_unresolved_images
    with pytest.raises(SystemExit):
        parser().parse_args(['run', '--scene-id', '1', '--force'])
    path = tmp_path / 'faces.yml'
    path.write_text('version: 1\nexecution: {retry_unresolved_images: true}\n')
    with pytest.raises(ValueError, match='unknown keys'):
        load(path)
