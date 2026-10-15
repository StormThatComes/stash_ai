"""Recognition diagnostic grouping retains bounded evidence without URL secrets."""
import json
from types import SimpleNamespace

import pytest

from stash_ai.recognition.database import FaceDatabase
from stash_ai.recognition.diagnostics import message
from stash_ai.recognition.media import safe_url
from stash_ai.recognition.portrait_removal import retry_precondition
from stash_ai.recognition.run import execute
from stash_ai.stash_api.transport import safe_error
from tests.test_recognition_run import setup as run_setup
from tests.test_recognition_metadata import setup as video_setup
from tests.test_recognition_vtt_portraits import setup as vtt_setup


def arguments(*ids, style='off'):
    return SimpleNamespace(scene_id=list(ids), all_scenes=False, progress_style=style)


def history(config):
    with FaceDatabase(config.database) as db:
        row = db.connection.execute('SELECT * FROM runs ORDER BY started DESC LIMIT 1').fetchone()
        return dict(row), json.loads(row['summary'])


@pytest.mark.parametrize('style', ['auto', 'compact', 'off', 'verbose'])
def test_distinct_same_type_scan_failures_are_visible_and_durable(tmp_path, monkeypatch, capsys, style):
    config, client, services, _ = run_setup(tmp_path, monkeypatch)
    reasons = {'1': 'synthetic missing VTT', '2': 'synthetic sprite read timeout'}
    def source(reader, scene, config): raise RuntimeError(reasons[scene['id']])
    services.source = source
    assert execute(arguments('1', '2', style=style), config, services=services) == 1
    output = capsys.readouterr()
    assert all(reason in output.out for reason in reasons.values())
    assert 'Recognition errors scan/RuntimeError: 2' in output.out
    row, summary = history(config)
    errors = summary['errors']
    assert row['status'] == 'partial' and summary['counts']['scenes_failed'] == 2
    assert errors['format'] == 1 and errors['total'] == 2 and errors['overflow'] == 0
    assert errors['groups'][0]['count'] == 2 and len(errors['groups'][0]['examples']) == 2
    assert all(reason in json.dumps(errors) for reason in reasons.values())
    assert client.closed


def test_publication_groups_keep_distinct_reasons_without_blind_write_retries(tmp_path, monkeypatch, capsys):
    config, client, services, _ = run_setup(tmp_path, monkeypatch)
    calls = []
    def publish(scene, person):
        calls.append(scene)
        raise RuntimeError('synthetic publication failure for scene ' + scene)
    client.add_scene_person = publish
    assert execute(arguments('1', '2'), config, services=services) == 1
    _, summary = history(config)
    group = summary['errors']['groups'][0]
    assert group['source'] == 'publication' and group['count'] == 2
    assert calls == ['1', '2'] and all(scene in json.dumps(group) for scene in calls)
    assert 'Recognition errors publication/RuntimeError: 2' in capsys.readouterr().out
    with FaceDatabase(config.database) as db:
        assert db.connection.execute("SELECT COUNT(*) FROM publications WHERE kind='link' AND status='unknown'").fetchone()[0] == 2


def test_group_and_example_limits_keep_exact_totals_and_sanitized_output(tmp_path, monkeypatch, capsys):
    config, _, services, _ = run_setup(tmp_path, monkeypatch)
    monkeypatch.setenv('STASH_API_KEY', 'synthetic-api-key')
    kinds = [type('Failure' + str(i), (RuntimeError,), {}) for i in range(10)]
    def source(reader, scene, config):
        number = int(scene['id']) - 1
        raise kinds[number // 3]('fetch http://operator:password@stash/input?custom=private-query#private-fragment '
                                 + 'synthetic-api-key ' + 'x' * 1000)
    services.source = source
    assert execute(arguments(*(str(i) for i in range(1, 31))), config, services=services) == 1
    _, summary = history(config)
    errors = summary['errors']
    assert errors['total'] == 30 and len(errors['groups']) == 6 and errors['overflow'] == 12
    assert all(group['count'] == 3 and len(group['examples']) == 2 for group in errors['groups'])
    assert all(len(example) <= 350 for group in errors['groups'] for example in group['examples'])
    text = capsys.readouterr().out + json.dumps(summary)
    assert 'Other recognition errors: 12' in text
    for secret in ('operator', 'password', 'private-query', 'private-fragment', 'synthetic-api-key'):
        assert secret not in text


def test_interruption_preserves_collected_failures_and_the_original_exception(tmp_path, monkeypatch, capsys):
    config, client, services, models = run_setup(tmp_path, monkeypatch)
    original = services.source
    def source(reader, scene, config):
        if scene['id'] == '1': raise ValueError('first scene unavailable')
        return original(reader, scene, config)
    services.source = source
    def interrupted(*args):
        raise KeyboardInterrupt('primary scan interruption')
    services.backend = interrupted
    with pytest.raises(KeyboardInterrupt, match='primary scan interruption'):
        execute(arguments('1', '2'), config, services=services)
    row, summary = history(config)
    assert row['status'] == 'interrupted' and 'primary scan interruption' in row['error']
    assert summary['errors']['total'] == 1 and 'first scene unavailable' in json.dumps(summary['errors'])
    assert 'Recognition errors scan/ValueError: 1' in capsys.readouterr().out and client.closed


@pytest.mark.parametrize('boundary', ['snapshot', 'render'])
def test_optional_diagnostic_failures_do_not_skip_run_history(tmp_path, monkeypatch, boundary):
    import stash_ai.recognition.run as runner
    config, _, services, _ = run_setup(tmp_path, monkeypatch)
    def source(*args): raise ValueError('source unavailable')
    def fail(*args): raise OSError('diagnostic failure')
    services.source = source
    monkeypatch.setattr(runner, 'error_snapshot' if boundary == 'snapshot' else 'render_errors', fail)
    assert execute(arguments('1'), config, services=services) == 1
    row, summary = history(config)
    assert row['status'] == 'partial' and row['finished'] is not None
    counter = 'recognition_error_snapshot_failures' if boundary == 'snapshot' else 'recognition_error_summary_failures'
    assert summary['mutations'][counter] == 1
    if boundary == 'snapshot':
        assert summary['errors']['unavailable']
    else:
        assert summary['errors']['total'] == 1


def test_secondary_render_interruption_cannot_replace_a_primary_scan_interruption(tmp_path, monkeypatch):
    import stash_ai.recognition.run as runner
    config, client, services, _ = run_setup(tmp_path, monkeypatch)
    services.backend = lambda *args: (_ for _ in ()).throw(KeyboardInterrupt('primary scan interruption'))
    def render(*args): raise KeyboardInterrupt('secondary rendering interruption')
    monkeypatch.setattr(runner, 'render_errors', render)
    with pytest.raises(KeyboardInterrupt, match='primary scan interruption'):
        execute(arguments('1'), config, services=services)
    row, summary = history(config)
    assert row['status'] == 'interrupted' and row['finished'] is not None and client.closed
    assert summary['mutations']['cleanup_failures'] == 1


@pytest.mark.parametrize('url,expected', [
    ('https://operator:password@stash:443/a?token=secret#fragment', 'https://stash:443/a'),
    ('http://%6fperator:pass%40word@stash/a', 'http://stash/a'),
    ('http://operator:password@[::1]:9999/a?default=true', 'http://[::1]:9999/a'),
    ('//operator:password@stash/a', '//stash/a'),
    ('http://operator:pass@word@stash/a', 'http://stash/a'),
    ('http://operator@stash/a', 'http://stash/a'),
    ('http://\u017c\u00f3\u0142\u0107:password@stash/a', 'http://stash/a'),
])
def test_source_url_sanitization_keeps_only_non_secret_identity(url, expected):
    assert safe_url(url) == expected


def test_invalid_url_errors_and_native_netloc_messages_do_not_expose_userinfo():
    with pytest.raises(ValueError, match='Invalid recognition source URL') as error:
        safe_url('http://operator:password@stash\uff0fevil')
    assert 'operator' not in str(error.value) and 'password' not in str(error.value)
    native = safe_error(ValueError("netloc 'operator:password@stash\uff0fevil' contains invalid characters"))
    assert 'operator' not in native and 'password' not in native
    assert message('read //operator:password@stash/a?custom=secret#fragment failed') == 'read //stash/a failed'


def test_rejected_video_source_and_fallback_records_have_no_credentials(tmp_path, monkeypatch, capsys):
    config, stash, services = video_setup(tmp_path, monkeypatch)
    original = stash.scene
    raw = 'http://operator:password@stash/scene/1/stream?custom=private-query#private-fragment'
    def scene(identifier):
        result = original(identifier)
        result['paths']['stream'] = raw
        return result
    stash.scene = scene
    class Rejected:
        def __init__(self, client, scene, config): pass
        def __enter__(self): raise ValueError('unusable stream ' + raw)
        def close(self): pass
    services.decoder = Rejected
    assert execute(arguments('1'), config, services=services) == 0
    with FaceDatabase(config.database) as db:
        values = [row[0] for row in db.connection.execute('SELECT metadata FROM portraits')]
        assert values and any(json.loads(value).get('source') == 'http://stash/scene/1/stream' for value in values)
        assert any(json.loads(value).get('source_type') == 'vtt' for value in values)
    text = '\n'.join(values) + capsys.readouterr().out
    assert all(secret not in text for secret in ('operator', 'password', 'private-query', 'private-fragment'))


def test_image_write_preconditions_strip_credentials_without_changing_the_live_read(tmp_path, monkeypatch):
    config, stash, services, _ = vtt_setup(tmp_path, monkeypatch)
    create = stash.create_person
    def person(name, token):
        result = create(name, token)
        stash.profiles[result['id']]['image_path'] = f"http://operator:password@stash/performer/{result['id']}/image?default=true&token=secret"
        return stash.performer(result['id'])
    stash.create_person = person
    assert execute(arguments('1'), config, services=services) == 0
    performer = stash.links['1'][-1]
    with FaceDatabase(config.database) as db:
        row = db.connection.execute("SELECT precondition FROM publications WHERE kind='image'").fetchone()
        value = json.loads(row[0])
        assert value['missing'] and value['image_path'] == f'http://stash/performer/{performer}/image'
        assert 'operator' not in row[0] and 'password' not in row[0]
    assert stash.images[performer]


def test_cached_video_source_is_sanitized_without_reloading_models_or_rewriting_image(tmp_path, monkeypatch):
    config, stash, services = video_setup(tmp_path, monkeypatch)
    assert execute(arguments('1'), config, services=services) == 0
    with FaceDatabase(config.database) as db:
        for row in db.connection.execute("SELECT id,metadata FROM portraits WHERE json_extract(metadata,'$.source_type')='video'").fetchall():
            data = json.loads(row['metadata'])
            data['source'] = 'http://operator:password@stash/scene/1/stream?token=secret#fragment'
            db.connection.execute('UPDATE portraits SET metadata=? WHERE id=?', (json.dumps(data), row['id']))
        db.connection.commit()
    writes = list(stash.writes)
    services.backend = services.decoder = lambda *args: pytest.fail('sanitization must reuse validated evidence')
    assert execute(arguments('1'), config, services=services) == 0
    assert stash.writes == writes
    with FaceDatabase(config.database) as db:
        values = [row[0] for row in db.connection.execute("SELECT metadata FROM portraits WHERE json_extract(metadata,'$.source_type')='video'")]
        assert all(json.loads(value)['source'] == 'http://stash/scene/1/stream' for value in values)


def test_retry_predecessor_error_is_sanitized_without_changing_its_recovery_identity():
    previous = {'id': 'intent', 'status': 'unknown', 'payload': '{"portrait":"cached","sha256":"abc"}',
                'precondition': '{"image_sha256":"old"}',
                'error': 'http://operator:password@stash/a?secret=query#fragment synthetic-key'}
    result = retry_precondition(previous, api_key='synthetic-key')['retry_of']
    assert result['id'] == 'intent' and result['payload'] == {'portrait': 'cached', 'sha256': 'abc'}
    assert result['image_sha256'] == 'old' and result['error'] == 'http://stash/a [redacted]'


import json
import pytest

@pytest.mark.parametrize('stage', ['rebuild', 'names', 'portraits'])
def test_non_scene_errors_are_retained_in_history(tmp_path, monkeypatch, stage):
    import stash_ai.recognition.run as runner
    from stash_ai.recognition.database import FaceDatabase
    from tests.test_recognition_run import setup, run
    config, client, services, _ = setup(tmp_path, monkeypatch)
    reason = 'synthetic ' + stage + ' failure'
    def fail(*a, **kw):
        raise RuntimeError(reason)
    if stage == 'rebuild':
        monkeypatch.setattr(runner, 'rebuild_selected', fail)
    elif stage == 'names':
        services.metadata_plan = fail
    else:
        import stash_ai.recognition.metadata as metadata
        monkeypatch.setattr(metadata, 'fill_portrait', fail)
    assert run(config, services, '1') == 1
    with FaceDatabase(config.database) as db:
        row = db.connection.execute('SELECT * FROM runs').fetchone()
        assert row['status'] == 'partial' and row['error'] is None
        assert reason in row['summary']
        assert json.loads(row['summary'])['errors']['total'] == 1
        assert reason not in ''.join(r[0] for r in db.connection.execute('SELECT data FROM progress'))

