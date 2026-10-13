import json
from types import SimpleNamespace

import pytest

from stash_ai.recognition.cli import parser
from stash_ai.recognition.config import load
from stash_ai.recognition.database import FaceDatabase
from stash_ai.recognition.run import Services, execute, instance_scope
from stash_ai.recognition.stash import RecognitionStash
from tests.helpers.recognition import Backend, Stash


def setup(tmp_path, monkeypatch, **settings):
    monkeypatch.setenv('STASH_URL', 'http://stash/')
    path = tmp_path / 'facial_recognition.yml'
    path.write_text('version: 1\ndetection: {min_sharpness: 0}\nportraits: {enabled: false}\n')
    config = load(path)
    client = Stash()
    backends = []
    def backend(*args):
        value = Backend(**settings)
        backends.append(value)
        return value
    services = Services(client=lambda **kwargs: client, assets=lambda *a, **k: {}, backend=backend)
    return config, client, services, backends


def run(config, services, *ids):
    args = SimpleNamespace(scene_id=list(ids), all_scenes=False, progress_style='off')
    return execute(args, config, services=services)


def test_selected_scene_repeat_reuses_vectors_preserves_manual_links_and_has_one_history(tmp_path, monkeypatch):
    config, client, services, backends = setup(tmp_path, monkeypatch)
    assert run(config, services, '1') == 0
    assert client.links['1'][0] == '77'
    assert len(client.links['1']) == 2 and client.links['2'] == []
    assert len(backends) == 1 and backends[0].closed
    writes = list(client.writes)
    assert run(config, services, '1') == 0
    assert client.writes == writes and len(backends) == 1
    assert all('/1.' in url for url in client.media_calls)
    with FaceDatabase(config.database) as state:
        rows = state.connection.execute('SELECT * FROM runs').fetchall()
        assert len(rows) == 2 and all(row['status'] == 'success' for row in rows)
        assert json.loads(rows[0]['selection']) == ['1']
        assert json.loads(rows[0]['summary'])['mutations']['mutation_confirmed'] == 2
        assert state.connection.execute('SELECT COUNT(*) FROM people').fetchone()[0] == 1
        assert state.connection.execute('SELECT COUNT(*) FROM observations').fetchone()[0] == 3
    assert not (tmp_path / 'stash_ai.sqlite3').exists()


@pytest.mark.parametrize('kind', ['create', 'link'])
def test_lost_response_reconciles_without_duplicate_write(tmp_path, monkeypatch, kind):
    config, client, services, _ = setup(tmp_path, monkeypatch)
    setattr(client, 'fail_' + kind, True)
    assert run(config, services, '1') == 1
    writes = list(client.writes)
    assert run(config, services, '1') == 0
    assert sum(write[0] == kind for write in client.writes) == 1
    assert len(client.links['1']) == 2
    if kind == 'link':
        assert client.writes == writes


def test_invisible_uncertain_create_not_retried_even_for_other_selected_scene(tmp_path, monkeypatch):
    config, client, services, _ = setup(tmp_path, monkeypatch)
    client.fail_create, client.create_visible = True, False
    assert run(config, services, '1') == 1
    assert run(config, services, '2') == 1
    assert sum(write[0] == 'create' for write in client.writes) == 1
    assert client.links == {'1': ['77'], '2': []}


def test_selected_scene_does_not_resume_other_pending_link(tmp_path, monkeypatch):
    config, client, services, _ = setup(tmp_path, monkeypatch)
    client.fail_link = True
    assert run(config, services, '1') == 1
    assert run(config, services, '2') == 0
    assert sum(write[:2] == ('link', '1') for write in client.writes) == 1
    assert client.links['2'] == [client.links['1'][-1]]
    with FaceDatabase(config.database) as state:
        assert state.connection.execute("SELECT status FROM publications WHERE scene='1'").fetchone()[0] == 'unknown'


def test_changed_ownership_never_rebinds_or_duplicates(tmp_path, monkeypatch):
    config, client, services, _ = setup(tmp_path, monkeypatch)
    assert run(config, services, '1') == 0
    client.profiles[client.links['1'][-1]]['details'] = 'edited manually'
    writes = list(client.writes)
    assert run(config, services, '2') == 1
    assert client.writes == writes


@pytest.mark.parametrize('argv', [[], ['scan'], ['run'], ['run', '--scene-id', '1', '--all-scenes'],
    ['run', '--scene-id', '-1'], ['run', '--scene-id', '1', '--models', 'all'],
    ['run', '--scene-id', '1', '--dry-run'], ['review']])
def test_scope_and_excluded_commands_fail_before_resources(argv):
    with pytest.raises(SystemExit):
        parser().parse_args(argv)


def test_root_dispatch_avoids_tag_runtime_and_normalizes_ids(monkeypatch):
    import stash_ai.cli as cli
    import stash_ai.recognition.cli as recognition
    monkeypatch.setattr(cli, 'RuntimeContext', lambda: pytest.fail('tag runtime must stay dormant'))
    seen = []
    monkeypatch.setattr(recognition, 'main', lambda args: seen.append(args) or 0)
    assert cli.main(['recognition', 'run', '--scene-id', '6306']) == 0
    assert seen == [['run', '--scene-id', '6306']]
    assert parser().parse_args(['run', '--scene-id', '0001']).scene_id == ['1']


def test_graphql_link_uses_add_and_rejects_wrong_acknowledgement():
    client = RecognitionStash.__new__(RecognitionStash)
    def gql(query, variables):
        assert variables == {'input': {'ids': ['1'], 'performer_ids': {'ids': ['5'], 'mode': 'ADD'}}}
        return {'bulkSceneUpdate': [{'id': '2', 'performers': [{'id': '5'}]}]}
    client.gql = gql
    with pytest.raises(RuntimeError, match='ID'):
        client.add_scene_person('1', '5')


def test_scope_rejects_credentials_and_differs_between_instances():
    assert instance_scope('http://stash/') == instance_scope('http://STASH:80')
    assert instance_scope('http://stash/a') != instance_scope('http://stash/b')
    with pytest.raises(ValueError):
        instance_scope('https://user:password@stash')
