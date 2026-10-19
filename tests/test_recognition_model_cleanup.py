"""Cross-namespace link isolation and explicit orphan performer deletion."""
from dataclasses import replace
import json
import uuid
from types import SimpleNamespace

import pytest

from stash_ai.recognition.cleanup_orphans import execute as cleanup, tokens
from stash_ai.recognition.config import plain
from stash_ai.recognition.database import FaceDatabase
from stash_ai.recognition.run import instance_scope
from stash_ai.recognition.stash import RecognitionStash
from tests.test_recognition_run import setup, run


def another_model(config):
    changed = replace(config, data=plain(config.data))
    changed.data['models']['recognizer'] = (
        'auraface' if config.data['models']['recognizer'] == 'sface' else 'sface')
    return changed


def cleanup_args(*, apply=False):
    return SimpleNamespace(apply=apply, progress_style='off')


def inventory(client):
    # The real client filters by the substring on the Stash side and then
    # checks for an exact application-issued token on the caller side.
    client.recognition_performers = lambda: (
        dict(p) for p in list(client.profiles.values())
        if 'stash-ai-recognition:' in (p.get('details') or ''))
    def delete(performer):
        client.writes.append(('delete', performer))
        client.profiles.pop(performer)
        for scene in client.links:
            client.links[scene] = [p for p in client.links[scene] if p != performer]
        return True
    client.delete_person = delete


def test_model_switch_detaches_other_model_only_for_selected_scene(tmp_path, monkeypatch):
    config, client, services, _ = setup(tmp_path, monkeypatch)
    assert run(config, services, '1', '2') == 0
    old = client.links['2'][0]
    assert client.links['1'] == ['77', old]
    changed = another_model(config)
    assert run(changed, services, '1') == 0
    assert client.links['1'][0] == '77'
    assert old not in client.links['1']
    assert client.links['2'] == [old]  # outside selection
    assert old in client.profiles  # never delete when changing models
    newcomer = next(p for p in client.links['1'] if p != '77')
    assert newcomer != old and newcomer in client.profiles
    writes = list(client.writes)
    assert run(changed, services, '1') == 0
    assert client.writes == writes  # no recurring old-model unlink
    with FaceDatabase(config.database) as db:
        history = json.loads(db.connection.execute(
            'SELECT summary FROM runs ORDER BY started DESC LIMIT 1').fetchone()[0])
        assert history['counts'].get('other_model_links_removed', 0) == 0


def test_switch_back_relinks_previous_model_without_erasing_new_one(tmp_path, monkeypatch):
    config, client, services, _ = setup(tmp_path, monkeypatch)
    assert run(config, services, '1') == 0
    old = client.links['1'][-1]
    changed = another_model(config)
    assert run(changed, services, '1') == 0
    newer = client.links['1'][-1]
    assert old != newer and old not in client.links['1']
    assert run(config, services, '1') == 0
    assert old in client.links['1'] and newer not in client.links['1']
    assert old in client.profiles and newer in client.profiles


def test_model_switch_detaches_other_model_even_without_prior_link_journal(tmp_path, monkeypatch):
    config, client, services, _ = setup(tmp_path, monkeypatch)
    assert run(config, services, '2') == 0
    old = client.links['2'][0]
    client.links['1'].append(old)  # link added outside recognition; no journal row
    assert run(another_model(config), services, '1') == 0
    assert old not in client.links['1'] and old in client.links['2']
    assert old in client.profiles and '77' in client.links['1']


def test_other_model_unknown_ownership_does_not_unlink(tmp_path, monkeypatch):
    config, client, services, _ = setup(tmp_path, monkeypatch)
    assert run(config, services, '1') == 0
    old = client.links['1'][-1]
    client.profiles[old]['details'] = 'someone changed this profile'
    assert run(another_model(config), services, '1') == 1
    assert old in client.links['1'] and old in client.profiles
    assert not any(write == ('unlink', '1', old) for write in client.writes)


def test_orphan_cleanup_is_explicit_protects_known_and_manual_profiles(tmp_path, monkeypatch):
    config, client, services, _ = setup(tmp_path, monkeypatch)
    assert run(config, services, '1') == 0
    with FaceDatabase(config.database) as db:
        local = db.connection.execute('SELECT token,performer FROM people').fetchone()
        orphan = f'stash-ai-recognition:{db.installation}:{uuid.uuid4()}'
    known = local['performer']
    client.profiles['888'] = {'id': '888', 'name': 'Renamed orphan', 'details': orphan, 'image_path': ''}
    client.profiles['889'] = {'id': '889', 'name': 'Manual notes',
                              'details': 'stale text stash-ai-recognition: not a token', 'image_path': ''}
    client.profiles['890'] = {'id': '890', 'name': 'Mixed ownership',
                              'details': orphan + '\n' + local['token'], 'image_path': ''}
    inventory(client)
    assert tokens(client.profiles['888']) == {orphan}
    assert cleanup(cleanup_args(), config, services=services) == 0
    assert not any(item[0] == 'delete' for item in client.writes)
    assert cleanup(cleanup_args(apply=True), config, services=services) == 0
    assert '888' not in client.profiles
    assert all(identifier in client.profiles for identifier in (known, '77', '889', '890'))
    assert cleanup(cleanup_args(apply=True), config, services=services) == 0
    assert client.writes.count(('delete', '888')) == 1
    with FaceDatabase(config.database) as db:
        rows = db.connection.execute("SELECT * FROM publications WHERE kind='delete'").fetchall()
        assert len(rows) == 1 and rows[0]['status'] == 'confirmed'
        history = db.connection.execute('SELECT config,summary FROM runs ORDER BY started DESC').fetchall()
        assert json.loads(history[0]['config'])['invocation']['command'] == 'cleanup'
        assert json.loads(history[0]['config'])['invocation']['apply'] is True


def test_cleanup_refuses_missing_database_even_with_apply(tmp_path, monkeypatch):
    config, client, services, _ = setup(tmp_path, monkeypatch)
    inventory(client)
    with pytest.raises(FileNotFoundError, match='existing face database'):
        cleanup(cleanup_args(apply=True), config, services=services)
    assert not client.writes


def test_unknown_deletion_not_blindly_retried(tmp_path, monkeypatch):
    config, client, services, _ = setup(tmp_path, monkeypatch)
    assert run(config, services, '1') == 0
    with FaceDatabase(config.database) as db:
        orphan = f'stash-ai-recognition:{db.installation}:{uuid.uuid4()}'
    client.profiles['888'] = {'id': '888', 'name': 'Orphan', 'details': orphan, 'image_path': ''}
    inventory(client)
    def ambiguous(performer):
        client.writes.append(('delete_attempt', performer))
        raise TimeoutError('uncertain network acknowledgement')
    client.delete_person = ambiguous
    assert cleanup(cleanup_args(apply=True), config, services=services) == 1
    assert cleanup(cleanup_args(apply=True), config, services=services) == 1
    assert client.writes.count(('delete_attempt', '888')) == 1


def test_recognition_inventory_and_deletion_use_supported_graphql_shapes():
    client = RecognitionStash.__new__(RecognitionStash)
    calls = []
    def gql(query, variables):
        calls.append((query, variables))
        if 'RecognitionOwnedProfiles' in query:
            assert variables['needle'] == 'stash-ai-recognition:'
            assert variables['filter']['page'] == 1
            return {'findPerformers': {'performers': [{'id': '42', 'name': 'AI Person',
                                                       'details': 'note', 'image_path': ''}]}}
        assert 'performerDestroy(input:$input)' in query
        assert variables == {'input': {'id': '42'}}
        return {'performerDestroy': True}
    client.gql = gql
    assert [p['id'] for p in client.recognition_performers()] == ['42']
    assert client.delete_person('42') is True
    assert len(calls) == 2
