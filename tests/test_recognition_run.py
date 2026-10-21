import json
from dataclasses import replace
from types import SimpleNamespace

import pytest

from stash_ai.recognition.cli import parser
from stash_ai.recognition.config import load, plain
from stash_ai.recognition.database import FaceDatabase
from stash_ai.recognition.run import Services, execute, instance_scope
from stash_ai.recognition.stash import RecognitionStash
from tests.helpers.recognition import Backend, Stash


def setup(tmp_path, monkeypatch, **settings):
    monkeypatch.setenv('STASH_URL', 'http://stash/')
    path = tmp_path / 'facial_recognition.yml'
    path.write_text('version: 1\ndetection: {min_sharpness: 0, max_pose_deviation: null, min_visible_face_fraction: 0, require_complete_face: false}\nportraits: {enabled: false}\n')
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


@pytest.mark.parametrize('argv', [['run', '--scene-id', '6306'], ['portraits', '--performer-id', '123']])
def test_root_dispatch_avoids_tag_runtime_and_normalizes_ids(monkeypatch, argv):
    import stash_ai.cli as cli
    import stash_ai.recognition.cli as recognition
    monkeypatch.setattr(cli, 'RuntimeContext', lambda: pytest.fail('tag runtime must stay dormant'))
    seen = []
    monkeypatch.setattr(recognition, 'main', lambda args: seen.append(args) or 0)
    assert cli.main(['recognition', *argv]) == 0
    assert seen == [argv]
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


def test_reference_metrics_reach_console_history_and_cached_run(tmp_path, monkeypatch, capsys):
    import stash_ai.recognition.run as runner
    config, _, services, _ = setup(tmp_path, monkeypatch)
    statuses = []
    monkeypatch.setattr(runner, 'status', lambda report, **values: statuses.append(values))
    assert run(config, services, '1') == 0
    first_output = capsys.readouterr().out
    assert 'Face references scene 1:' not in first_output
    assert 'samples=3 stage=retained' in first_output
    assert 'Recognition success:' in first_output and '| face references=' in first_output
    assert any('MiB (' in v.get('references', '') and v['reference_samples'] == 3 for v in statuses)
    assert run(config, services, '1') == 0
    assert 'stage=cached' in capsys.readouterr().out
    with FaceDatabase(config.database) as state:
        histories = state.connection.execute('SELECT * FROM runs ORDER BY started').fetchall()
        for row in histories:
            summary = json.loads(row['summary'])['references']
            progress = json.loads(state.connection.execute(
                "SELECT data FROM progress WHERE run=? AND phase='references'", (row['id'],)).fetchone()[0])
            assert summary == progress
            assert summary['bytes'] == summary['peak_bytes'] == 48
            assert summary['budget_bytes'] == 134217728
            assert summary['people'] == 1 and summary['samples'] == 3
            assert summary['recognizer'] == 'sface' and summary['max_samples_per_person'] == 32
        assert json.loads(histories[0]['summary'])['references']['stage'] == 'retained'
        assert json.loads(histories[1]['summary'])['references']['stage'] == 'cached'


def test_reference_run_peak_is_maximum_across_scenes_not_sum(tmp_path, monkeypatch):
    config, _, services, _ = setup(tmp_path, monkeypatch)
    assert run(config, services, '1', '2') == 0
    with FaceDatabase(config.database) as state:
        summary = json.loads(state.connection.execute('SELECT summary FROM runs').fetchone()[0])['references']
        assert summary['scene'] == '2'
        assert summary['bytes'] == summary['peak_bytes'] == 96
        assert summary['people'] == 1 and summary['samples'] == 6


def test_reference_history_write_cannot_commit_failed_matching(tmp_path, monkeypatch, capsys):
    config, client, services, _ = setup(tmp_path, monkeypatch)
    assert run(config, services, '1') == 0
    writes = list(client.writes)
    config = replace(config, data=plain(config.data))
    config.data['clustering']['max_reference_bytes'] = 64
    assert run(config, services, '2') == 1
    output = capsys.readouterr().out
    assert 'required=96 bytes, limit=64 bytes' in output
    assert 'samples=6 stage=selected_and_historical' in output
    assert client.writes == writes
    with FaceDatabase(config.database) as state:
        summary = json.loads(state.connection.execute('SELECT summary FROM runs ORDER BY started DESC').fetchone()[0])
        assert summary['references']['bytes'] == summary['references']['peak_bytes'] == 96
        assert summary['counts']['scenes_failed'] == 1
        assert state.connection.execute('SELECT COUNT(*) FROM members').fetchone()[0] == 3
        assert state.connection.execute('SELECT COUNT(*) FROM candidates').fetchone()[0] == 3
        assert state.connection.execute("SELECT COUNT(*) FROM candidates c JOIN scans s ON s.id=c.scan WHERE s.scene='2'").fetchone()[0] == 0


def test_later_scene_merges_earlier_metadata_target_and_preserves_oldest_profile(tmp_path, monkeypatch, capsys):
    import math
    import numpy as np
    config, client, services, _ = setup(tmp_path, monkeypatch)
    strict = replace(config, data=plain(config.data))
    strict.data['recognition'].update(match_threshold=.9, ambiguity_margin=.01)
    assert run(strict, services, '1') == 0
    oldest = client.links['1'][-1]
    values = iter([np.array([x, math.sqrt(1-x*x), 0., 0.]) for x in [.61, .61, .88]])
    def varied(*args):
        backend = Backend()
        backend.embed = lambda image, face: next(values)
        return backend
    services.backend = varied
    assert run(strict, services, '2') == 0
    youngest = client.links['2'][0]
    assert oldest != youngest
    targets = []
    services.metadata = lambda database, client, report, config, plan, scope, target, *args, **kwargs: targets.append(target) or {}
    services.backend = lambda *args: Backend(vector=np.array([1.8, .6, 0., 0.]) / math.sqrt(3.6))
    assert run(config, services, '1', '2', '10') == 0
    assert youngest in client.profiles and client.links['2'] == [oldest]
    assert len(targets) == 1 and targets[0]['performer'] == oldest
    assert [s['scene'] for s in targets[0]['scans']] == ['1', '2', '10']
    assert 'scene_links_removed=1' in capsys.readouterr().out
    writes = list(client.writes)
    services.backend = lambda *args: pytest.fail('merge rerun must reuse retained embeddings')
    assert run(config, services, '1', '2', '10') == 0
    assert client.writes == writes
    with FaceDatabase(config.database) as db:
        assert db.connection.execute('SELECT COUNT(*) FROM identity_merges').fetchone()[0] == 0


def test_cold_replay_withholds_near_rival_and_updates_only_selected_links(tmp_path, monkeypatch):
    import math
    config, client, services, _ = setup(tmp_path, monkeypatch)
    strict = replace(config, data=plain(config.data))
    strict.data['recognition'].update(match_threshold=.98, ambiguity_margin=.01)
    performers = []
    for scene, vector in [('1', [1.,0,0,0]), ('2', [.8,.6,0,0]), ('3', [.35,math.sqrt(1-.35**2),0,0])]:
        services.backend = lambda *args, vector=vector: Backend(vector=vector)
        assert run(strict, services, scene) == 0
        performers.append(client.links[scene][-1])
    services.backend = lambda *args: Backend(vector=[.8,.6,0,0])
    assert run(config, services, '1', '2', '3', '10') == 0
    assert performers[1] in client.profiles
    assert performers[2] in client.profiles
    assert client.links['10'] == [performers[0]]
    assert client.links['3'] == []
    assert client.links['2'] == [performers[0]]
    with FaceDatabase(config.database) as db:
        summary = json.loads(db.connection.execute('SELECT summary FROM runs ORDER BY started DESC LIMIT 1').fetchone()[0])
        assert summary['counts']['scene_links_removed'] == 2
        assert summary['counts']['observations_withheld'] == 3


def test_rebuild_does_not_dispatch_destructive_performer_merge(tmp_path, monkeypatch):
    config, client, services, _ = setup(tmp_path, monkeypatch)
    strict = replace(config, data=plain(config.data))
    strict.data['recognition'].update(match_threshold=.98, ambiguity_margin=.01)
    assert run(strict, services, '1') == 0
    services.backend = lambda *args: Backend(vector=[.8,.6,0,0])
    assert run(strict, services, '2') == 0
    client.merge_person = lambda *args: pytest.fail('rebuild must preserve profiles for future splitting')
    assert run(config, services, '1', '2', '10') == 0
    assert sum(w[0] == 'merge' for w in client.writes) == 0
    assert client.links['2'] == [client.links['1'][-1]]


def test_local_merge_publishes_independently_of_other_model_namespace(tmp_path, monkeypatch):
    import math
    config, client, services, _ = setup(tmp_path, monkeypatch)
    assert run(config, services, '99') == 0
    old = client.links['99'][-1]
    other = replace(config, data=plain(config.data))
    other.data['models']['recognizer'] = 'auraface'
    strict = replace(other, data=plain(other.data))
    strict.data['recognition'].update(match_threshold=.98, ambiguity_margin=.01)
    assert run(strict, services, '1') == 0
    services.backend = lambda *args: Backend(vector=[.8,.6,0,0])
    assert run(strict, services, '2') == 0
    services.backend = lambda *args: Backend(vector=[1.8 / math.sqrt(3.6), .6 / math.sqrt(3.6), 0, 0])
    assert run(other, services, '1', '2', '10') == 0
    assert client.links['99'] == [old] and old in client.profiles
    assert len(client.links['1']) > 1 and client.links['1'][0] == '77'
    assert client.links['2'] and client.links['10']
    assert all(p != old for scene in ('1', '2', '10') for p in client.links[scene])
    with FaceDatabase(config.database) as db:
        assert db.connection.execute('SELECT COUNT(*) FROM identity_merges').fetchone()[0] == 0
        assert not db.connection.execute(
            "SELECT 1 FROM publications WHERE kind='namespace' AND status='withheld'").fetchone()


import json
import sqlite3
import uuid
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest


def test_model_switch_unlinks_old_namespace_and_reports_count(tmp_path,monkeypatch,capsys):
    from tests.test_recognition_run import setup,run
    from tests.test_recognition_rebuild import changed
    from stash_ai.recognition.database import FaceDatabase
    config,client,services,_ = setup(tmp_path,monkeypatch)
    assert run(config,services,'1') == 0
    old = client.links['1'][-1]
    capsys.readouterr()
    alternate = changed(config,models={'recognizer':'auraface'})
    assert run(alternate,services,'1') == 0
    output = capsys.readouterr().out
    assert 'Recognition success' in output and 'other_model_links_removed=1' in output
    assert old not in client.links['1'] and old in client.profiles
    with FaceDatabase(config.database) as db:
        assert not db.connection.execute(
            "SELECT 1 FROM publications WHERE kind='namespace' AND status='withheld'").fetchone()
        summary = json.loads(db.connection.execute('SELECT summary FROM runs ORDER BY started DESC LIMIT 1').fetchone()[0])
        assert summary['counts']['other_model_links_removed'] == 1
        assert summary['counts']['groups_withheld'] == 0

