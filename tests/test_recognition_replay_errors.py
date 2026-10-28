"""Replay failures retain their cause and identify unthrottled stage locations."""
import json

import pytest

from stash_ai.recognition import replay_simulation
from stash_ai.recognition.progress import stage
from tests.test_recognition_merge_report import seeded


@pytest.mark.parametrize('with_progress', [False, True])
def test_replay_error_identifies_stage_even_when_not_printed(tmp_path, monkeypatch, with_progress):
    config, _ = seeded(tmp_path)
    before = config.database.read_bytes()
    failure = ValueError('The truth value of an array with more than one element is ambiguous.')
    events = []

    def fail(*args):
        with stage('Confirm cross-scene provisional identities'):
            pass
        with stage('Build early verified appearance views'):
            raise failure

    monkeypatch.setattr(replay_simulation, 'rebuild_selected', fail)
    monkeypatch.setattr(replay_simulation.time, 'monotonic', lambda: 100.)
    with pytest.raises(ValueError) as caught:
        replay_simulation.replay(config, {}, progress=(
            lambda name, state, item: events.append(state)) if with_progress else None)
    assert caught.value is failure
    note = failure.__notes__[0]
    assert 'Recognition replay baseline; stage: Build early verified appearance views;' in note
    assert 'test_recognition_replay_errors.py:' in note and '(fail)' in note
    assert 'replay_simulation.py:' in note
    assert 'private-crop' not in note and 'user:secret' not in note
    assert str(tmp_path) not in note
    if with_progress:
        assert 'stage: Confirm cross-scene provisional identities' in events
        assert 'stage: Build early verified appearance views' not in events
        assert 'complete' not in events
    assert config.database.read_bytes() == before


@pytest.mark.parametrize('with_note', [False, True])
def test_cli_prints_exception_notes_with_existing_credential_redaction(monkeypatch, capsys, with_note):
    from stash_ai import cli
    failure = ValueError('synthetic failure')
    if with_note:
        failure.add_note('Recognition replay trial-34; stage: Test; api-key-value '
                         'https://user:password@stash/private?token=token-value')

    def fail(**kwargs):
        raise failure

    monkeypatch.setattr(cli, 'main', fail)
    monkeypatch.setattr(cli.sys, 'argv', ['stash_ai', 'recognition'])
    monkeypatch.setenv('STASH_API_KEY', 'api-key-value')
    assert cli.run() == 1
    output = capsys.readouterr()
    assert not output.out
    assert output.err.startswith('ERROR: synthetic failure\n')
    if with_note:
        assert 'Recognition replay trial-34; stage: Test;' in output.err
        assert '[redacted]' in output.err
        assert all(value not in output.err for value in ('api-key-value', 'user:password', 'token-value'))
    else:
        assert output.err == 'ERROR: synthetic failure\n'


def test_failed_trial_preserves_completed_winner_and_can_resume(tmp_path, monkeypatch):
    from benchmarks.benchmark_merge_solver import fixture
    from stash_ai.recognition.fragmentation_report import _open_readonly
    from stash_ai.recognition.merge_solver import solve
    from stash_ai.recognition.merge_solver_checkpoint import winner_file
    config, goals = fixture(tmp_path/'input', 'query')
    checkpoint, yaml = tmp_path/'solver.json', tmp_path/'winner.yml'
    original = replay_simulation.rebuild_selected
    saved = {}

    def progress(name, state, item):
        if name == 'trial-2' and state == 'start':
            saved.update(journal=checkpoint.read_bytes(), yaml=yaml.read_bytes())

    def fail_after_completed_trials(*args):
        if saved:
            with stage('Synthetic rebuild failure'):
                raise ValueError('synthetic array error')
        return original(*args)

    monkeypatch.setattr(replay_simulation, 'rebuild_selected', fail_after_completed_trials)
    before = config.database.read_bytes()
    with _open_readonly(config.database) as db, pytest.raises(ValueError) as caught:
        solve(config, goals, db, maximum=4, checkpoint=checkpoint,
              winner_writer=winner_file(yaml), progress=progress)
    assert 'Recognition replay trial-2; stage: Synthetic rebuild failure;' in caught.value.__notes__[0]
    assert yaml.read_bytes() == saved['yaml']
    assert checkpoint.read_bytes() == saved['journal']
    assert len(json.loads(checkpoint.read_text())['records']) == 2
    assert config.database.read_bytes() == before
    monkeypatch.setattr(replay_simulation, 'rebuild_selected', original)
    with _open_readonly(config.database) as db:
        resumed = solve(config, goals, db, maximum=4, checkpoint=checkpoint, resume=True,
                        winner_writer=winner_file(yaml, resume=True))
    assert resumed['search']['restored_trials'] == 2
    assert 'trial-2' in resumed['simulation']['trials']
