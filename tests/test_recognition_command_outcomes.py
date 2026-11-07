import json

import pytest

from stash_ai.recognition.cli import main
from stash_ai.recognition.database import FaceDatabase
from stash_ai.recognition.merge_solver_checkpoint import Journal
from stash_ai.recognition.publication_probe import Probe
from stash_ai.recognition.run import execute
from tests.test_recognition_merge_trial_progress import arguments, auxiliary_goals, scripted_search
from tests.test_recognition_progress import args
from tests.test_recognition_run import setup


def test_interrupt_after_durable_winner_uses_result(tmp_path, monkeypatch, capsys):
    config, goals = auxiliary_goals(tmp_path, sets=2)
    scripted_search(monkeypatch)
    record = Journal.record

    def interrupt(journal, name, *values):
        record(journal, name, *values)
        if name == 'trial-1':
            raise KeyboardInterrupt

    monkeypatch.setattr(Journal, 'record', interrupt)
    options = arguments(config, goals, tmp_path, 'compact')
    assert main(options + ['--checkpoint', str(tmp_path/'journal.json')]) == 130
    output = capsys.readouterr()
    result = json.loads(output.out)
    assert result['selected_trial'] == 'trial-1'
    line = next(line for line in output.err.splitlines() if line.startswith('Merge solver best completed'))
    assert 'trial-1' in line and 'sets 2/2' in line and 'Solver 2/3' in line and 'best baseline' not in line
    assert 'Phase: Solve merge goals interrupted' in output.err


def test_publication_probe_failure_keeps_rebuild_outcome(tmp_path, monkeypatch, capsys):
    config, _, services, _ = setup(tmp_path, monkeypatch)
    def fail(*args):
        raise OSError('injected probe failure')
    monkeypatch.setattr(Probe, 'prepare', fail)
    assert execute(args('1', '2'), config, services=services) == 1
    text = capsys.readouterr().out
    assert 'ERROR recognition publication:' in text and 'ERROR recognition rebuild:' not in text
    with FaceDatabase(config.database) as db:
        counts = json.loads(db.connection.execute('SELECT summary FROM runs').fetchone()[0])['counts']
    assert counts['publication_not_attempted'] == 2
    assert counts.get('scenes_failed', 0) == 0


def test_pose_interrupt_names_scene_and_sparse_gap(tmp_path, monkeypatch, capsys):
    from stash_ai.recognition.person_collection import PersonCollector
    from tests.test_recognition_gpu_memory import configured

    config, _, services, _ = setup(tmp_path, monkeypatch)
    config = configured(config, 'sequential')
    config.data['sapiensid']['enabled'] = True
    config.data['execution']['sequential_scene_batch_size'] = 4
    def stop(*args):
        raise KeyboardInterrupt
    monkeypatch.setattr(PersonCollector, 'enrich', stop)
    with pytest.raises(KeyboardInterrupt):
        execute(args('1', '2'), config, services=services)
    output = capsys.readouterr().out
    assert 'resume scene=1 pass=pose' in output and 'first_missing=0' in output
