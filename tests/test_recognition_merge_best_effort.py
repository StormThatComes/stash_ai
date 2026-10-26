import json
import uuid

import pytest

from tests.test_recognition_merge_report import seeded
from stash_ai.recognition.database import FaceDatabase
from stash_ai.recognition.fragmentation_report import _open_readonly
from stash_ai.recognition.merge_goals import assess, objective, resolve, evaluate, unmet_conditions
from stash_ai.recognition.merge_solver import solve
from stash_ai.recognition.merge_goal_output import export, summary


def goal_sets(*groups):
    return dict(groups=[dict(index=i, members=[dict(identity_id=f'{i}-{j}', performer_id=str(j),
        observations=[dict(id=k) for k in ids]) for j, ids in enumerate(group)]) for i, group in enumerate(groups)])


def incomplete_source(tmp_path):
    """One historical member is absent from every possible latest-scan replay."""
    config, people = seeded(tmp_path)
    with FaceDatabase(config.database) as state:
        db = state.connection
        original = db.execute('SELECT o.* FROM observations o JOIN members m ON m.observation=o.id '
                              'WHERE m.person=? LIMIT 1', (people[0],)).fetchone()
        scan = list(db.execute('SELECT * FROM scans WHERE id=?', (original['scan'],)).fetchone())
        scan[0], scan[2], scan[-3], scan[-2] = str(uuid.uuid4()), '99', 1, 1
        db.execute('INSERT INTO scans VALUES (?,?,?,?,?,?,?,?,?,?,?)', scan)
        observation = list(original)
        observation[0], observation[1] = str(uuid.uuid4()), scan[0]
        db.execute('INSERT INTO observations VALUES (?,?,?,?,?,?,?,?,?,?,?,?)', observation)
        db.execute('INSERT INTO members VALUES (?,?)', (observation[0], people[0]))
        scan[0], scan[4], scan[-3], scan[-2] = str(uuid.uuid4()), 'new-input', 0, 0
        db.execute('INSERT INTO scans VALUES (?,?,?,?,?,?,?,?,?,?,?)', scan)
        db.commit()
    return config, people


def test_zero_complete_counts_still_distinguish_coverage_and_consolidation():
    goals = goal_sets([list('abcd')])
    observations = dict.fromkeys('abcd', '1')
    lost = assess(goals, observations, {}, dict.fromkeys('abcd'))
    partial = assess(goals, observations, {}, dict(a='x', b='y', c=None, d=None))
    consolidated = assess(goals, observations, {}, dict(a='x', b='x', c=None, d=None))
    more = assess(goals, observations, {}, dict(a='x', b='x', c='x', d=None))
    assert all(s['groups'][0]['largest_complete_performers'] == 0 for s in (lost, partial, consolidated, more))
    assert objective(more) > objective(consolidated) > objective(partial) > objective(lost)
    assert more['groups'][0]['assigned_fraction'] == .75
    assert more['groups'][0]['consolidation_fraction'] == .75
    # Discarding an awkward fragment cannot beat assigning every observation.
    split = assess(goals, observations, {}, dict(a='x', b='x', c='x', d='y'))
    assert objective(split) > objective(more)


def test_set_separation_and_fair_coverage_precede_fallback_merges():
    goals = goal_sets([list('ab')], [list('cd')])
    observations = dict.fromkeys('abcd', '1')
    separate = assess(goals, observations, {}, dict(a='x', b=None, c='y', d=None))
    mixed = assess(goals, observations, {}, dict.fromkeys('abcd', 'x'))
    assert objective(separate) > objective(mixed)
    assert mixed['cross_set_observation_pairs'] == 4
    assert next(v for v in unmet_conditions(mixed) if v['reason'] == 'cross_set_merge')['sets'] == [0, 1]
    uneven = assess(goals, observations, {}, dict(a='x', b='x', c=None, d=None))
    assert objective(separate) > objective(uneven)


def test_partial_metrics_weight_performers_equally_and_report_empty_members():
    goals = goal_sets([list('abcdefghij'), ['k'], []])
    status = assess(goals, dict.fromkeys('abcdefghijk', '1'), {}, dict.fromkeys('abcdefghij', 'x'))
    group = status['groups'][0]
    assert group['assigned'] == 10 and group['observations'] == 11
    assert group['assigned_fraction'] == pytest.approx(1/3)
    assert group['consolidation_fraction'] == pytest.approx(1/3)
    assert any(v['reason'] == 'empty_identity' for v in unmet_conditions(status))


def test_real_solver_selects_and_exports_best_effort_with_missing_history(tmp_path):
    from stash_ai.recognition.config import load
    config, people = incomplete_source(tmp_path)
    before = config.database.read_bytes()
    with _open_readonly(config.database) as db:
        goals = resolve(db, config, [people], by_performer=False)
        result = dict(goals, **solve(config, goals, db, maximum=12))
    assert result['status'] == 'best_effort'
    assert not any(t['goal_result']['admissible'] for t in result['simulation']['trials'].values())
    assert result['selected_trial'] != 'baseline'
    assert result['best_result'] == result['recommendation']
    assert result['best_result']['changed'] and not result['best_result']['admissible']
    chosen = result['simulation']['trials'][result['selected_trial']]['goal_result']
    baseline = result['simulation']['trials']['baseline']['goal_result']
    assert objective(chosen) > objective(baseline)
    assert all(objective(chosen) >= objective(t['goal_result']) for t in result['simulation']['trials'].values())
    assert not result['search']['proves_optimality']
    missing = next(v for v in result['best_result']['unmet_conditions'] if v['reason'] == 'missing_latest_observations')
    assert missing['count'] == 1 and missing['identity'] == people[0]
    export(config, result, tmp_path/'result.json', tmp_path/'result.yml')
    assert result['yaml_exported'] and result['yaml_export_quality'] == 'best_effort'
    assert result['explanation']['unresolved'] == result['best_result']['unmet_conditions']
    assert 'best_effort' in (tmp_path/'result.yml').read_text().splitlines()[0]
    with _open_readonly(config.database) as db:
        repeated = evaluate(load(tmp_path/'result.yml'), goals, db)['trials']['baseline']['goal_result']
    assert objective(repeated) == objective(chosen)
    assert config.database.read_bytes() == before
    assert 'missing_latest_observations' in summary(result)
    json.dumps(result, allow_nan=False)


@pytest.mark.parametrize('style', ['auto', 'off'])
def test_cli_reports_and_exports_unchanged_best_effort_baseline(tmp_path, capsys, style):
    from stash_ai.recognition.cli import main
    config, people = incomplete_source(tmp_path)
    output, yaml = tmp_path/'result.json', tmp_path/'result.yml'
    args = ['report-merge', '--face-config', str(config.source_path), '--faces-db', str(config.database),
            '--solve', '--max-trials', '1', '--output', str(output), '--yaml-output', str(yaml), '--progress-style', style]
    for person in people:
        args.extend(['--identity-id', person])
    assert main(args) == 0
    result = json.loads(output.read_text())
    assert result['status'] == 'best_effort' and result['selected_trial'] == 'baseline'
    assert result['recommendation'] is None
    assert result['best_result']['changes'] == [] and not result['best_result']['changed']
    assert result['yaml_exported'] and yaml.exists()
    captured = capsys.readouterr()
    assert captured.out == ''
    if style == 'auto':
        assert 'best completed trial: baseline (best_effort)' in captured.err
        assert 'configuration=unchanged' in captured.err and 'assigned=' in captured.err
        assert 'missing_latest_observations' in captured.err
    else:
        assert captured.err == ''


def test_only_cross_set_merge_is_returned_with_explicit_violation(tmp_path):
    config, people = seeded(tmp_path, vectors=[[1., 0, 0, 0], [1., 0, 0, 0]])
    with _open_readonly(config.database) as db:
        goals = resolve(db, config, [[p] for p in people], by_performer=False)
        result = dict(goals, **solve(config, goals, db, maximum=1))
    assert result['status'] == 'best_effort'
    assert result['selected_trial'] == 'baseline'
    assert all(g['fully_merged'] for g in result['simulation']['trials']['baseline']['goal_result']['groups'])
    assert not result['best_result']['fully_solved']
    conflict = next(v for v in result['best_result']['unmet_conditions'] if v['reason'] == 'cross_set_merge')
    assert conflict['sets'] == [0, 1] and conflict['observation_pairs'] == 9
    assert 'disjoint=False' in summary(result)


def test_interrupt_before_baseline_does_not_invent_a_result(tmp_path):
    config, people = seeded(tmp_path)
    def progress(*args):
        raise KeyboardInterrupt
    with _open_readonly(config.database) as db:
        goals = resolve(db, config, [people], by_performer=False)
        result = dict(goals, **solve(config, goals, db, maximum=4, progress=progress))
    assert result['interrupted'] and result['status'] == 'no_completed_trials'
    assert result['best_result'] is None and result['selected_trial'] is None
    export(config, result, tmp_path/'result.json', tmp_path/'result.yml')
    assert not result['yaml_exported'] and not (tmp_path/'result.yml').exists()
    assert result['explanation']['selected_trial'] is None
    assert 'no completed trial' in summary(result)


def test_interrupt_keeps_completed_best_effort_and_excludes_unfinished_trial(tmp_path):
    config, people = incomplete_source(tmp_path)
    def progress(name, stage, item):
        if name == 'trial-2' and stage == 'start':
            raise KeyboardInterrupt
    with _open_readonly(config.database) as db:
        goals = resolve(db, config, [people], by_performer=False)
        result = solve(config, goals, db, maximum=8, progress=progress)
    assert result['interrupted'] and result['status'] == 'best_effort'
    assert set(result['simulation']['trials']) == {'baseline', 'trial-1'}
    assert result['best_result']['trial'] in result['simulation']['trials']


def test_interrupt_after_ranking_does_not_select_unrecorded_trial(tmp_path, monkeypatch):
    from stash_ai.recognition import merge_solver
    replay = merge_solver.replay
    def interrupted_replay(*args, **kwargs):
        consume = kwargs['consume']
        def interrupt_after_consume(name, *values):
            item = consume(name, *values)
            if name != 'baseline':
                raise KeyboardInterrupt
            return item
        kwargs['consume'] = interrupt_after_consume
        return replay(*args, **kwargs)
    monkeypatch.setattr(merge_solver, 'replay', interrupted_replay)
    config, people = incomplete_source(tmp_path)
    with _open_readonly(config.database) as db:
        goals = resolve(db, config, [people], by_performer=False)
        result = solve(config, goals, db, maximum=8)
    assert result['interrupted']
    assert set(result['simulation']['trials']) == {'baseline'}
    assert result['best_result']['trial'] == 'baseline'
