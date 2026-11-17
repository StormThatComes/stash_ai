"""The default constrained search and the explicit legacy solver share replay contracts."""
from contextlib import closing
from pathlib import Path

import pytest
import yaml

from stash_ai.recognition import cli
from stash_ai.recognition.config import load
from stash_ai.recognition.config_check import report
from stash_ai.recognition.fragmentation_report import _open_readonly
from stash_ai.recognition.merge_goals import resolve
from stash_ai.recognition.merge_report_policy import patched
from stash_ai.recognition.merge_solver import solve
from stash_ai.recognition.solver_config_policy import StrictPolicy
from tests.test_recognition_config_check import profile
from tests.test_recognition_merge_report import seeded


def clean(config):
    counts = report(config)['counts']
    return counts['warning'] == counts['advice'] == 0


def test_both_shipped_profiles_can_be_repaired_in_multiple_directions():
    root = Path(__file__).resolve().parents[1]
    for name in ('facial_recognition.yml', 'facial_recognition_solver.yml'):
        config = load(root / name)
        assert config.data['execution']['solver_strict'] is True
        policy = StrictPolicy(config)
        assert len(policy.input_issues) == 2
        repairs = policy.repairs(config, {}, initial=True)
        assert len(repairs) >= 2
        assert all(clean(patched(config, patch)) for patch in repairs)
        assert any('sapiensid.global_matching.match_threshold' in patch for patch in repairs)
        assert any('sapiensid.global_matching.ambiguity_margin' in patch for patch in repairs)


def test_projection_preserves_routes_and_couples_adjacent_and_view_thresholds():
    source = load(None)
    policy = StrictPolicy(source)
    config = patched(source, policy.repairs(source, {}, initial=True)[0])
    for changes in ({'sapiensid.identities.track_similarity_threshold': .845},
                    {'clustering.view_match_threshold': config.data['recognition']['match_threshold']}):
        results = policy.repairs(config, changes)
        assert results and all(clean(patched(config, patch)) for patch in results)
    assert not policy.repairs(config, {'clustering.same_scene_match_bonus': 0.})
    inherited = patched(config, {'sapiensid.identities.tracklet_matching': True,
                                 'sapiensid.identities.endpoint_matching': None})
    policy = StrictPolicy(inherited)
    disabled = patched(inherited, {'sapiensid.identities.endpoint_matching': False})
    assert 'sapiensid.identities.endpoint_matching' in policy.violations(disabled)


@pytest.mark.parametrize('mode', ['full', 'diff'])
def test_default_replays_only_clean_candidates_and_compares_to_unrepaired_yaml(tmp_path, monkeypatch, mode):
    import json
    from stash_ai.recognition import replay_simulation
    config, _ = seeded(tmp_path, vectors=[[1., 0.], [.83, (1-.83**2)**.5]])
    # Exercise a real pre-existing v1 YAML with no mode key.
    document = yaml.safe_load(config.source)
    document.pop('execution')
    config.source_path.write_text(yaml.safe_dump(document))
    config = load(config.source_path, database=config.database)
    original = config.source_path.read_bytes()
    seen = []
    rebuild = replay_simulation.rebuild_selected
    def traced(state, scans, variant):
        seen.append(clean(variant))
        return rebuild(state, scans, variant)
    monkeypatch.setattr(replay_simulation, 'rebuild_selected', traced)
    output = tmp_path / 'winner.yml'
    result_path = tmp_path / 'report.json'
    args = ['report-merge', '--face-config', str(config.source_path), '--faces-db', str(config.database),
            '--performer-set', '900,901', '--solve', '--max-trials', '6', '--progress-style', 'off',
            '--solver-report', 'full', '--yaml-mode', mode, '--yaml-output', str(output), '--output', str(result_path)]
    assert cli.main(args) == 0
    result = json.loads(result_path.read_text())
    assert seen[0] is False and all(seen[1:]) and 2 <= len(seen) <= 6
    assert result['config_policy']['mode'] == 'strict'
    assert result['config_policy']['input_control_replays'] == 1
    assert result['search']['attempted'] == len(seen)
    assert result['simulation']['input_control']['fully_solved'] is False
    assert result['best_result']['fully_solved']
    assert result['simulation']['trials']['baseline']['versus_baseline']['newly_joined_pairs'] > 0
    for trial in result['simulation']['trials'].values():
        config_patch = {v['parameter']: v['proposed'] for v in trial['changes']}
        assert clean(patched(config, config_patch))
    if mode == 'diff':
        for section, edits in yaml.safe_load(output.read_text()).items():
            document.setdefault(section, {}).update(edits)
        output.write_text(yaml.safe_dump(document))
    assert clean(load(output)) and config.source_path.read_bytes() == original


@pytest.mark.parametrize('path', ['pose.refinement.min_presence', 'pose.point_quality.min_presence'])
def test_gate_blocks_scan_rule_and_reports_it_before_replay(tmp_path, monkeypatch, path):
    from stash_ai.recognition import merge_solver
    c, _ = seeded(tmp_path)
    config = load(profile(tmp_path, {path: 1., **({'pose.refinement.min_presence': None} if 'point_quality' in path else {})}), database=c.database)
    monkeypatch.setattr(merge_solver, 'replay', lambda *a, **kw: pytest.fail('Unexpected replay'))
    with closing(_open_readonly(c.database)) as db:
        result = solve(config, {'scope': 'scope', 'namespace': 'ns', 'groups': []}, db)
    assert result['status'] == 'strict_config_blocked' and result['best_result'] is None
    blocked = {row['rule']: row for row in result['config_policy']['blockers']}
    assert path in blocked['pose.refinement.min_presence-ceiling']['settings']
    assert result['search']['executed'] == 0


def test_strict_budget_checkpoint_and_escape_hatch(tmp_path):
    c, _ = seeded(tmp_path)
    c = patched(c, {'execution.solver_strict': True})
    path = tmp_path / 'strict.json'
    with closing(_open_readonly(c.database)) as db:
        goals = resolve(db, c, [['900','901']])
        limited = solve(c, goals, db, maximum=1)
        assert limited['status'] == 'strict_config_budget' and limited['recommendation'] is None
        result = solve(c, goals, db, maximum=5, checkpoint=path)
        resumed = solve(c, goals, db, maximum=5, checkpoint=path, resume=True)
        assert resumed['best_result'] == result['best_result']
        assert resumed['simulation']['input_control'] == result['simulation']['input_control']
        with pytest.raises(ValueError, match='[Cc]heckpoint'):
            solve(c, goals, db, maximum=5, checkpoint=path, resume=True, strict=False)
        assert solve(c, goals, db, maximum=1, strict=False)['selected_trial'] == 'baseline'
    assert cli.parser().parse_args(['report-merge', '--solve', '--no-strict', '--performer-set', '900,901']).strict is False


def test_portrait_preference_is_repaired_before_search(tmp_path):
    config = load(profile(tmp_path, {'portraits.weights.resolution': 0.}))
    policy = StrictPolicy(config)
    repaired = policy.repairs(config, {}, initial=True)
    assert repaired
    assert all(p['portraits.weights.resolution'] > 0 and clean(patched(config, p)) for p in repaired)


def test_parallel_search_only_returns_compliant_trials(tmp_path, monkeypatch):
    from benchmarks.benchmark_merge_solver import fixture
    monkeypatch.setattr('stash_ai.recognition.solver_execution.os.process_cpu_count', lambda: 4, raising=False)
    config, goals = fixture(tmp_path / 'input', 'disjoint')
    execution = dict(cpus=4, trials=2, scheduling='batch', memory_bytes=128 * 1024**3)
    with closing(_open_readonly(config.database)) as db:
        result = solve(config, goals, db, maximum=7, strict=True, execution=execution)
    trials = result['simulation']['trials']
    assert result['search']['execution']['trials'] == 2
    assert result['config_policy']['input_control_replays'] == 1
    assert 2 < result['search']['executed'] <= 7 and result['search']['failed_trials'] == 0
    for row in trials.values():
        assert clean(patched(config, {c['parameter']: c['proposed'] for c in row['changes']}))
