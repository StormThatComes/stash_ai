import pytest

from benchmarks.benchmark_merge_solver import fixture
from stash_ai.recognition.fragmentation_report import _open_readonly
from stash_ai.recognition.merge_report_policy import parameter, patched
from stash_ai.recognition.merge_solver import solve
from stash_ai.recognition.merge_solver_domains import normalize
from stash_ai.recognition.merge_solver_polish import ablations, allowance
from stash_ai.recognition.merge_solver_robustness import probes
from stash_ai.recognition.merge_solver_search import BeamSearch, PREFIX, cadence, quality_gain
from tests.test_recognition_merge_solver_exploration import auxiliary_goals


def test_diverse_outcomes_expand_beam_and_repeated_plateau_shrinks_it(tmp_path):
    config, _ = fixture(tmp_path, 'query')
    search = BeamSearch(['hybrid'], lambda *a: iter(()), 500, config)
    search.observe('baseline', {}, (2, 0, 0), 'base')
    for i in range(16):
        search.observe(str(i), {PREFIX+'face_floor': i/20}, (1, -1, -i), str(i), family='hybrid')
    assert search.width == 8 and len(search.beam) == 8 and 'baseline' in search.beam
    for i in range(16, 32):
        search.observe(str(i), {PREFIX+'face_floor': .7}, (1, -1, -i), 'same', family='hybrid')
    assert search.width == 3 and len(search.beam) <= 3 and 'baseline' in search.beam


def test_reward_uses_only_first_improved_objective():
    assert quality_gain((1, 0), (0, 10000)) == 0
    assert quality_gain((0, 10000), (1, 0)) == 1
    assert quality_gain((1, .1), (1, .5)) > quality_gain((1, .1), (1, .2)) > 0
    assert quality_gain((1, .1), (1, .1)) == 0


def test_operator_allocation_uses_gain_per_cost_without_changing_winner():
    search = BeamSearch(['face'], lambda *a: iter(()), 100)
    search.observe('baseline', {}, (0, 0, 0), 'base')
    search.observe('slow', {}, (1, 0, 0), 'a', family='face', operator='slow', elapsed=100)
    search.observe('fast', {}, (1.5, 0, 0), 'b', family='face', operator='fast', elapsed=1)
    assert search.operators['slow']['gain'] > search.operators['fast']['gain']
    assert next(search.operations({'slow': iter(['slow']), 'fast': iter(['fast'])})) == 'fast'
    assert search.incumbent == 'fast'


@pytest.mark.parametrize('default', [2, 5])
def test_lane_cadence_keeps_exploration_but_rewards_recent_progress(default):
    assert cadence({}, default) == default
    assert cadence({'recent': [.1, .2]}, default) < default
    assert cadence({'recent': [0, 0, 0, 0]}, default) > default
    assert cadence({'recent': [0, 0, .1, .1]}, default) < default


def test_polishing_reserve_scales_with_patch_without_exceeding_global_budget():
    assert allowance(19, range(50), range(50)) == 0
    assert allowance(500, range(12), range(12)) > allowance(500, range(1), range(1))
    for maximum in (20, 30, 100, 500):
        assert 4 <= allowance(maximum, range(50), range(50)) <= min(24, maximum//4)


def test_ablation_restores_dependencies_and_removes_inactive_parameters_first(tmp_path):
    config, _ = fixture(tmp_path, 'query')
    floor = PREFIX+'face_floor'
    extra = 'recognition.cooccurrence_merge_threshold'
    changes = normalize(config, {floor: .7, extra: .99, 'recognition.match_threshold': .3})
    candidates = list(ablations(config, changes, {floor: 20}))
    assert candidates and floor in candidates[0]
    assert any(extra not in p and p.get(floor) == .7 for p in candidates)
    for patch in candidates:
        assert len(patch) < len(changes)
        patched(config, patch)


def test_robustness_covers_booleans_coordinates_and_joint_boundaries(tmp_path):
    config, _ = fixture(tmp_path, 'query')
    a, b = PREFIX+'face_floor', PREFIX+'person_threshold'
    switch = 'clustering.anchor_view_matching'
    changes = normalize(config, {a: .7, b: .8})
    active = {a: 3, b: 2, switch: 1, PREFIX+'min_support': 1}
    candidates = list(probes(config, changes, active, limit=12, options={a: [.6999, .7001]}))
    assert any(p.get(a) == .6999 for p in candidates)
    assert any(p.get(a, parameter(config.data, a)) != .7 and
               p.get(b, parameter(config.data, b)) != .8 for p in candidates[:3])
    assert any(p.get(switch, parameter(config.data, switch)) != parameter(config.data, switch) for p in candidates)
    assert len(candidates) == len({tuple(sorted(p.items())) for p in candidates}) <= 12
    for patch in candidates:
        patched(config, patch)


def test_full_replay_ablation_removes_redundant_gate_and_resumes(tmp_path, monkeypatch):
    config, goals = auxiliary_goals(tmp_path)
    def offer(self):
        if not self.normal_completed:
            return 'hybrid', 'adaptive', 'baseline', 'coordinate_probe', {
                PREFIX+'face_floor': .78, 'recognition.cooccurrence_merge_threshold': .99}
        self.stop_reason = 'frontier_exhausted'
        raise StopIteration
    monkeypatch.setattr(BeamSearch, 'next', offer)
    path = tmp_path/'checkpoint.json'
    phases = {}
    def stop(name, stage, item):
        if stage == 'start':
            phases[name] = item.get('search_phase')
        if stage == 'complete' and phases.get(name) == 'minimization':
            raise KeyboardInterrupt
    before = config.database.read_bytes()
    with _open_readonly(config.database) as db:
        first = solve(config, goals, db, maximum=30, checkpoint=path, progress=stop)
        assert first['interrupted']
        result = solve(config, goals, db, maximum=30, checkpoint=path, resume=True)
        fresh = solve(config, goals, db, maximum=30)
    assert result['search']['restored_trials'] == len(first['simulation']['trials'])
    assert result['best_result']['fully_solved']
    assert [c['parameter'] for c in result['best_result']['changes']] == [PREFIX+'face_floor']
    assert result['best_result']['changes'] == fresh['best_result']['changes']
    assert result['search']['executed_by_phase']['minimization'] > 0
    assert result['search']['executed_by_phase']['robustness'] > 0
    assert config.database.read_bytes() == before
