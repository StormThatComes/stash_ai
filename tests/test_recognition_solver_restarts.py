import pytest

from stash_ai.recognition.merge_solver_search import BeamSearch, PREFIX
from tests.test_solver_mechanism_planning import milestone


def test_third_restart_solves_without_repeating_earlier_proposals():
    opened = []
    def factory(parent, changes, family, mode):
        if mode == 'restart' and parent == 'baseline':
            opened.append(parent)
            for i in range(80):
                yield 'face', 'restart_probe', {'x': i}
    beam = BeamSearch(['face'], factory, 100)
    beam.observe('baseline', {}, (1, 0, 0), 'baseline')
    seen = set()
    for i in range(99):
        family, phase, parent, _, changes = beam.next()
        assert phase == 'restart' and changes['x'] not in seen
        seen.add(changes['x'])
        solved = changes['x'] == 25
        beam.completed(phase, parent=parent)
        beam.observe(str(i), changes, (2 if solved else 0, -1, -i), 'solved' if solved else 'same', family=family)
        if solved:
            break
    assert beam.restarts == 3 and len(seen) == 26 and opened == ['baseline']
    assert beam.incumbent != 'baseline'


def test_finite_restart_streams_really_exhaust():
    def factory(parent, changes, family, mode):
        if mode == 'restart' and parent == 'baseline':
            yield 'face', 'restart_probe', {'x': 1}
    beam = BeamSearch(['face'], factory, 500)
    beam.observe('baseline', {}, (1, 0, 0), 'baseline')
    family, phase, parent, _, changes = beam.next()
    beam.completed(phase, parent=parent)
    beam.observe('one', changes, (0, -1, -1), 'other', family=family)
    with pytest.raises(StopIteration):
        beam.next()
    assert beam.stop_reason == 'frontier_exhausted' and beam.restarts == 1


def test_new_view_gets_followups_at_plateau_without_changing_winner_rank():
    def factory(parent, changes, family, mode):
        if parent == 'view':
            yield 'mechanism_plan', dict(changes, match=True)
    beam = BeamSearch(['face'], factory, 200)
    beam.observe('baseline', {}, (1, 0, 0), 'same', mechanisms=milestone(1))
    beam.stale = beam.patience
    beam.observe('view', {'view': True}, (1, -1, -1), 'same', family='face', mechanisms=milestone(3))
    assert beam.stale == 0 and beam.last_progress and beam.progress_events == 1
    assert beam.incumbent == 'baseline' and beam.best_quality == (1,)
    family, phase, parent, _, changes = beam.next()
    assert parent == 'view' and changes['match']
    beam.completed(phase, parent=parent)
    beam.observe('merged', changes, (2, -2, -2), 'merged', family=family, mechanisms=milestone(3))
    assert beam.incumbent == 'merged'


def test_repeated_or_unsafe_milestone_cannot_reset_patience_again():
    beam = BeamSearch(['face'], lambda *a: iter(()), 200)
    beam.observe('baseline', {}, (1, 0, 0), 'same', mechanisms=milestone(1))
    beam.observe('view', {'view': True}, (1, -1, -1), 'same', mechanisms=milestone(2), family='face')
    assert beam.last_progress
    for i in range(8):
        beam.observe(str(i), {'x': i}, (0, -1, -i), str(i), family='face',
                     mechanisms=milestone(3) if i % 2 else milestone(2), admissible=i % 2 == 0)
        assert not beam.last_progress
    assert beam.stale == 8 and beam.progress_events == 1


def test_fixed_scene_and_support_counts_grant_progress_once():
    from stash_ai.recognition.merge_solver_milestones import measurements
    def state(scenes, core):
        return dict(construction=dict(groups=[dict(index=0, covered_scenes=scenes,
                    required_scenes=3, core_support=core, core_required=4)]))
    beam = BeamSearch(['face'], lambda *a: iter(()), 200)
    beam.observe('baseline', {}, (1, 0, 0), 'same', intermediate=measurements(state(1, 2), {}))
    beam.observe('support', {'x': 1}, (1, -1, -1), 'same', family='face',
                 intermediate=measurements(state(1, 3), {}))
    assert beam.last_progress
    beam.observe('scene', {'x': 2}, (1, -1, -2), 'new', family='face',
                 intermediate=measurements(state(2, 3), {}))
    assert beam.last_progress
    beam.observe('repeat', {'x': 3}, (1, -1, -3), 'same', family='face',
                 intermediate=measurements(state(1, 3), {}))
    assert not beam.last_progress and beam.progress_events == 2


def test_joint_restarts_cover_routes_and_preserve_support_domains(tmp_path):
    from benchmarks.benchmark_merge_solver import fixture
    from stash_ai.recognition.merge_solver_restarts import joint_probes
    from stash_ai.recognition.merge_solver_domains import normalize
    from stash_ai.recognition.merge_report_policy import patched
    config, _ = fixture(tmp_path, 'query')
    options = {PREFIX+'face_floor': [.2, .4, .6], PREFIX+'person_threshold': [.3, .6, .9],
               PREFIX+'min_support': [1, 2, 3], 'clustering.view_min_support': [1, 2, 4],
               'clustering.view_match_threshold': [.4, .6, .8], 'recognition.match_threshold': [.5, .8]}
    candidates = list(joint_probes(config, {}, options))
    assert any(len(p) == len(options) for p in candidates)
    assert any('clustering.view_min_support' in p and PREFIX+'person_threshold' in p for p in candidates)
    for changes in candidates:
        assert all(v >= 2 for k, v in changes.items() if k.endswith('min_support'))
        patched(config, normalize(config, changes))
