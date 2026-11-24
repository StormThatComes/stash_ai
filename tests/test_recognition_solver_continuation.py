"""A plateau and outstanding worker results must not close useful search."""
import pytest

from stash_ai.recognition.merge_solver_search import BeamSearch, PREFIX


@pytest.mark.parametrize('early', [False, True])
def test_solution_beyond_old_plateau_limit(early):
    def factory(parent, changes, family, mode):
        if parent != 'baseline' and not mode:
            return
        for i in range(500):
            if mode:
                yield 'face', 'local_probe', {'probe': i}
            else:
                yield 'coordinate_probe', {'x': i}
    beam = BeamSearch(['face'], factory, 500, stop_on_stagnation=early)
    beam.observe('baseline', {}, (1, 0, 0), 'baseline')
    solved = False
    for i in range(499):
        try:
            family, phase, parent, strategy, changes = beam.next()
        except StopIteration:
            break
        solved = changes.get('x') == 100
        beam.completed(phase, parent=parent)
        beam.observe(str(i), changes, (2 if solved else 0, -1, -i),
                     'solved' if solved else 'same', family=family)
        if solved:
            break
    assert solved is not early
    if early:
        assert beam.stop_reason == 'stagnation_after_refinement'
    else:
        assert beam.executed > 120 and beam.incumbent != 'baseline'


@pytest.mark.parametrize('scheduling', ['batch', 'async'])
def test_eight_pending_results_are_drained_before_exhaustion(tmp_path, monkeypatch, scheduling):
    from tests.test_recognition_merge_solver_exploration import auxiliary_goals
    from stash_ai.recognition.fragmentation_report import _open_readonly
    from stash_ai.recognition.merge_solver import solve
    from stash_ai.recognition.merge_solver_bidirectional import BidirectionalSearch
    from stash_ai.recognition import solver_execution
    config, goals = auxiliary_goals(tmp_path)
    monkeypatch.setattr(solver_execution.os, 'process_cpu_count', lambda: 16, raising=False)
    monkeypatch.setattr(BidirectionalSearch, 'next', lambda *a, **k: None)
    issued = []
    waiting = []
    def offer(self):
        if len(issued) < 8:
            changes = {PREFIX+'face_floor': .801 + len(issued)*.001}
        elif self.executed < 8:
            waiting.append(self.executed)
            self.stop_reason = 'frontier_exhausted'
            raise StopIteration
        elif len(issued) == 8:
            changes = {PREFIX+'face_floor': .78}
        else:
            self.stop_reason = 'frontier_exhausted'
            raise StopIteration
        issued.append(changes)
        return 'hybrid', 'baseline_exploration', 'baseline', 'coordinate_probe', changes
    monkeypatch.setattr(BeamSearch, 'next', offer)
    before = config.database.read_bytes()
    with _open_readonly(config.database) as db:
        result = solve(config, goals, db, maximum=10, targeted_trials=0, feasibility_trials=0,
                       execution=dict(cpus=16, trials=8, scheduling=scheduling,
                                      memory_bytes=16*1024**3))
    assert waiting, 'Exercise exhaustion while completed results are still pending.'
    assert result['best_result']['fully_solved']
    assert result['search']['attempted'] == 10
    assert result['simulation']['trials']['trial-9']['goal_result']['fully_solved']
    assert config.database.read_bytes() == before


def test_stagnation_policy_cli_and_yaml(tmp_path):
    from stash_ai.recognition.cli import parser
    from stash_ai.recognition.config import load
    argv = ['report-merge', '--performer-set', '1,2']
    assert parser().parse_args(argv+['--solve', '--stop-on-stagnation']).stop_on_stagnation is True
    assert parser().parse_args(argv+['--solve', '--no-stop-on-stagnation']).stop_on_stagnation is False
    with pytest.raises(SystemExit):
        parser().parse_args(argv+['--stop-on-stagnation'])
    path = tmp_path/'config.yml'
    for literal, expected in [('false', False), ('true', True)]:
        path.write_text('version: 2\nexecution:\n  solver:\n    stop_on_stagnation: '+literal+'\n')
        assert load(path).data['execution']['solver_stop_on_stagnation'] is expected
    path.write_text('version: 2\nexecution:\n  solver:\n    stop_on_stagnation: 1\n')
    with pytest.raises(ValueError, match='must be boolean'):
        load(path)
