from stash_ai.recognition.solver_explanation import blockers


def test_dominant_reason_is_per_unresolved_set_and_keeps_cue_zero():
    groups = [dict(index=i, fully_merged=i == 1) for i in range(3)]
    trace = [dict(observation='a', reason=r, cue=0) for r in ('support', 'margin', 'support')]
    inputs = [dict(index=i, certificates=[dict(reason='missing witness', scene='42')]) for i in range(3)]
    assert blockers(groups, trace, {'a': (0, 'p', '17')}, inputs) == [
        'set 1: support (2 sampled) · scene 17 cue 0', 'set 3: missing witness · scene 42']


def test_structured_parameter_names_the_actual_changed_rule():
    from stash_ai.recognition.solver_explanation import parameter_change
    old, new = {'any':[{'min':2}]}, {'any':[{'min':3}]}
    assert parameter_change(dict(parameter='rules',previous=old,current=new)) == 'rules.any[0].min 2 → 3'


def test_limited_hints_do_not_mark_replay_as_failed(tmp_path, monkeypatch, capsys):
    import json
    from stash_ai.recognition.cli import main
    from stash_ai.recognition.resource_errors import ResourceLimit
    from tests.test_recognition_merge_trial_progress import arguments, auxiliary_goals
    config, goals = auxiliary_goals(tmp_path, sets=2)
    def limited(*args, **kwargs):
        raise ResourceLimit('synthetic diagnostic cap')
    monkeypatch.setattr('stash_ai.recognition.merge_solver_hybrid.analyse', limited)
    for style in ('compact', 'verbose'):
        destination = tmp_path / style
        destination.mkdir()
        assert main(arguments(config, goals, destination, style)) == 0
        output = capsys.readouterr()
        report = json.loads(output.out)
        assert report['selected_trial'] and report['search']['failed_trials'] == 0
        assert output.err.count('Partial hybrid diagnostics') == 1
        if style == 'compact':
            assert '· scene ' in output.err and ' cue ' in output.err
