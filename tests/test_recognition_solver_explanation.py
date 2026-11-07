from stash_ai.recognition.solver_explanation import blockers


def test_dominant_reason_is_per_unresolved_set_and_keeps_cue_zero():
    groups = [dict(index=i, fully_merged=i == 1) for i in range(3)]
    trace = [dict(observation='a', reason=r, cue=0) for r in ('support', 'margin', 'support')]
    inputs = [dict(index=i, certificates=[dict(reason='missing witness', scene='42')]) for i in range(3)]
    assert blockers(groups, trace, {'a': (0, 'p', '17')}, inputs) == [
        'set 1: support (2 sampled) · scene 17 cue 0', 'set 3: missing witness · scene 42']
