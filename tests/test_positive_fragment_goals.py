from stash_ai.recognition.positive_observations import assess
from stash_ai.recognition.sequence_validation import sequence_metrics


def test_provisional_body_requires_independently_recovered_owner():
    label = dict(set=0, scene='7', requested_observation_ids=['rear'], observation_ids=['rear'])
    goals = dict(positive_observations=[label], positive_reference_ids={0:['root']})
    for owner, passed in ((None, False), ('new', False), ('performer', True)):
        status = dict(admissible=True, fully_solved=True)
        membership = {'root':'performer', 'rear':owner}
        assess(goals,status,membership,membership)
        assert status['fully_solved'] == passed


def test_ordinary_switch_and_midshot_takeover():
    labels = {'a':'A','b':'B','c':'C','d':'C','e':'B','f':'B'}
    cues = dict(a=0,b=0,c=1,d=2,e=1,f=2)
    samples = {k:dict(slices=['entrant'] if person=='C' else []) for k,person in labels.items()}
    decisions = {k:dict(accepted_identity=v) for k,v in dict(a='1',b='2',c='1',d='3',e='2',f='2').items()}
    result = sequence_metrics(labels,samples,{k:('s',v) for k,v in cues.items()},decisions)
    assert result['entrant_takeovers']==result['scene_id_switches']==1
