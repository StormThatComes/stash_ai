from stash_ai.recognition.contact_threshold import path_interval
from stash_ai.recognition.contact_holdout import visible_scores
from stash_ai.recognition.sequence_validation import sequence_metrics


def test_entire_swap_requires_an_entry_and_return():
    edges = [((.9,.88),(.88,.9)),((.83,.4),(.4,.83))]
    assert path_interval(edges)==(.83,.4)


def test_two_distinct_face_roots_report_split_and_designated_reference():
    decisions = {o:dict(accepted_identity=p,evidence_kind='face',evidence=dict(reason='known_identity'))
                 for o,p in [('a','X'),('b','Y')]}
    groups = {('s',0,'A'):['a'],('s',1,'A'):['b']}
    labels = dict(a='A',b='A')
    assert visible_scores(groups,decisions,labels)['split_reference']==2
    assert visible_scores(groups,decisions,labels,['a'])['fragment']==1


def test_abstaining_inside_contact_preserves_only_endpoint_recovery():
    labels, locations, samples, decisions = {}, {}, {}, {}
    for p in 'AB':
        for i in range(3):
            oid = p+str(i)
            labels[oid], locations[oid] = p,('s',i)
            samples[oid] = dict(slices=['entangled'] if i==1 else [])
            decisions[oid] = dict(accepted_identity=None if i==1 else p,evidence_kind='person')
    result = sequence_metrics(labels,samples,locations,decisions)
    assert result['endpoint_returns']==result['contact_interior']['withheld']==1
    assert result['contact_interior']['correct']==0
