from types import SimpleNamespace
import numpy as np
from stash_ai.recognition.config import defaults
from stash_ai.recognition.disk_body_modes import Records,ComparisonBudget
from stash_ai.recognition.hybrid_modes import body_modes


def test_complete_link_certificates_preserve_all_witnesses_and_bound_work():
    config=SimpleNamespace(data=defaults())
    config.data['sapiensid']['matching'].update(person_threshold=.8,min_support=1)
    rng=np.random.default_rng(42)
    rows=[]
    for i in range(180):
        vector=np.zeros(4096,dtype='<f4');vector[0]=1
        vector[1:9]=rng.normal(0,.005,8);vector/=np.linalg.norm(vector)
        rows.append(dict(id=str(i),scan='s',seconds=i,quality='{}',embedding=vector.tobytes()))
    expected=body_modes(rows,config)
    spool=Records();budget=ComparisonBudget(config,SimpleNamespace(advance=lambda:None))
    try:
        for row in rows:spool.add(row)
        actual,support=spool.modes(config,budget)
        assert support==len(rows) and len(actual)==len(expected)==1
        assert np.array_equal(actual[0][:],expected[0])
        assert budget.count<180  # Previous scalar path performs 16110 dots.
    finally:spool.close()


def test_metadata_veto_does_not_read_vectors():
    from stash_ai.recognition.body_certificates import pairs
    config=SimpleNamespace(data=defaults())
    bank={p:dict(refs=[],cores=[],faces=[],support=10,cues={('same',0)},missing_original=False) for p in ('a','b')}
    def unexpected(*args):raise AssertionError('metadata veto read vectors')
    result=list(pairs([('a','b')],bank,config,SimpleNamespace(compare=unexpected,stats={})))
    assert result[0]['blockers']==['person_copresence']
    assert result[0]['score'] is None and result[0]['measurements']=='not_computed_metadata_veto'
