from types import SimpleNamespace

import pytest

from stash_ai.recognition.stash import RecognitionStash,RecognitionReader
from stash_ai.stash_api.transport import StashTransport


def reader(gql):
    value=object.__new__(RecognitionReader)
    value.gql=gql
    return value


def test_bounded_alias_batches_validate_every_requested_id_and_null():
    calls=[]
    def gql(query,variables):
        assert StashTransport._read_only(query,variables)
        calls.append((query,variables))
        return {f'n{i}':{'id':v} if v!='2' else None for i,v in enumerate(variables.values())}
    client=reader(gql)
    result=client.performers([str(i) for i in range(1,67)])
    assert len(calls)==3 and max(len(v) for _,v in calls)==32
    assert result['2'] is None and result['66']['id']=='66'
    assert client.scenes(['1','1'])=={'1':{'id':'1'}}
    with pytest.raises(RuntimeError):
        client.performers(['not-id'])


def test_one_wrong_entry_does_not_authorize_another_target():
    client=reader(lambda q,v:{'n0':{'id':'99'},'n1':{'id':'2'}})
    result=client.performers(['1','2'])
    assert isinstance(result['1'],RuntimeError) and result['2']=={'id':'2'}


def test_envelope_error_falls_back_to_separate_proven_reads():
    calls=[]
    def gql(query,variables):
        calls.append(query)
        if 'Batch' in query:
            raise RuntimeError('one field unavailable')
        if variables['id']=='2':
            raise RuntimeError('target 2 unavailable')
        return {'findPerformer':{'id':variables['id']}}
    result=reader(gql).performers(['1','2'])
    assert result['1']=={'id':'1'} and isinstance(result['2'],RuntimeError)
    assert len(calls)==3
