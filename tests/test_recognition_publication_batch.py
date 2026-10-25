import json
import pytest
from stash_ai.recognition.database import FaceDatabase
from stash_ai.recognition.publication_batch import ensure_people,publish_people
from stash_ai.recognition.publication import PendingRecognition
from stash_ai.recognition.stash import RecognitionStash
from stash_ai.runtime.report import RunReport
from tests.helpers.recognition import Stash
from tests.test_recognition_merge_report import seeded


class Batched(Stash):
    def __init__(self):super().__init__();self.batch_reads=[];self.batch_writes=[];self.lost=False
    def performers(self,ids):
        self.batch_reads.append(list(ids));return {i:dict(self.profiles[i]) for i in ids}
    def add_scene_people(self,scene,people):
        self.batch_writes.append(list(people))
        self.links[scene]=list(dict.fromkeys(self.links[scene]+people))
        if self.lost:raise TimeoutError('lost batch acknowledgement')
        return self.scene(scene)


def bind(state,client,people):
    for index,person in enumerate(people):
        identifier=str(100+index)
        token=f'stash-ai-recognition:{state.installation}:{person}'
        state.connection.execute('UPDATE people SET token=? WHERE id=?',(token,person))
        client.profiles[identifier]=dict(id=identifier,details=token)
        state.connection.execute('UPDATE people SET performer=? WHERE id=?',(identifier,person))
    state.connection.commit()


def test_batch_publication_reads_and_adds_once_preserving_manual_links(tmp_path):
    config,people=seeded(tmp_path);client=Batched();report=RunReport(False)
    with FaceDatabase(config.database) as state:
        bind(state,client,people)
        targets=ensure_people(state,client,report,'scope','1',people)
        assert len(client.batch_reads)==1
        publish_people(state,client,report,'scope','1',targets)
        assert client.batch_writes==[list(targets.values())]
        assert set(client.links['1'])=={'77',*targets.values()}
        assert len(client.batch_reads)==2
        publish_people(state,client,report,'scope','1',targets)
        assert len(client.batch_writes)==1
        assert {r[0] for r in state.connection.execute('SELECT status FROM publications')}=={'observed'}


def test_unknown_batch_reconciles_observed_but_never_retries_missing(tmp_path):
    config,people=seeded(tmp_path);client=Batched();report=RunReport(False)
    with FaceDatabase(config.database) as state:
        bind(state,client,people);targets=ensure_people(state,client,report,'scope','1',people)
        client.lost=True
        with pytest.raises(TimeoutError):publish_people(state,client,report,'scope','1',targets)
        assert {r[0] for r in state.connection.execute('SELECT status FROM publications')}=={'unknown'}
        linked=list(client.links['1']);client.links['1']=['77']
        with pytest.raises(PendingRecognition,match='no blind retry'):
            publish_people(state,client,report,'scope','1',targets)
        assert len(client.batch_writes)==1
        client.links['1']=linked
        publish_people(state,client,report,'scope','1',targets)
        assert len(client.batch_writes)==1


def test_changed_profile_ownership_blocks_entire_batch(tmp_path):
    config,people=seeded(tmp_path);client=Batched();report=RunReport(False)
    with FaceDatabase(config.database) as state:
        bind(state,client,people);targets=ensure_people(state,client,report,'scope','1',people)
        client.profiles[next(iter(targets.values()))]['details']='manual owner'
        with pytest.raises(PendingRecognition,match='ownership changed'):
            publish_people(state,client,report,'scope','1',targets)
        assert not client.batch_writes


def test_bulk_add_contract_confirms_every_requested_performer():
    client=RecognitionStash.__new__(RecognitionStash);calls=[]
    def gql(query,variables):
        calls.append(variables)
        return {'bulkSceneUpdate':[{'id':'1','performers':[{'id':'77'},{'id':'100'},{'id':'101'}]}]}
    client.gql=gql
    client.add_scene_people('1',['100','101'])
    assert calls==[{'input':{'ids':['1'],'performer_ids':{'ids':['100','101'],'mode':'ADD'}}}]


def test_batch_does_not_adopt_an_existing_manual_relationship(tmp_path):
    config,people=seeded(tmp_path);client=Batched();report=RunReport(False)
    with FaceDatabase(config.database) as state:
        bind(state,client,people);targets=ensure_people(state,client,report,'scope','1',people)
        first=people[0];client.links['1'].append(targets[first])
        publish_people(state,client,report,'scope','1',targets)
        payload=state.connection.execute('SELECT payload FROM publications WHERE person=?',(first,)).fetchone()[0]
        assert json.loads(payload)['managed'] is False
        assert client.batch_writes==[[targets[p] for p in people if p!=first]]
