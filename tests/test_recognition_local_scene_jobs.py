import json
import pytest
from tests.test_recognition_rebuild import library, changed, partition, run, two_people
from stash_ai.recognition.database import FaceDatabase


@pytest.mark.parametrize('backend',['threads','processes'])
def test_parallel_local_scenes_match_serial(tmp_path,monkeypatch,backend):
    monkeypatch.setattr('stash_ai.recognition.compute.available_cpus',lambda:4)
    outputs=[]
    for name,workers in [('serial',1),('parallel',2)]:
        cfg,_,services,_=library(tmp_path/name,monkeypatch,two_people())
        cfg=changed(cfg,clustering={'anchor_view_matching':True},execution={
            'rebuild_workers':workers,'rebuild_cohort_backend':backend,
            'rebuild_queue_bytes':64*1024**2,'profile_rebuild':True})
        assert run(cfg,services,'1','2')==0
        with FaceDatabase(cfg.database) as state:
            candidates=[tuple(r) for r in state.connection.execute(
                "SELECT s.scene,o.cue,c.status FROM candidates c JOIN observations o ON "
                "json_extract(c.payload,'$.observation')=o.id JOIN scans s ON s.id=o.scan ORDER BY s.scene,o.cue")]
        outputs.append((partition(cfg),candidates))
    assert outputs[0]==outputs[1]


def test_reference_scope_defers_allocation(monkeypatch):
    import sqlite3
    from contextlib import closing
    from stash_ai.recognition import reference_index as index
    calls=[]
    class Fake:
        def __init__(self,*args):calls.append('build');self.enabled=True
        def close(self):calls.append('close')
    monkeypatch.setattr(index,'ReferenceIndex',Fake)
    with closing(sqlite3.connect(':memory:')) as db:
        db.execute('CREATE TABLE candidates(payload TEXT,scan TEXT,status TEXT)')
        with index.references(db,'scope','namespace',None):assert calls==[]
        with index.references(db,'scope','namespace',None):
            assert index.current(db) is index.current(db)
    assert calls==['build','close']


def test_packed_fields_keep_non_uuid_ids_and_owner_updates():
    from stash_ai.recognition.packed_ids import Order,Ranks,Flags,Values
    ids=['705571f6-d2a3-4a55-a626-c2cd63fc9320','custom','705571F6-D2A3-4A55-A626-C2CD63FC9320']
    order=Order();ranks=Ranks()
    for rank,value in enumerate(ids):ranks[value]=rank;order.append(value)
    flags=Flags(ranks,3);owners=Values(ranks,3)
    for value in ids:flags.add(value);owners[value]='one'
    assert [order[i] for i in range(3)]==ids
    assert len(order.data)==48 and len(flags.bits)==3 and owners.ids.itemsize==4
    owners[ids[0]]='two';assert owners.pop(ids[0])=='two'
    assert owners.get(ids[0]) is None and owners.get('missing') is None
    assert all(value in flags for value in ids)
