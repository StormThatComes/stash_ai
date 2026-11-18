from contextlib import ExitStack
from types import SimpleNamespace
import numpy as np
import pytest


def test_disk_tree_preserves_all_ram_tree_pairs_and_pruning():
    from stash_ai.recognition.centroid_tree import Tree
    rng=np.random.default_rng(7)
    values=rng.normal(size=(20,4096));values/=np.linalg.norm(values,axis=1)[:,None]
    with ExitStack() as stack:
        search=SimpleNamespace(stack=stack,entries=list(range(20)),block=2,rounding=.001,stats={})
        ram=Tree(search,lambda i:values[i:i+2])
        disk=Tree(search,lambda i:values[i:i+2],paged=True)
        active=np.arange(20)%3==0
        for floor in (None,.2,.99):
            assert list(disk.pairs(active,floor))==list(ram.pairs(active,floor))
        assert search.stats['centroid_tree_disk_bytes']>0
        assert len(disk.nodes.cache)<=4


def test_pair_pages_deduplicate_reverse_without_retaining_python_set():
    from stash_ai.recognition.paged_pairs import Pairs
    with Pairs(2**20) as pairs:
        for i in range(2000):
            assert pairs.add(('root',str(i)))
            assert not pairs.add((str(i),'root'))
        assert pairs.count==2000
        assert len(list(pairs))==2000


def test_shared_host_budget_accounts_for_other_pools_and_native_rss(monkeypatch):
    from stash_ai.recognition import host_memory
    from stash_ai.recognition.resource_errors import ResourceLimit
    monkeypatch.setattr(host_memory,'process_rss',lambda:100)
    budget=host_memory.Budget(1000)
    assert budget.replace(('retained',1),400)
    assert not budget.replace(('rebuild',1),501)
    assert budget.replace(('rebuild',1),500)
    assert budget.available()==0
    monkeypatch.setattr(host_memory,'process_rss',lambda:1200)
    assert budget.replace(('retained',1),0)
    with pytest.raises(ResourceLimit):budget.replace(('queue',1),1,required=True)


def test_empty_capacity_report_does_not_invent_payload_estimate(tmp_path):
    from stash_ai.recognition.database import FaceDatabase
    from stash_ai.recognition.config import defaults
    from stash_ai.recognition.capacity import estimate
    with FaceDatabase(tmp_path/'faces.db') as database:
        report=estimate(database.connection,SimpleNamespace(data=defaults()))
    assert report['target_frames']==3000000
    assert report['components']['faces']['target_rows']==4200000
    assert report['components']['bodies']['target_rows']==4200000
    assert report['projected_current_evidence_bytes'] is None


def test_capacity_measures_cold_crops_and_compressed_masks_in_selected_namespace():
    import sqlite3
    from stash_ai.recognition.capacity import estimate
    from stash_ai.recognition.config import defaults
    from stash_ai.recognition.models import namespace
    config=SimpleNamespace(data=defaults());model=namespace(config)
    with sqlite3.connect(':memory:') as db:
        db.executescript('''CREATE TABLE metadata(key TEXT,value TEXT);
            CREATE TABLE scans(id TEXT,scope TEXT,namespace TEXT,scene TEXT,status TEXT);
            CREATE TABLE observations(id TEXT,scan TEXT,cue INTEGER,embedding BLOB,crop BLOB,geometry TEXT,quality TEXT);
            CREATE TABLE body_observations(id TEXT,scan TEXT,cue INTEGER,embedding BLOB,crop BLOB,geometry TEXT,quality TEXT);
            CREATE TABLE crop_payloads(id BLOB,data BLOB);
            CREATE TABLE body_quality_masks(observation TEXT,mask BLOB);''')
        db.execute('INSERT INTO scans VALUES (?,?,?,?,?)',('s','scope',model,'1','complete'))
        db.execute('INSERT INTO scans VALUES (?,?,?,?,?)',('other','scope','foreign-model','2','complete'))
        address=b'crop:'+b'a'*64
        db.execute('INSERT INTO crop_payloads VALUES (?,?)',(address,b'x'*7000))
        db.execute('INSERT INTO observations VALUES (?,?,?,?,?,?,?)',('f','s',0,b'x'*2048,address,'[]','{}'))
        db.execute('INSERT INTO observations VALUES (?,?,?,?,?,?,?)',('wrong','other',0,b'x'*32,None,'[]','{}'))
        db.execute('INSERT INTO body_observations VALUES (?,?,?,?,?,?,?)',('b','s',0,b'x'*16384,None,'[]','{}'))
        db.execute('INSERT INTO body_quality_masks VALUES (?,?)',('b',b'x'*123))
        result=estimate(db,config,scenes=1,cues=1,positive_fraction=1,faces=2,bodies=2,overhead=1)
    assert result['sampled_scenes']==['1']
    assert result['components']['faces']['mean_payload_bytes']['crop']==7000
    assert result['components']['bodies']['mean_payload_bytes']['mask']==123
    assert result['projected_current_evidence_bytes']==2*(2048+7000+4+16384+123+4)
