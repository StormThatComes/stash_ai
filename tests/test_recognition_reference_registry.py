"""Global reference repair and coverage gaps must not manufacture duplicates."""
import math,uuid
import numpy as np
import pytest

from stash_ai.recognition.database import FaceDatabase
from stash_ai.recognition.historical_references import capture,context
from stash_ai.recognition.rebuild import rebuild_selected
from tests.test_recognition_anchor_views import config_for,A,X,C,RIVAL
from tests.test_recognition_frozen_partition import fixture
from tests.test_recognition_global_views import seed
from tests.test_recognition_scene_context import candidates
from tests.test_recognition_rebuild import changed


def pending(db,item,status='partial'):
    columns=[r[1] for r in db.connection.execute('PRAGMA table_info(scans)')]
    row=dict(db.connection.execute('SELECT * FROM scans WHERE id=?',(item['scan'],)).fetchone())
    row.update(id=str(uuid.uuid4()),input_sha=str(uuid.uuid4()),status=status,cursor=0)
    db.connection.execute('INSERT INTO scans ('+','.join(columns)+') VALUES ('+','.join('?' for _ in columns)+')',
        [row[c] for c in columns]);db.connection.commit()
    return row['id']


@pytest.mark.parametrize('status',['partial','complete'])
def test_unavailable_reference_defers_only_possible_duplicate_until_restored(tmp_path,status):
    config=config_for(tmp_path)
    with FaceDatabase(config.database) as db:
        first=fixture(db,config,'1',[A]*3);rebuild_selected(db,[first],config)
        owner=candidates(db,first['scan'],config)[0][1]
        replacement=pending(db,first,status)
        before=context(db.connection,[],config,'scope','ns')
        second=fixture(db,config,'2',[A]*3);rebuild_selected(db,[second],config)
        rows=candidates(db,second['scan'],config)
        assert all(r[0]=='withheld' and r[2]['reason']=='reference_bank_unavailable' for r in rows)
        unrelated=fixture(db,config,'3',[RIVAL]*3);rebuild_selected(db,[unrelated],config)
        assert all(r[0]=='accepted' and r[1]!=owner for r in candidates(db,unrelated['scan'],config))
        db.connection.execute('DELETE FROM scans WHERE id=?',(replacement,));db.connection.commit()
        assert context(db.connection,[],config,'scope','ns')!=before
        rebuild_selected(db,[second],config)
        assert all(r[0]=='accepted' and r[1]==owner for r in candidates(db,second['scan'],config))


def angle(degrees):
    value=math.radians(degrees)
    return np.array([math.cos(value),math.sin(value),0.,0.])


def test_old_fragments_share_canonical_label_and_preserve_original_ownership(tmp_path):
    config=config_for(tmp_path)
    with FaceDatabase(config.database) as db:
        first,a=seed(db,config,'1',A);second,b=seed(db,config,'2',angle(12))
        assert a!=b
        db.connection.execute('UPDATE people SET performer=? WHERE id=?',('101',a))
        db.connection.execute('UPDATE people SET performer=? WHERE id=?',('102',b));db.connection.commit()
        labels=[tuple(r) for r in db.connection.execute('SELECT * FROM people ORDER BY id')]
        observations=[tuple(r) for r in db.connection.execute('SELECT * FROM observations ORDER BY id')]
        history=capture(db.connection,[],config,'scope','ns','report')
        assert {g.identifier for g in history.groups}=={a}
        query=fixture(db,config,'3',[angle(6)]*3);rebuild_selected(db,[query],config)
        assert all(r[0]=='accepted' and r[1]==a for r in candidates(db,query['scan'],config))
        assert {r[0] for r in db.connection.execute('SELECT DISTINCT person FROM members')}=={a}
        assert [tuple(r) for r in db.connection.execute('SELECT * FROM people ORDER BY id')]==labels
        old=[tuple(r) for r in db.connection.execute('SELECT * FROM observations WHERE scan IN (?,?) ORDER BY id',(first,second))]
        assert old==observations
        assert db.connection.execute('SELECT 1 FROM identity_origins WHERE person=?',(b,)).fetchone()
        rows=candidates(db,query['scan'],config)
        rebuild_selected(db,[query],config)
        assert candidates(db,query['scan'],config)==rows


def test_registry_does_not_collapse_strong_chain_with_incompatible_endpoints(tmp_path):
    config=config_for(tmp_path)
    with FaceDatabase(config.database) as db:
        for i,degrees in enumerate((0,30,60),1):seed(db,config,str(i),angle(degrees))
        history=capture(db.connection,[],config,'scope','ns','chain')
        assert len({g.identifier for g in history.groups})==3
        assert not history.actions


@pytest.mark.parametrize('auxiliary',[False,True])
@pytest.mark.parametrize('source_first',[False,True])
@pytest.mark.parametrize('count',[2,4])
def test_reverse_history_reconciliation_keeps_whole_independent_source_core(tmp_path,auxiliary,source_first,count):
    config=changed(config_for(tmp_path,view_prototypes_enabled=True),recognition={'min_identity_support':2})
    with FaceDatabase(config.database) as db:
        source=fixture(db,config,'2',[X]*3+[A]*3,range(3)) if auxiliary else fixture(db,config,'2',[A]*3+[X]*3)
        query=fixture(db,config,'1',[C]*count)
        for item in ([source,query] if source_first else [query,source]):rebuild_selected(db,[item],config)
        rows=[r for item in (source,query) for r in candidates(db,item['scan'],config)]
        assert all(r[0]=='accepted' for r in rows)
        assert len({r[1] for r in rows})==1
        if auxiliary and not source_first:
            recovered=candidates(db,query['scan'],config)
            assert all(r[2]['reason']=='global_reference_attachment' for r in recovered)
            later=fixture(db,config,'3',[angle(84)]*3);rebuild_selected(db,[later],config)
            assert all(r[1]!=rows[0][1] for r in candidates(db,later['scan'],config))


def test_historical_copresence_still_requires_reflection_strength(tmp_path):
    config=config_for(tmp_path)
    with FaceDatabase(config.database) as db:
        first,a=seed(db,config,'1',A);second,b=seed(db,config,'2',angle(32))
        observation=db.connection.execute('SELECT id FROM observations WHERE scan=? LIMIT 1',(second,)).fetchone()[0]
        db.connection.execute('UPDATE observations SET scan=?,cue=0,seconds=0 WHERE id=?',(first,observation))
        db.connection.execute("UPDATE candidates SET scan=? WHERE json_extract(payload,'$.observation')=?",(first,observation))
        db.connection.commit()
        history=capture(db.connection,[],config,'scope','ns','copresent')
        assert {g.identifier for g in history.groups}=={a,b}
        assert not history.actions
