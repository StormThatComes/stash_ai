import json
from types import SimpleNamespace
from stash_ai.recognition.database import FaceDatabase
from tests.test_recognition_body_identity import configuration,scene
from tests.test_recognition_merge_report import seeded


def inventory(p):
    from stash_ai.recognition.body_identity import rebuild
    from stash_ai.recognition.publication_queue import inventory
    c=configuration(p)
    with FaceDatabase(c.database) as state:
        db=state.connection;a,_=scene(db,c);db.commit();rebuild(state,[a],c,{a['scan']:[]})
        db.execute("UPDATE people SET performer='123'");db.commit()
        return inventory(db,'scope','ns')

def publication_race(p,after_add=False):
    from collections import Counter
    from tests.test_recognition_publication_batch import Batched,bind
    from stash_ai.recognition.publication_queue import enqueue,snapshot,current
    from stash_ai.recognition.run import _publish_selection
    from stash_ai.runtime.report import RunReport
    c,people=seeded(p);client=Batched()
    with FaceDatabase(c.database) as state:
        db=state.connection;bind(state,client,people)
        ns=db.execute('SELECT namespace FROM people LIMIT 1').fetchone()[0]
        row=db.execute("SELECT * FROM scans WHERE scene='1'").fetchone();selected=[dict(scan=row['id'],scene='1')]
        with db:enqueue(db,'scope',ns,'1',c,'initial')
        old=client.performers;calls=[]
        def change(ids):
            result=old(ids)
            if not calls:
                calls.append(1)
                with db:
                    db.execute('UPDATE members SET person=? WHERE person=?',(people[1],people[0]))
                    db.execute('UPDATE candidates SET person=? WHERE person=?',(people[1],people[0]))
                    enqueue(db,'scope',ns,'1',c,'concurrent_rebuild')
            return result
        if after_add:
            add=client.add_scene_people
            def changed_add(scene,ids):
                result=add(scene,ids);change(ids);return result
            client.add_scene_people=changed_add
        else:client.performers=change
        counts=Counter()
        state.progress=lambda *args,**kwargs:None
        _publish_selection(state,client,RunReport(False),c,'scope',ns,selected,{row['id']:[dict(person=people[0],status='accepted',observations=3)]},{},counts,'','test',SimpleNamespace(advance=lambda **kw:None))
        return dict(desired=snapshot(db,'scope',ns,'1',c)[1],stale_person=people[0],published=client.links['1'],writes=client.batch_writes,counts=dict(counts),queued=current(db,'scope',ns,'1') is not None)

def preflight_unknown(p):
    from tests.test_recognition_publication_batch import Batched,bind
    from stash_ai.recognition.publication_batch import publish_people
    from stash_ai.recognition.scene_sync import remove_stale_links,reconcile_unlinks
    from stash_ai.runtime.report import RunReport
    c,people=seeded(p);client=Batched();report=RunReport(False)
    with FaceDatabase(c.database) as state:
        db=state.connection;bind(state,client,people)
        ns=db.execute('SELECT namespace FROM people LIMIT 1').fetchone()[0]
        publish_people(state,client,report,'scope','1',{people[0]:'100'})
        original=client.performer;calls=[]
        def transient(identifier):
            value=original(identifier);calls.append(identifier)
            if len(calls)==2:value['details']='changed just before dispatch'
            return value
        client.performer=transient;errors=[]
        try:remove_stale_links(state,client,report,'scope','1',set(),ns)
        except Exception as e:errors.append(str(e))
        client.performer=original
        try:reconcile_unlinks(state,client,'scope','1')
        except Exception as e:errors.append(str(e))
        statuses=[tuple(r) for r in db.execute("SELECT kind,status FROM publications WHERE kind='unlink'")]
        removed=remove_stale_links(state,client,report,'scope','1',set(),ns)
        return dict(errors=errors,unlink_status=statuses,removed=removed,links=client.links['1'])


def test_normal_run_stops_before_stale_add(tmp_path):
    result=publication_race(tmp_path)
    assert result['writes']==[]
    assert result['published']==['77']
    assert result['counts'].get('scenes_processed',0)==0
    assert result['queued']


def test_normal_run_keeps_debt_if_assignments_change_after_add(tmp_path):
    result=publication_race(tmp_path,after_add=True)
    assert result['writes']==[['100']]
    assert result['counts'].get('scenes_processed',0)==0
    assert result['counts'].get('scene_links_removed',0)==0
    assert result['queued']


def test_preflight_rejection_is_retryable_without_uncertain_journal(tmp_path):
    result=preflight_unknown(tmp_path)
    assert len(result['errors'])==1
    assert result['unlink_status']==[('unlink','failed')]
    assert result['removed']==1
    assert result['links']==['77']


def test_inventory_includes_published_body_only_people(tmp_path):
    result=inventory(tmp_path)
    assert result['identities_with_latest_memberships']==1
    assert result['retained_profiles_without_latest_memberships']==[]


def test_create_guard_rejection_is_failed_before_dispatch(tmp_path):
    import pytest
    from tests.test_recognition_publication_batch import Batched,bind
    from stash_ai.recognition.publication import ensure_person,PendingRecognition
    from stash_ai.runtime.report import RunReport
    c,people=seeded(tmp_path);client=Batched();calls=[]
    with FaceDatabase(c.database) as state:
        bind(state,client,people);person=people[0]
        state.connection.execute('UPDATE people SET performer=NULL WHERE id=?',(person,));state.connection.commit()
        client.profiles.pop('100')
        def guard():
            calls.append(1)
            if len(calls)==2:raise PendingRecognition('changed before create')
        with pytest.raises(PendingRecognition,match='changed before create'):
            ensure_person(state,client,RunReport(False),'scope','1',person,guard=guard)
        assert state.connection.execute("SELECT status FROM publications WHERE kind='create'").fetchone()[0]=='failed'
        assert not client.writes
        assert ensure_person(state,client,RunReport(False),'scope','1',person)


def test_inventory_excludes_provisional_and_disabled_bodies(tmp_path):
    from stash_ai.recognition.body_identity import rebuild
    from stash_ai.recognition.publication_queue import inventory
    c=configuration(tmp_path)
    with FaceDatabase(c.database) as state:
        db=state.connection;a,_=scene(db,c,'1');b,_=scene(db,c,'2',vectors=[[1,0]]*2)
        db.commit();rebuild(state,[a,b],c,{a['scan']:[],b['scan']:[]})
        db.execute("UPDATE people SET performer='123'");db.commit()
        result=inventory(db,'scope','ns',c)
        assert result['identities_with_latest_memberships']==1
        assert len(result['retained_profiles_without_latest_memberships'])==1
        c.data['sapiensid']['identities']['enabled']=False
        result=inventory(db,'scope','ns',c)
        assert result['identities_with_latest_memberships']==0
        assert len(result['retained_profiles_without_latest_memberships'])==2
