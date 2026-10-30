"""Fast certificates retain mutation protection and ignore unrelated bookkeeping."""
import sqlite3
import uuid
import pytest
from stash_ai.recognition.database import FaceDatabase, encoded
from stash_ai.recognition.publication_snapshot import Snapshot
from stash_ai.recognition.publication_queue import current, validate, body_policy
from stash_ai.recognition.clustering import matching_policy
from stash_ai.recognition.publication import PendingRecognition
from stash_ai.recognition.replay_signature import fingerprint
from tests.test_recognition_merge_report import seeded


def capture(db,config,namespace):
    reader=Snapshot(db,'scope',namespace,'1',config)
    item,people,signature=reader.read()
    entry=dict(item,people=people,signature=signature,revision=(current(db,'scope',namespace,'1') or {}).get('revision'),
               policy=matching_policy(config),body=body_policy(config))
    return reader,entry


def test_publication_reuses_vectors_across_owned_bookkeeping(tmp_path,monkeypatch):
    config,people=seeded(tmp_path)
    import stash_ai.recognition.publication_queue as queue
    calls=[];original=queue.snapshot
    def measured(*a,**kw):calls.append(1);return original(*a,**kw)
    monkeypatch.setattr(queue,'snapshot',measured)
    with FaceDatabase(config.database) as state:
        db=state.connection;ns=db.execute('SELECT namespace FROM people LIMIT 1').fetchone()[0]
        reader,entry=capture(db,config,ns)
        for i in range(20):
            with db:
                db.execute('UPDATE people SET performer=? WHERE id=?',(str(i),people[0]))
                db.execute('INSERT OR REPLACE INTO metadata VALUES (?,?)',('unrelated-progress',str(i)))
            validate(db,'scope',ns,entry,config,snapshot_reader=reader)
        assert len(calls)==1


@pytest.mark.parametrize('change',['vector','token','rollback','external','trigger'])
def test_publication_certificate_detects_changes_without_rebasing_after_rollback(tmp_path,change):
    config,people=seeded(tmp_path)
    with FaceDatabase(config.database) as state:
        db=state.connection;ns=db.execute('SELECT namespace FROM people LIMIT 1').fetchone()[0]
        reader,entry=capture(db,config,ns)
        sql="UPDATE observations SET geometry='[0,0,3,3]' WHERE scan=?"
        args=(entry['scan'],)
        if change=='vector':sql='UPDATE observations SET embedding=zeroblob(length(embedding)) WHERE scan=?'
        if change=='token':sql="UPDATE people SET token=token||'-changed' WHERE id=?";args=(people[0],)
        if change=='external':
            with sqlite3.connect(config.database) as other:other.execute(sql,args)
        elif change=='trigger':db.execute('DROP TRIGGER recognition_version_observations_update')
        elif change=='rollback':
            db.execute(sql,args)
            assert reader.read()[2]!=entry['signature']
            db.rollback()
            validate(db,'scope',ns,entry,config,snapshot_reader=reader)
            db.execute("UPDATE observations SET geometry='[0,0,4,4]' WHERE scan=?",args)
        else:db.execute(sql,args)
        with pytest.raises((PendingRecognition,ValueError)):
            validate(db,'scope',ns,entry,config,snapshot_reader=reader)


def test_body_replace_moving_id_invalidates_old_scene_even_without_delete_trigger(tmp_path):
    from tests.test_recognition_body_identity import configuration,scene
    from stash_ai.recognition.body_identity import rebuild
    config=configuration(tmp_path)
    with FaceDatabase(config.database) as state:
        db=state.connection;first,ids=scene(db,config,'1');second,_=scene(db,config,'2')
        db.commit();rebuild(state,[first,second],config,{first['scan']:[],second['scan']:[]})
        reader,entry=capture(db,config,'ns')
        row=list(db.execute('SELECT * FROM body_observations WHERE id=?',(ids[0],)).fetchone());row[1]=second['scan']
        db.execute('INSERT OR REPLACE INTO body_observations VALUES (?,?,?,?,?,?,?,?,?,?,?)',row)
        with pytest.raises((PendingRecognition,ValueError)):
            validate(db,'scope','ns',entry,config,snapshot_reader=reader)


def test_checkpoint_ignores_crops_logs_and_foreign_scope(tmp_path):
    config,_=seeded(tmp_path)
    with FaceDatabase(config.database) as state:
        db=state.connection;ns=db.execute('SELECT namespace FROM scans LIMIT 1').fetchone()[0]
        before=fingerprint(db,'scope',ns)
        db.execute('UPDATE observations SET crop=?',(b'jpeg-not-replay-input'*1000,))
        db.execute("INSERT INTO metadata VALUES ('run-progress','anything')")
        row=list(db.execute('SELECT * FROM scans LIMIT 1').fetchone());row[0]=str(uuid.uuid4());row[1]='foreign'
        db.execute('INSERT INTO scans VALUES (?,?,?,?,?,?,?,?,?,?,?)',row)
        obs=list(db.execute('SELECT * FROM observations LIMIT 1').fetchone());obs[0]=str(uuid.uuid4());obs[1]=row[0]
        db.execute('INSERT INTO observations VALUES (?,?,?,?,?,?,?,?,?,?,?,?)',obs)
        assert fingerprint(db,'scope',ns)==before
        db.execute("UPDATE observations SET quality='{}' WHERE id=?",(obs[0],))
        assert fingerprint(db,'scope',ns)==before
        db.execute("UPDATE observations SET geometry='[1,2,3,4]' WHERE id!=?",(obs[0],))
        assert fingerprint(db,'scope',ns)!=before


@pytest.mark.parametrize('table,column,value',[
    ('body_observations','quality','{}'),('body_observations','embedding',bytes(16384)),
    ('body_members','role','auxiliary'),('scans','status','partial'),('people','token','replacement')])
def test_checkpoint_includes_body_evidence_roles_and_bindings(tmp_path,table,column,value):
    from tests.test_recognition_body_identity import configuration,scene
    from stash_ai.recognition.body_identity import rebuild
    config=configuration(tmp_path)
    with FaceDatabase(config.database) as state:
        db=state.connection;item,_=scene(db,config);db.commit();rebuild(state,[item],config,{item['scan']:[]})
        before=fingerprint(db,'scope','ns')
        db.execute(f'UPDATE {table} SET {column}=?',(value,))
        assert fingerprint(db,'scope','ns')!=before


def test_checkpoint_includes_aliases_and_explicit_scan_heads(tmp_path):
    config,people=seeded(tmp_path)
    with FaceDatabase(config.database) as state:
        db=state.connection;ns=db.execute('SELECT namespace FROM scans LIMIT 1').fetchone()[0]
        before=fingerprint(db,'scope',ns)
        db.execute('INSERT INTO identity_merges VALUES (?,?,?,?,?)',(people[0],people[1],1.,'{}','local'))
        assert fingerprint(db,'scope',ns)!=before
        db.rollback()
        scan=db.execute('SELECT id FROM scans LIMIT 1').fetchone()[0]
        db.execute('INSERT INTO metadata VALUES (?,?)',('recognition-scan-head:test',encoded(dict(scan=scan,latest_created_scan=scan))))
        assert fingerprint(db,'scope',ns)!=before
