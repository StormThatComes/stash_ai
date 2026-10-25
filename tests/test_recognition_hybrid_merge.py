from dataclasses import replace
import numpy as np

from tests.test_recognition_merge_report import seeded
from stash_ai.recognition.config import plain
from stash_ai.recognition.database import FaceDatabase
from stash_ai.recognition.person_signal import save,contract
from stash_ai.recognition.hybrid_merge import apply,rebuild,signature
from stash_ai.recognition.clustering import matching_policy


def prepared(tmp_path):
    config,people=seeded(tmp_path)
    data=plain(config.data);data['sapiensid']['enabled']=True
    config=replace(config,data=data)
    with FaceDatabase(config.database) as state:
        db=state.connection
        db.execute('UPDATE candidates SET policy=?',(matching_policy(config),))
        for row in db.execute('SELECT id FROM observations').fetchall():
            save(db,row['id'],contract(config),{'association':'unique'},np.ones(4096))
        db.commit()
    return config,people


def test_apply_preserves_oldest_and_downgrades_source(tmp_path):
    from stash_ai.recognition.reference_roles import strict_sql
    config,people=prepared(tmp_path)
    with FaceDatabase(config.database) as state:
        db=state.connection
        selected=[dict(scan=r['id'],scene=r['scene']) for r in db.execute('SELECT * FROM scans')]
        result=apply(state,selected,config)
        assert result['decisions'][0]['outcome']=='merge'
        assert set(r[0] for r in db.execute('SELECT person FROM members'))=={people[0]}
        assert db.execute('SELECT source,target,status FROM identity_merges').fetchone()[:]==(people[1],people[0],'pending')
        assert db.execute('SELECT COUNT(*) FROM observations o JOIN members m ON m.observation=o.id WHERE '+strict_sql()).fetchone()[0]==3
        assert len(apply(state,selected,config)['decisions'])==1
        assert db.execute('SELECT COUNT(*) FROM identity_merges').fetchone()[0]==1


def test_partial_scope_cannot_merge_external_members(tmp_path):
    config,people=prepared(tmp_path)
    with FaceDatabase(config.database) as state:
        scan=state.connection.execute('SELECT id,scene FROM scans LIMIT 1').fetchone()
        result=apply(state,[dict(scan=scan['id'],scene=scan['scene'])],config)
        assert result['decisions'][0]['blockers']==['identity_outside_selected_scans']
        assert state.connection.execute('SELECT COUNT(*) FROM identity_merges').fetchone()[0]==0


def test_warm_reuse_survives_publication_but_not_changed_vectors(tmp_path):
    config,people=prepared(tmp_path)
    with FaceDatabase(config.database) as state:
        db=state.connection;selected=[dict(scan=r['id'],scene=r['scene']) for r in db.execute('SELECT * FROM scans')]
        calls=[]
        rebuild(state,selected,config,lambda:calls.append(1),lambda:None)
        db.execute("UPDATE people SET performer='10000'");db.execute("UPDATE identity_merges SET status='confirmed'");db.commit()
        rebuild(state,selected,config,lambda:calls.append(1),lambda:None)
        assert calls==[1]
        db.execute("UPDATE person_evidence SET metadata='{}'");db.commit()
        rebuild(state,selected,config,lambda:calls.append(1),lambda:None)
        assert calls==[1,1]


def test_production_rebuild_and_repeat_keep_hybrid_assignment(tmp_path):
    from stash_ai.recognition.rebuild import rebuild_selected
    from stash_ai.recognition.config import _freeze
    config,people=prepared(tmp_path)
    config=replace(config,data=_freeze(config.data))
    with FaceDatabase(config.database) as state:
        db=state.connection
        selected=[dict(scan=r['id'],scene=r['scene']) for r in db.execute('SELECT * FROM scans')]
        rebuild_selected(state,selected,config)
        before=list(map(tuple,db.execute('SELECT * FROM members ORDER BY observation')))
        assert {r[1] for r in before}=={people[0]}
        assert db.execute('SELECT COUNT(*) FROM identity_merges').fetchone()[0]==1
        rebuild_selected(state,selected,config)
        assert list(map(tuple,db.execute('SELECT * FROM members ORDER BY observation')))==before
        assert db.execute('SELECT COUNT(*) FROM identity_merges').fetchone()[0]==1
