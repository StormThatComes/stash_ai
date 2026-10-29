"""Publication validates the actual decisions, including legacy journal debt."""
from dataclasses import replace
import pytest

from stash_ai.recognition.config import plain
from stash_ai.recognition.database import FaceDatabase
from stash_ai.recognition.publication_queue import plan,snapshot,adopt,current
from stash_ai.recognition.body_identity import rebuild
from tests.test_recognition_body_identity import configuration,scene
from tests.test_recognition_merge_report import seeded


def test_journal_inventory_cannot_relabel_old_face_decisions_as_current(tmp_path):
    config,_=seeded(tmp_path)
    with FaceDatabase(config.database) as state:
        db=state.connection;namespace=db.execute('SELECT namespace FROM people LIMIT 1').fetchone()[0]
        prior=plan(db,'scope',namespace,config)
        assert all(r['status']=='ready' for r in prior)
        changed=replace(config,data=plain(config.data))
        changed.data['recognition']['min_identity_support']=100
        assert all(r['status']=='blocked' for r in plan(db,'scope',namespace,changed))
        with pytest.raises(ValueError,match='Face decision policy'):
            adopt(db,'scope',namespace,changed,prior[0])
        assert current(db,'scope',namespace,prior[0]['scene']) is None


@pytest.mark.parametrize('queue',[True,False])
def test_collection_retune_blocks_both_queue_and_legacy_inventory(tmp_path,queue):
    config=configuration(tmp_path)
    with FaceDatabase(config.database) as state:
        db=state.connection;item,ids=scene(db,config);db.commit()
        rebuild(state,[item],config,{item['scan']:[]})
        if not queue:db.execute("DELETE FROM metadata WHERE key LIKE 'identity-publication:%'");db.commit()
        assert plan(db,'scope','ns',config)[0]['status']=='ready'
        config.data['sapiensid']['collection']['min_sharpness']=999.
        assert plan(db,'scope','ns',config)[0]['status']=='blocked'


@pytest.mark.parametrize('damage',['missing_decision','missing_cue','old_policy','foreign_contract'])
def test_partial_body_decisions_never_become_an_empty_publication(tmp_path,damage):
    config=configuration(tmp_path)
    with FaceDatabase(config.database) as state:
        db=state.connection;item,ids=scene(db,config);db.commit()
        rebuild(state,[item],config,{item['scan']:[]})
        assert len(snapshot(db,'scope','ns','1',config)[1])==1
        if damage=='missing_decision':db.execute('DELETE FROM body_members WHERE observation=?',(ids[0],))
        elif damage=='missing_cue':db.execute('DELETE FROM body_cues WHERE cue=0')
        elif damage=='old_policy':db.execute("UPDATE body_members SET policy='old'")
        else:db.execute("UPDATE body_observations SET contract='old'")
        db.commit()
        assert plan(db,'scope','ns',config)[0]['status']=='blocked'
        with pytest.raises(ValueError,match='person'):
            snapshot(db,'scope','ns','1',config)


def test_complete_empty_body_collection_can_publish_an_empty_scene(tmp_path):
    config=configuration(tmp_path)
    with FaceDatabase(config.database) as state:
        db=state.connection;item,ids=scene(db,config)
        db.execute('DELETE FROM body_observations');db.commit()
        rebuild(state,[item],config,{item['scan']:[]})
        result=plan(db,'scope','ns',config)[0]
        assert result['status']=='ready' and result['people']==[]
