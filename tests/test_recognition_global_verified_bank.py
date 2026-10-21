"""Durable direct confirmations train global views; recovered views never do."""
import math
import pytest
import numpy as np
from stash_ai.recognition.database import FaceDatabase
from stash_ai.recognition.rebuild import rebuild_selected
from stash_ai.recognition.historical_references import capture,context
from tests.test_recognition_anchor_views import config_for,A,C,X
from tests.test_recognition_global_resolution import source
from tests.test_recognition_clustering import scan
from tests.test_recognition_scoped_global_identities import snapshot,selected
from tests.test_recognition_scene_context import candidates
from tests.test_recognition_rebuild import changed


@pytest.mark.parametrize('auxiliary',[False,True])
def test_confirmed_auxiliary_view_is_available_in_later_selected_run(tmp_path,auxiliary):
    config=config_for(tmp_path,view_prototypes_enabled=True)
    with FaceDatabase(config.database) as db:
        first=source(db,config,'1');rebuild_selected(db,[first],config)
        owner=candidates(db,first['scan'],config)[0][1]
        before=snapshot(db,first['scan'])
        sid,_=scan(db,config,'2',[C]*3,count=3)
        if auxiliary:
            db.connection.execute("UPDATE observations SET quality=json_set(quality,'$.pose_deviation',.3) WHERE scan=?",(sid,))
            db.connection.commit()
        selected_scenes=[{'scene':'2','scan':sid}]
        rebuild_selected(db,selected_scenes,config)
        rows=candidates(db,sid,config)
        assert all(r[0]=='accepted' and r[1]==owner and r[2]['reference_role']=='attachment' for r in rows)
        assert snapshot(db,first['scan'])==before
        if not auxiliary:
            assert all(r[2]['historical_reference']['view_evidence'][0]['authority']=='verified_view' for r in rows)
        rebuild_selected(db,selected_scenes,config)
        assert candidates(db,sid,config)==rows


def test_removed_direct_support_invalidates_global_view_context(tmp_path):
    config=config_for(tmp_path,view_prototypes_enabled=True)
    with FaceDatabase(config.database) as db:
        first=source(db,config,'1');rebuild_selected(db,[first],config)
        before=context(db.connection,[],config,'scope','ns')
        assert any(v.authority=='verified_view' for v in capture(db.connection,[],config,'scope','ns',before).views)
        db.connection.execute('DELETE FROM members WHERE observation IN (SELECT id FROM observations WHERE scan=? AND cue=5)',(first['scan'],))
        db.connection.commit()
        after=context(db.connection,[],config,'scope','ns')
        assert after!=before
        assert not any(v.authority=='verified_view' for v in capture(db.connection,[],config,'scope','ns',after).views)


def test_historical_recovered_view_cannot_promote_another_view(tmp_path):
    config=config_for(tmp_path,view_prototypes_enabled=True)
    beyond=np.array([.1,math.sqrt(1-.1**2),0.,0.])
    with FaceDatabase(config.database) as db:
        first=source(db,config,'1');rebuild_selected(db,[first],config)
        owner=candidates(db,first['scan'],config)[0][1]
        second=selected(db,config,'2',[C]*3)
        assert all(r[1]==owner for r in candidates(db,second,config))
        third=selected(db,config,'3',[beyond]*3)
        assert all(r[1]!=owner for r in candidates(db,third,config))


@pytest.mark.parametrize('change,eligible',[
    ({'min_face_pixels':64},True),
    ({'min_face_pixels':512},False),
    ({'min_visible_face_fraction':.8},False),
])
def test_cross_policy_references_require_current_quality_measurements(tmp_path,change,eligible):
    config=config_for(tmp_path)
    with FaceDatabase(config.database) as db:
        selected(db,config,'1',[A]*3)
        tuned=changed(config,detection=change)
        history=capture(db.connection,[],tuned,'scope','ns','test')
        assert bool(history.groups)==eligible


def test_cross_policy_evidence_with_old_quality_contract_is_excluded(tmp_path):
    config=config_for(tmp_path)
    with FaceDatabase(config.database) as db:
        sid=selected(db,config,'1',[A]*3)
        db.connection.execute("UPDATE observations SET quality=json_remove(quality,'$.pose_contract') WHERE scan=?",(sid,))
        db.connection.commit()
        tuned=changed(config,detection={'min_face_pixels':64})
        assert not capture(db.connection,[],tuned,'scope','ns','test').groups
        assert not capture(db.connection,[],config,'scope','different-namespace','test').groups


def test_competing_strict_identity_vetoes_historical_auxiliary_promotion(tmp_path):
    config=config_for(tmp_path,view_prototypes_enabled=True)
    with FaceDatabase(config.database) as db:
        first=source(db,config,'1');rebuild_selected(db,[first],config)
        # A separate strict C identity competes strongly for X. A stored old
        # auxiliary decision alone cannot override today's frozen competition.
        legacy=changed(config,clustering={'view_prototypes_enabled':False})
        selected(db,legacy,'2',[C]*3)
        bank=capture(db.connection,[],config,'scope','ns','test')
        assert not any(v.authority=='verified_view' for v in bank.views)
