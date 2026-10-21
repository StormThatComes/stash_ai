"""Visibility is a shared admission rule for live and retained evidence."""
import json
import pytest

from stash_ai.recognition.config import load,validate,plain
from stash_ai.recognition.database import FaceDatabase
from stash_ai.recognition.pose_policy import is_anchor,anchor_sql,auxiliary_sql,compact_quality
from stash_ai.recognition.visibility import CONTRACT_ID
from stash_ai.recognition.scanning import scan_identity
from stash_ai.recognition.clustering import matching_policy,_groups
from stash_ai.recognition.historical_references import capture,context
from stash_ai.recognition.rebuild import rebuild_selected
from tests.test_recognition_anchor_views import config_for,A
from tests.test_recognition_frozen_partition import fixture
from tests.test_recognition_rebuild import changed
from tests.test_recognition_scene_context import candidates


def configured(tmp_path):
    return changed(config_for(tmp_path),clustering={
        'min_anchor_visible_face_fraction':.8,'min_auxiliary_visible_face_fraction':.5})


def measured(db,config,scene,fractions):
    # Retained measurements can be requalified without another inference.
    scan_config=changed(config,clustering={'min_anchor_visible_face_fraction':None,'min_auxiliary_visible_face_fraction':None})
    item=fixture(db,scan_config,scene,[A]*len(fractions))
    for cue,fraction in enumerate(fractions):
        db.connection.execute("UPDATE observations SET quality=json_set(quality,'$.visible_face_fraction',?,"
            "'$.visibility',json(?)) WHERE scan=? AND cue=?",
            (fraction,json.dumps({'contract':CONTRACT_ID}),item['scan'],cue))
    db.connection.commit();return item


@pytest.mark.parametrize('pose',[None,.2])
def test_front_occluded_faces_attach_but_do_not_train_or_seed(tmp_path,pose):
    config=changed(configured(tmp_path),clustering={'max_anchor_pose_deviation':pose})
    with FaceDatabase(config.database) as db:
        item=measured(db,config,'1',[.9]*3+[.65]*3+[.4])
        rebuild_selected(db,[item],config)
        rows=candidates(db,item['scan'],config)
        assert all(r[0]=='accepted' and r[2]['reference_role']=='strict' for r in rows[:3])
        assert all(r[0]=='accepted' and r[1]==rows[0][1] and r[2]['reason']=='auxiliary_identity'
                   and r[2]['reference_role']=='attachment' for r in rows[3:6])
        assert rows[-1][0]=='withheld' and rows[-1][2]['reason']=='auxiliary_visibility_insufficient'
        assert len(_groups(db.connection,'scope','ns',config.data['clustering'])[0].samples)==3
        profiles=measured(db,config,'2',[.65]*3);rebuild_selected(db,[profiles],config)
        assert all(r[0]=='accepted' and r[1]==rows[0][1] for r in candidates(db,profiles['scan'],config))
    with FaceDatabase(tmp_path/'only_profiles.sqlite') as db:
        item=measured(db,config,'3',[.65]*3);rebuild_selected(db,[item],config)
        assert not db.connection.execute('SELECT 1 FROM members').fetchone()


@pytest.mark.parametrize('value',[None,True,'0.9',-.1,1.1,.49,.5,.79,.8,1.])
def test_python_sql_and_compact_replay_have_identical_role_admission(tmp_path,value):
    config=configured(tmp_path);grouping=config.data['clustering']
    quality=json.dumps({'pose_deviation':.1,'visible_face_fraction':value,'visibility':{'contract':CONTRACT_ID}})
    expected=type(value) in (int,float) and .8<=value<=1
    with FaceDatabase(config.database) as db:
        for raw in (quality,compact_quality(quality)):
            assert is_anchor(raw,grouping)==expected
            assert bool(db.connection.execute('SELECT '+anchor_sql('o',grouping)+' FROM (SELECT ? AS quality) o',(raw,)).fetchone()[0])==expected
            auxiliary=bool(db.connection.execute('SELECT '+auxiliary_sql('o',grouping)+' FROM (SELECT ? AS quality) o',(raw,)).fetchone()[0])
            assert auxiliary==(type(value) in (int,float) and .5<=value<.8)


def test_retuning_role_visibility_reuses_scan_and_invalidates_matching_and_dependencies(tmp_path):
    config=configured(tmp_path)
    tuned=changed(config,clustering={'min_anchor_visible_face_fraction':.9})
    assert scan_identity(config,'ns')==scan_identity(tuned,'ns')
    assert matching_policy(config)!=matching_policy(tuned)
    with FaceDatabase(config.database) as db:
        item=measured(db,config,'1',[.85]*3);rebuild_selected(db,[item],config)
        before=context(db.connection,[],config,'scope','ns')
        db.connection.execute("UPDATE observations SET quality=json_set(quality,'$.visible_face_fraction',.7) WHERE scan=?",(item['scan'],));db.connection.commit()
        assert before!=context(db.connection,[],config,'scope','ns')
        assert not capture(db.connection,[],config,'scope','ns','changed').groups
        rebuild_selected(db,[item],config)
        assert all(r[0]=='withheld' for r in candidates(db,item['scan'],config))


def test_visibility_contract_is_required_and_old_measurement_cannot_seed(tmp_path):
    config=configured(tmp_path)
    with FaceDatabase(config.database) as db:
        item=measured(db,config,'1',[.9]*3)
        db.connection.execute("UPDATE observations SET quality=json_set(quality,'$.visibility.contract','obsolete')");db.connection.commit()
        rebuild_selected(db,[item],config)
        assert all(r[0]=='withheld' for r in candidates(db,item['scan'],config))


@pytest.mark.parametrize('anchor,auxiliary',[(.4,.5),(.8,True),(.8,1.1),(-.1,.0)])
def test_invalid_role_visibility_configuration_fails_before_work(tmp_path,anchor,auxiliary):
    data=plain(configured(tmp_path).data)
    data['clustering'].update(min_anchor_visible_face_fraction=anchor,min_auxiliary_visible_face_fraction=auxiliary)
    with pytest.raises(ValueError,match='visible_face_fraction'):validate(data)


def test_global_scan_floor_cannot_hide_lower_auxiliary_threshold(tmp_path):
    data=plain(configured(tmp_path).data);data['detection']['min_visible_face_fraction']=.8
    with pytest.raises(ValueError,match='lower the scan floor'):validate(data)


def test_scan_measures_role_visibility_when_global_gate_is_disabled(tmp_path):
    from tests.test_recognition_clustering import scan
    from stash_ai.recognition.media import SpriteInput
    from stash_ai.recognition.scanning import scan_scene
    from stash_ai.recognition.models import asset_selection
    from tests.helpers.recognition import MediaClient,Backend
    config=configured(tmp_path);client=MediaClient('1',3)
    class Visibility:
        calls=0
        def measure(self,*args):
            self.calls+=1
            return dict(visible_face_fraction=.65,visibility={'contract':CONTRACT_ID})
    visibility=Visibility()
    assert 'faceocc' in asset_selection(config)[1]
    with FaceDatabase(config.database) as db,SpriteInput(client,client.scene,config) as source:
        sid,_=scan_scene(db,'scope',client.scene,config,'ns',source,Backend,visibility=visibility)
        assert visibility.calls==3
        assert all(json.loads(r[0])['visible_face_fraction']==.65 for r in db.connection.execute('SELECT quality FROM observations'))
        rebuild_selected(db,[dict(scene='1',scan=sid)],config)
        assert not db.connection.execute('SELECT 1 FROM members').fetchone()
