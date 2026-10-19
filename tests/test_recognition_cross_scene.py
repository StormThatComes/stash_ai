"""Cross-scene support is opt-in, non-transitive and fully replayable."""
from dataclasses import replace
import json
import math

import numpy as np
import pytest

from stash_ai.recognition.clustering import matching_policy
from stash_ai.recognition.config import plain
from stash_ai.recognition.database import FaceDatabase
from stash_ai.recognition.rebuild import rebuild_selected
from stash_ai.recognition.scanning import scan_identity
from tests.test_recognition_clustering import scan, setup

A = np.array([1.,0.,0.,0.])
B = np.array([0.,1.,0.,0.])
NEAR = np.array([.9,math.sqrt(1-.9**2),0.,0.])


def config_for(tmp_path, *, enabled=True, minimum=2):
    config = setup(tmp_path)
    data=plain(config.data)
    data['recognition']['min_identity_support']=minimum
    data['recognition']['cross_scene_provisional']=enabled
    return replace(config,data=data)


def replay(db, config, scenes):
    selected=[]
    for scene, vectors in scenes:
        sid,_=scan(db,config,scene,vectors,count=len(vectors))
        selected.append({'scene':scene,'scan':sid})
    rebuild_selected(db,selected,config)
    return selected


def statuses(db,config,selected):
    policy=matching_policy(config)
    return [list(db.connection.execute(
        "SELECT c.status,c.person,json_extract(c.payload,'$.reason') AS reason "
        "FROM candidates c WHERE c.scan=? AND c.policy=? ORDER BY c.id",
        (item['scan'],policy))) for item in selected]


def test_two_independent_singletons_from_different_scenes_form_one_person(tmp_path):
    config=config_for(tmp_path)
    with FaceDatabase(config.database) as db:
        selected=replay(db,config,[('1',[A]),('2',[A])])
        rows=statuses(db,config,selected)
        assert [r[0]['status'] for r in rows]==['accepted','accepted']
        assert len({r[0]['person'] for r in rows})==1
        assert all(r[0]['reason']=='cross_scene_provisional' for r in rows)
        assert db.connection.execute('SELECT COUNT(*) FROM people').fetchone()[0]==1
        assert db.connection.execute('SELECT COUNT(*) FROM identity_merges').fetchone()[0]==0
        # Cached, identical replay cannot change the partition.
        rebuild_selected(db,selected,config)
        assert statuses(db,config,selected)==rows


def test_provisional_singleton_can_attach_to_already_confirmed_other_scene_person(tmp_path):
    config=config_for(tmp_path)
    with FaceDatabase(config.database) as db:
        selected=replay(db,config,[('1',[A]),('2',[A,A])])
        rows=statuses(db,config,selected)
        assert rows[0][0]['reason']=='cross_scene_identity'
        assert rows[0][0]['status']=='accepted'
        assert len({r['person'] for group in rows for r in group})==1


@pytest.mark.parametrize('scenes',[
    [('1',[A]),('2',[B])],
    [('1',[A])],
])
def test_unrelated_or_same_scene_only_cannot_establish_provisional_person(tmp_path,scenes):
    config=config_for(tmp_path)
    with FaceDatabase(config.database) as db:
        selected=replay(db,config,scenes)
        assert all(row['status']=='withheld'
                   for group in statuses(db,config,selected) for row in group)
        assert db.connection.execute('SELECT COUNT(*) FROM people').fetchone()[0]==0


def test_opt_out_preserves_withheld_result_and_changes_only_matching_policy(tmp_path):
    enabled=config_for(tmp_path)
    disabled=config_for(tmp_path,enabled=False)
    assert matching_policy(enabled)!=matching_policy(disabled)
    assert scan_identity(enabled,'ns')==scan_identity(disabled,'ns')
    with FaceDatabase(enabled.database) as db:
        selected=replay(db,disabled,[('1',[A]),('2',[A])])
        assert all(r['status']=='withheld' for x in statuses(db,disabled,selected) for r in x)
        rebuild_selected(db,selected,enabled)
        assert all(r['status']=='accepted' for x in statuses(db,enabled,selected) for r in x)
        rebuild_selected(db,selected,disabled)
        assert all(r['status']=='withheld' for x in statuses(db,disabled,selected) for r in x)


def test_ambiguous_provisional_neighbours_remain_unpublished(tmp_path):
    config=config_for(tmp_path)
    # Both neighbours match A closely, but one cue cannot distinguish B/C.
    C=np.array([.9,-math.sqrt(1-.9**2),0.,0.])
    with FaceDatabase(config.database) as db:
        selected=replay(db,config,[('1',[A]),('2',[NEAR]),('3',[C])])
        assert all(r['status']=='withheld'
                   for group in statuses(db,config,selected) for r in group)
        assert db.connection.execute('SELECT COUNT(*) FROM identity_merges').fetchone()[0]==0


def test_min_support_three_needs_three_independent_scenes(tmp_path):
    config=config_for(tmp_path,minimum=3)
    with FaceDatabase(config.database) as db:
        selected=replay(db,config,[('1',[A]),('2',[A]),('3',[A])])
        rows=statuses(db,config,selected)
        assert len({r['person'] for group in rows for r in group})==1
        assert all(r['status']=='accepted' for group in rows for r in group)


def test_configuration_rejects_non_boolean_flag(tmp_path):
    path=tmp_path/'invalid.yml'
    path.write_text('version: 1\nrecognition: {cross_scene_provisional: "true"}\n')
    from stash_ai.recognition.config import load
    with pytest.raises(ValueError,match='cross_scene_provisional'):
        load(path)
