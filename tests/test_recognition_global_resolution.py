"""Automatic labels and unrelated scene IDs cannot veto a frozen view repair."""
import pytest
from stash_ai.recognition.database import FaceDatabase
from stash_ai.recognition.rebuild import rebuild_selected
from tests.test_recognition_anchor_views import config_for,A,X,C
from tests.test_recognition_clustering import scan
from tests.test_recognition_scene_context import candidates
from tests.test_recognition_rebuild import changed


def source(db,config,scene):
    sid,_=scan(db,config,scene,[A]*3+[X]*3,count=6)
    db.connection.execute("UPDATE observations SET quality=json_set(quality,'$.pose_deviation',.3) "
                          'WHERE scan=? AND cue>=3',(sid,))
    db.connection.commit()
    return {'scene':scene,'scan':sid}


def partition(db,selection,config):
    return {r[1] for item in selection for r in candidates(db,item['scan'],config) if r[0]=='accepted'}


@pytest.mark.parametrize('scenes',[('1','2'),('2','1')])
def test_global_view_resolution_does_not_use_scene_id_as_time(tmp_path,scenes):
    config=config_for(tmp_path,view_prototypes_enabled=True)
    with FaceDatabase(config.database) as db:
        first=source(db,config,scenes[0])
        sid,_=scan(db,config,scenes[1],[C]*3,count=3)
        selected=[first,{'scene':scenes[1],'scan':sid}]
        rebuild_selected(db,selected,config)
        assert len(partition(db,selected,config))==1
        assert all(r[0]=='accepted' for item in selected for r in candidates(db,item['scan'],config))
        assert all(r[2]['reference_role']=='attachment' for r in candidates(db,sid,config))


def test_joint_rebuild_repairs_owned_fragments_and_keeps_label_records(tmp_path):
    config=config_for(tmp_path,view_prototypes_enabled=True)
    legacy=changed(config,clustering={'view_prototypes_enabled':False})
    with FaceDatabase(config.database) as db:
        first=source(db,legacy,'1');rebuild_selected(db,[first],legacy)
        sid,_=scan(db,legacy,'2',[C]*3,count=3)
        selected=[first,{'scene':'2','scan':sid}]
        rebuild_selected(db,[selected[1]],legacy)
        old=partition(db,selected,legacy)
        assert len(old)==2
        for index,person in enumerate(sorted(old)):
            db.connection.execute('UPDATE people SET performer=? WHERE id=?',(str(100+index),person))
        db.connection.commit()
        labels=[tuple(r) for r in db.connection.execute('SELECT id,performer,token FROM people ORDER BY id')]
        rebuild_selected(db,selected,config)
        assert len(partition(db,selected,config))==1
        assert partition(db,selected,config)<=old
        assert [tuple(r) for r in db.connection.execute('SELECT id,performer,token FROM people ORDER BY id')]==labels
        # A cold replay after a policy change must make the same repair.
        cold=changed(config,recognition={'min_merge_support':4})
        rebuild_selected(db,selected,cold)
        assert len(partition(db,selected,cold))==1


def test_global_view_resolution_keeps_copresent_fragments_separate(tmp_path):
    config=config_for(tmp_path,view_prototypes_enabled=True)
    with FaceDatabase(config.database) as db:
        sid,_=scan(db,config,'1',[A]*3+[X]*3+[C]*3,count=9)
        db.connection.execute("UPDATE observations SET quality=json_set(quality,'$.pose_deviation',.3) "
                              'WHERE scan=? AND cue BETWEEN 3 AND 5',(sid,))
        db.connection.execute('UPDATE observations SET cue=0,seconds=0 WHERE scan=? AND cue=6',(sid,))
        db.connection.commit()
        selected=[{'scene':'1','scan':sid}]
        rebuild_selected(db,selected,config)
        assert len(partition(db,selected,config))==2
