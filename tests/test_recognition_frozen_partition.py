"""Scene IDs, source cue order and batching must not change identity evidence."""
import pytest

from stash_ai.recognition.database import FaceDatabase
from stash_ai.recognition.rebuild import rebuild_selected
from tests.test_recognition_anchor_views import config_for,A,X,C
from tests.test_recognition_clustering import scan
from tests.test_recognition_rebuild import changed
from tests.test_recognition_scene_context import candidates


def fixture(db,config,scene,vectors,aux=()):
    sid,_=scan(db,config,scene,vectors,count=len(vectors))
    for cue in aux:
        db.connection.execute("UPDATE observations SET quality=json_set(quality,'$.pose_deviation',.3) "
            'WHERE scan=? AND cue=?',(sid,cue))
    db.connection.commit()
    return dict(scene=scene,scan=sid)


@pytest.mark.parametrize('scenes',[('1','2'),('2','1')])
@pytest.mark.parametrize('count',[2,4])
def test_independent_local_core_survives_earlier_other_scene(tmp_path,scenes,count):
    config=changed(config_for(tmp_path),recognition={'min_identity_support':2})
    with FaceDatabase(config.database) as db:
        source=fixture(db,config,scenes[0],[A]*3+[X]*3)
        query=fixture(db,config,scenes[1],[C]*count)
        selected=[source,query]
        rebuild_selected(db,selected,config)
        rows=[row for item in selected for row in candidates(db,item['scan'],config)]
        assert len(rows)==6+count
        assert all(row[0]=='accepted' for row in rows)
        assert len({row[1] for row in rows})==1
        rebuild_selected(db,selected,config)
        assert rows==[row for item in selected for row in candidates(db,item['scan'],config)]


@pytest.mark.parametrize('late_core',[False,True])
@pytest.mark.parametrize('scenes',[('1','2'),('2','1')])
@pytest.mark.parametrize('joint',[False,True])
def test_source_profiles_can_precede_its_independent_strict_core(tmp_path,late_core,scenes,joint):
    config=config_for(tmp_path,view_prototypes_enabled=True)
    with FaceDatabase(config.database) as db:
        vectors,aux=([X]*3+[A]*3,range(3)) if late_core else ([A]*3+[X]*3,range(3,6))
        source=fixture(db,config,scenes[0],vectors,aux)
        if not joint:rebuild_selected(db,[source],config)
        query=fixture(db,config,scenes[1],[C]*3)
        rebuild_selected(db,[source,query] if joint else [query],config)
        rows=[row for item in (source,query) for row in candidates(db,item['scan'],config)]
        assert all(row[0]=='accepted' for row in rows)
        assert len({row[1] for row in rows})==1


def test_small_independent_cores_can_share_strict_identity_without_training_views(tmp_path):
    config=changed(config_for(tmp_path),recognition={'min_identity_support':2})
    with FaceDatabase(config.database) as db:
        selected=[fixture(db,config,str(i),[A]*2) for i in range(1,4)]
        rebuild_selected(db,selected,config)
        rows=[row for item in selected for row in candidates(db,item['scan'],config)]
        assert all(row[0]=='accepted' for row in rows)
        assert len({row[1] for row in rows})==1
