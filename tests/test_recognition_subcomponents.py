"""A strong reciprocal pair may survive a weaker connected neighbour."""
from itertools import permutations
import math
import numpy as np
import pytest

from stash_ai.recognition.database import FaceDatabase
from stash_ai.recognition.global_views import consolidation_plan
from stash_ai.recognition.rebuild import rebuild_selected
from tests.test_recognition_anchor_views import config_for
from tests.test_recognition_global_views import replay
from tests.test_recognition_rebuild import changed


def vector(angle):
    angle=math.radians(angle)
    return np.array([math.cos(angle),math.sin(angle),0.,0.])


@pytest.mark.parametrize('angles',list(permutations((0,5,29))))
def test_full_rebuild_recovers_pair_independent_of_scene_order(tmp_path,angles):
    config=changed(config_for(tmp_path),recognition={'match_threshold':.9,'ambiguity_margin':.02},
                   clustering={'view_match_threshold':.9,'view_ambiguity_margin':.02})
    with FaceDatabase(config.database) as db:
        selected=replay(db,config,{str(i):[vector(angle)]*3 for i,angle in enumerate(angles,1)})
        def partition():
            return dict(db.connection.execute('SELECT s.scene,m.person FROM members m '
                'JOIN observations o ON o.id=m.observation JOIN scans s ON s.id=o.scan'))
        actual=partition()
        first,second=(str(angles.index(a)+1) for a in (0,5))
        other=str(angles.index(29)+1)
        assert actual[first]==actual[second]!=actual[other]
        rebuild_selected(db,selected,config)
        assert partition()==actual
        db.connection.execute("DELETE FROM metadata WHERE key LIKE 'identity-rebuild:%'")
        db.connection.commit()
        rebuild_selected(db,selected,config)
        assert partition()==actual


def test_frozen_pair_records_original_component_and_all_outsiders(tmp_path):
    from tests.test_recognition_global_views import seed,views
    config=changed(config_for(tmp_path),recognition={'match_threshold':.9,'ambiguity_margin':.02},
                   clustering={'view_match_threshold':.9,'view_ambiguity_margin':.02})
    with FaceDatabase(config.database) as db:
        from stash_ai.recognition.clustering import cluster_scene
        from tests.test_recognition_clustering import scan
        seed_config=changed(config,recognition={'match_threshold':.99999,'ambiguity_margin':.000001})
        for scene,angle in enumerate((0,5,29),1):
            sid,_=scan(db,config,str(scene),[vector(angle)]*3)
            cluster_scene(db,sid,seed_config)
        frozen=views(db,config)
        original=[tuple(r) for r in db.connection.execute('SELECT * FROM members')]
        plan=consolidation_plan(db.connection,frozen,config)
        pair=next(p for p in plan if p['outcome']=='merge')
        assert pair['partition_strategy']=='reciprocal_supported_pair'
        assert len(pair['parent_component'])==3 and len(pair['people'])==2
        assert pair['nearest_rival']['rival'] not in pair['people']
        assert pair['nearest_rival']['gap']>=.02
        assert original==[tuple(r) for r in db.connection.execute('SELECT * FROM members')]
        assert len({p for d in plan for p in d['people']})==sum(len(d['people']) for d in plan)==3


def test_close_competitor_blocks_reciprocal_pair(tmp_path):
    config=changed(config_for(tmp_path),recognition={'match_threshold':.9,'ambiguity_margin':.1},
                   clustering={'view_match_threshold':.9,'view_ambiguity_margin':.1})
    with FaceDatabase(config.database) as db:
        from tests.test_recognition_clustering import scan
        from tests.test_recognition_global_views import views
        from stash_ai.recognition.clustering import cluster_scene
        strict=changed(config,recognition={'match_threshold':.99999,'ambiguity_margin':.000001})
        for scene,angle in enumerate((0,5,29),1):
            sid,_=scan(db,config,str(scene),[vector(angle)]*3)
            cluster_scene(db,sid,strict)
        assert all(p['outcome']=='blocked' for p in consolidation_plan(db.connection,views(db,config),config))
