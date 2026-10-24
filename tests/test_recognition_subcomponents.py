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


def test_three_member_clique_survives_a_weaker_neighbour(tmp_path):
    from tests.test_recognition_clustering import scan
    from tests.test_recognition_global_views import views
    from stash_ai.recognition.clustering import cluster_scene
    config=changed(config_for(tmp_path),recognition={'match_threshold':.9,'ambiguity_margin':.02},
                   clustering={'view_match_threshold':.9,'view_ambiguity_margin':.02})
    with FaceDatabase(config.database) as db:
        strict=changed(config,recognition={'match_threshold':.99999,'ambiguity_margin':.000001})
        for scene,angle in enumerate((0,4,8,32),1):
            sid,_=scan(db,config,str(scene),[vector(angle)]*3)
            cluster_scene(db,sid,strict)
        frozen=views(db,config)
        for bank in (frozen,tuple(reversed(frozen))):
            plan=consolidation_plan(db.connection,bank,config)
            merges=[p for p in plan if p['outcome']=='merge']
            assert len(merges)==1 and len(merges[0]['people'])==3
            assert merges[0]['partition_strategy']=='supported_complete_subgroup'
            assert merges[0]['nearest_rival']['gap']>=.02


def test_subgroup_budget_fails_closed_and_overlapping_hypotheses_veto():
    from stash_ai.recognition.frozen_subgroups import Edges,nonoverlapping
    edges=Edges(256)
    edges.add('a','b',.99);edges.add('b','c',.98)
    assert not edges.available and not edges.proposals(['a','b','c'],.9,.02)
    assert nonoverlapping([('a','b'),('b','c')])==[]
    assert nonoverlapping([('a','b'),('a','b','c')])==[('a','b','c')]


def test_final_canonical_retry_recovers_queries_without_training(tmp_path):
    from tests.test_recognition_clustering import scan
    from tests.test_recognition_scene_context import candidates
    from stash_ai.recognition.historical_references import capture
    config=changed(config_for(tmp_path),recognition={'match_threshold':.9,'ambiguity_margin':.02},
                   clustering={'view_match_threshold':.9,'view_ambiguity_margin':.02})
    with FaceDatabase(config.database) as db:
        selected=[]
        for scene,(angle,count) in enumerate(((0,3),(5,3),(29,3),(2,2)),1):
            sid,_=scan(db,config,str(scene),[vector(angle)]*count,count=count)
            selected.append(dict(scene=str(scene),scan=sid))
        rebuild_selected(db,selected,config)
        rows=candidates(db,selected[-1]['scan'],config)
        assert all(r[0]=='accepted' and r[2]['reason']=='final_recovery_identity' for r in rows)
        ids={r[2]['observation'] for r in rows}
        scope,ns=db.connection.execute('SELECT scope,namespace FROM scans LIMIT 1').fetchone()
        bank=capture(db.connection,[],config,scope,ns,'test')
        assert all(not ids.intersection(v.source_ids) for v in bank.views)
        assert all(not ids.intersection(g.sample_ids) for g in bank.groups)
