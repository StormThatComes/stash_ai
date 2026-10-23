"""A single strict look may root one generation using disjoint scene evidence."""
import pytest

from stash_ai.recognition.database import FaceDatabase
from stash_ai.recognition.rebuild import rebuild_selected
from tests.test_recognition_corroborated_views import bank
from tests.test_recognition_deterministic_resolution import configured
from tests.test_recognition_frozen_partition import fixture
from tests.test_recognition_query_support import appearance
from tests.test_recognition_scene_context import candidates


def seed(db,config,*,sources=2,queries=2,source_votes=3):
    source=[]
    for i in range(sources):
        source.append(fixture(db,config,str(i+1),[appearance(0)]*3+[appearance(45)]*source_votes))
    rebuild_selected(db,source,config)
    query=[]
    for i in range(queries):
        item=fixture(db,config,str(i+3),[appearance(65)]*3)
        rebuild_selected(db,[item],config);query.append(item)
    return source,query


def promoted(db,config):
    return [v for v in bank(db,config).views if v.authority=='corroborated_view']


def test_one_strict_look_with_two_source_scenes_can_promote_once(tmp_path):
    config=configured(tmp_path)
    with FaceDatabase(config.database) as db:
        source,queries=seed(db,config)
        views=promoted(db,config)
        assert views and all(len(v.promotion['parents'])==1 for v in views)
        proof=views[0].promotion
        assert proof['generation']==1 and proof['scene_count']==2
        parent=proof['parents'][0]
        assert parent['authority']=='strict'
        assert set(parent['source_scenes'])=={'1','2'}
        assert len(parent['source_scene_evidence'])==2
        assert all(len(ids)>=3 for _,ids in parent['source_scene_evidence'])
        assert not set(parent['source_scenes']).intersection(proof['scenes'])
        assert set(parent['support_ids'])<=set(views[0].source_ids)
        owner=candidates(db,source[0]['scan'],config)[0][1]
        selected=source+queries
        for scene in ('5','6'):
            item=fixture(db,config,scene,[appearance(88)]*3);rebuild_selected(db,[item],config)
            selected.append(item)
            assert all(s=='accepted' and p==owner and d['reference_role']=='attachment'
                       for s,p,d in candidates(db,item['scan'],config))
        # The newly recovered 88-degree look cannot train a second generation.
        assert all(float(v.group.samples[0]@appearance(65))>.999 for v in promoted(db,config))
        before=dict(db.connection.execute('SELECT observation,person FROM members'))
        db.connection.execute("DELETE FROM metadata WHERE key LIKE 'identity-rebuild:%'");db.connection.commit()
        rebuild_selected(db,selected,config)
        assert dict(db.connection.execute('SELECT observation,person FROM members'))==before
        assert all(v.promotion['generation']==1 for v in promoted(db,config))


@pytest.mark.parametrize('options',[dict(sources=1),dict(queries=1),dict(source_votes=1)])
def test_each_source_scene_and_two_query_scenes_need_independent_support(tmp_path,options):
    config=configured(tmp_path)
    with FaceDatabase(config.database) as db:
        seed(db,config,**options)
        assert not promoted(db,config)


@pytest.mark.parametrize('change',['owner','timestamp','supersede'])
def test_source_scene_invalidation_revokes_promotion(tmp_path,change):
    config=configured(tmp_path)
    with FaceDatabase(config.database) as db:
        source,_=seed(db,config)
        view=promoted(db,config)[0]
        scene,ids=view.promotion['parents'][0]['source_scene_evidence'][-1]
        if change=='owner':
            db.connection.execute('DELETE FROM members WHERE observation=?',(ids[0],))
        elif change=='timestamp':
            db.connection.execute('UPDATE observations SET seconds=0 WHERE scan=?',(source[-1]['scan'],))
        else:
            from tests.test_recognition_reference_registry import pending
            pending(db,source[-1])
        db.connection.commit()
        assert not promoted(db,config)


def test_source_scene_cannot_be_reused_as_query_scene(tmp_path):
    from stash_ai.recognition.corroborated_views import enrich
    from stash_ai.recognition.reference_usage import ReferenceUsage
    from stash_ai.recognition.historical_references import _source,parameters
    from stash_ai.recognition.reference_compatibility import fallback_sql
    config=configured(tmp_path)
    with FaceDatabase(config.database) as db:
        seed(db,config)
        history=bank(db,config)
        rows=list(db.connection.execute('SELECT m.person,o.*,s.scene '+
            _source(config,fallback_sql(db.connection,config,'ns'),attachment=True),
            parameters([],config,'scope','ns')))
        def reused():
            return (dict(row,scene='1' if row['scene']=='3' else row['scene']) for row in rows)
        result=enrich(db.connection,config,'scope','ns',[],history,
                      ReferenceUsage(config.data['clustering']['max_reference_bytes']),query_rows=reused)
        assert not result
