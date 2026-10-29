"""Real scan regressions for scoped aliases, bounded modes and dense body banks."""
import math
import json
from unittest.mock import patch
import numpy as np
import pytest
from stash_ai.recognition.database import FaceDatabase,encoded
from stash_ai.recognition.body_identity import scene_plan,Scores
from tests.test_recognition_body_identity import configuration,scene
from tests.test_recognition_merge_report import seeded


def test_1415_distinct_coherent_vectors_remain_one_identity_with_bounded_work(tmp_path,monkeypatch):
    config=configuration(tmp_path);count=1415
    vectors=[[math.cos(i*1e-6),math.sin(i*1e-6)] for i in range(count)]
    observed=[]
    class Meter(Scores):
        def __init__(self,config):super().__init__(config);observed.append(self)
    monkeypatch.setattr('stash_ai.recognition.body_identity.Scores',Meter)
    with FaceDatabase(config.database) as state:
        item,ids=scene(state.connection,config,vectors=vectors)
        state.connection.commit()
        _,_,decisions=scene_plan(state.connection,item['scan'],config)
        assert {r['status'] for r in decisions.values()}=={'accepted'}
        assert len({r['person'] for r in decisions.values()})==1
        assert observed[0].count<=3*count
        assert len({tuple(v) for v in vectors})==count


@pytest.mark.parametrize('seed',range(8))
def test_certificates_preserve_exact_decisions_with_outliers_and_rivals(tmp_path,seed):
    config=configuration(tmp_path);rng=np.random.default_rng(seed)
    angles=np.r_[rng.normal(0,.001,12),.55,-.55,rng.uniform(-1,1,16)]
    config.data['sapiensid']['identities'].update(match_threshold=.9,cohesion_threshold=.85,ambiguity_margin=seed*.01)
    with FaceDatabase(config.database) as state:
        item,_=scene(state.connection,config,vectors=[[math.cos(a),math.sin(a)] for a in angles])
        state.connection.commit()
        actual=scene_plan(state.connection,item['scan'],config)[2]
        with patch.object(Scores,'certifies',return_value=False):
            expected=scene_plan(state.connection,item['scan'],config)[2]
        assert actual==expected
        assert len({r['person'] for r in actual.values()})>1


def test_inconclusive_certificate_cannot_hide_original_outlier_or_threshold_edge(tmp_path):
    config=configuration(tmp_path);settings=config.data['sapiensid']['identities']
    settings.update(match_threshold=.85,cohesion_threshold=.85)
    # Successive extensions fit the current mean, but violate the first witness.
    vectors=[[math.cos(a),math.sin(a)] for a in [0,.4,.65]]
    with FaceDatabase(config.database) as state:
        item,ids=scene(state.connection,config,vectors=vectors)
        _,_,result=scene_plan(state.connection,item['scan'],config)
        assert result[ids[-1]]['status']=='withheld'
    scores=Scores(config)
    sample=dict(id='a',quality='{}',_vector=np.array([1,0],dtype='float32'))
    query=dict(id='b',quality='{}',_vector=np.array([.85,math.sqrt(1-.85**2)],dtype='float32'))
    assert not scores.certifies(query,[sample],.85)


def test_disk_mode_budget_is_enforced_before_next_dot_and_reports_completed_work(tmp_path):
    from stash_ai.recognition.disk_body_modes import Records,ComparisonBudget
    from types import SimpleNamespace
    config=configuration(tmp_path);config.data['sapiensid']['execution']['max_witness_comparisons']=1
    progress=[];budget=ComparisonBudget(config,SimpleNamespace(advance=lambda:progress.append(1)))
    spool=Records();directory=spool.path
    try:
        for i in range(10):
            vector=np.zeros(4096,dtype='<f4');vector[:2]=[math.cos(i*.001),math.sin(i*.001)]
            spool.add(dict(id=str(i),scan='s',seconds=float(i),quality='{}',embedding=vector.tobytes()))
        with pytest.raises(ValueError,match='max_witness_comparisons'):
            spool.modes(config,budget)
        assert budget.count==len(progress)==1
    finally:spool.close()
    assert not directory.exists()


def test_disk_mode_comparison_budget_is_shared_across_owners(tmp_path):
    from tests.test_recognition_hybrid_merge import prepared
    from stash_ai.recognition.hybrid_references import catalogue_rows
    config,people=prepared(tmp_path)
    config.data['sapiensid']['execution']['max_witness_comparisons']=4
    with FaceDatabase(config.database) as state:
        db=state.connection
        for person in people:
            for i,row in enumerate(db.execute('SELECT observation FROM members WHERE person=? ORDER BY observation',(person,))):
                value=np.zeros(4096,dtype='<f4');value[:2]=[math.cos(i*.001),math.sin(i*.001)]
                db.execute('UPDATE person_evidence SET embedding=? WHERE observation=?',(value.tobytes(),row[0]))
        db.commit();ns=db.execute('SELECT namespace FROM people LIMIT 1').fetchone()[0]
        with pytest.raises(ValueError,match='max_witness_comparisons'):
            catalogue_rows(db,config,'scope',ns,people)


@pytest.mark.parametrize('foreign', ['scope','namespace'])
def test_normal_rebuild_excludes_foreign_pixel_index_from_working_budget(tmp_path,foreign):
    from stash_ai.recognition.rebuild import rebuild_selected
    from stash_ai.recognition.evidence_pixels import aliases
    from dataclasses import replace
    from stash_ai.recognition.config import plain
    config,people=seeded(tmp_path,count=8)
    config=replace(config,data=plain(config.data))
    config.data['clustering']['max_reference_bytes']=49152
    with FaceDatabase(config.database) as state:
        db=state.connection;rng=np.random.default_rng(42)
        for i,row in enumerate(db.execute('SELECT id,quality FROM observations').fetchall()):
            metadata=json.loads(row['quality']);metadata.update(evidence_thumbnail=rng.integers(0,256,3072,dtype='uint8').tobytes().hex(),evidence_pixel_sha256=f'{i:064x}',evidence_aspect=1.)
            db.execute('UPDATE observations SET quality=? WHERE id=?',(encoded(metadata),row['id']))
        other=db.execute('SELECT o.scan FROM observations o JOIN members m ON m.observation=o.id WHERE m.person=? LIMIT 1',(people[1],)).fetchone()[0]
        db.execute('UPDATE scans SET '+foreign+'=? WHERE id=?',('foreign',other))
        db.execute('UPDATE people SET '+foreign+'=? WHERE id=?',('foreign',people[1]));db.commit()
        with pytest.raises(ValueError,match='budget'):list(aliases(db,budget=49152))
        item=dict(db.execute('SELECT id AS scan,scene FROM scans WHERE id!=?',(other,)).fetchone())
        rebuild_selected(state,[item],config)
        assert db.execute('SELECT COUNT(*) FROM members m JOIN observations o ON o.id=m.observation WHERE o.scan=?',(item['scan'],)).fetchone()[0]==8


def test_weak_queries_use_certificate_but_cannot_train_bank(tmp_path,monkeypatch):
    config=configuration(tmp_path);count=200
    config.data['sapiensid']['execution']['max_witness_comparisons']=2000
    observed=[]
    class Meter(Scores):
        def __init__(self,config):super().__init__(config);observed.append(self)
    monkeypatch.setattr('stash_ai.recognition.body_identity.Scores',Meter)
    with FaceDatabase(config.database) as state:
        vectors=[[math.cos(i*1e-6),math.sin(i*1e-6)] for i in range(count*2)]
        item,ids=scene(state.connection,config,vectors=vectors)
        for identifier in ids[count:]:
            state.connection.execute("UPDATE body_observations SET quality=json_set(quality,'$.quality.sharpness',0) WHERE id=?",(identifier,))
        _,_,decisions=scene_plan(state.connection,item['scan'],config)
        assert {decisions[i]['role'] for i in ids[count:]}=={'auxiliary'}
        assert {decisions[i]['status'] for i in ids[count:]}=={'accepted'}
        assert all(decisions[i]['independent_support']==count for i in ids[:count])
        assert observed[0].count<1000
