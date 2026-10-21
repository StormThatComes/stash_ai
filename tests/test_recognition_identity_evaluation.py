"""Partition metrics distinguish fragmentation, wrong merges and withholding."""
import pytest
from stash_ai.recognition.evaluation import evaluate_identities,compare_identity_runs


def truth():
    return [dict(observation=str(i),scene='1' if i<3 else '2',truth='A' if i<4 else 'B') for i in range(6)]


def test_identity_metrics_count_cross_scene_and_in_scene_fragmentation():
    predictions=['x','x','y','z','z',None]
    result=evaluate_identities([dict(row,predicted=p) for row,p in zip(truth(),predictions)])
    assert result['observations']==6 and result['assigned']==5 and result['withheld']==1
    assert result['fragmented_identities']==1 and result['extra_fragments']==2
    assert result['fragmented_scene_identities']==1 and result['extra_scene_fragments']==1
    assert result['merged_identities']==1 and result['false_merge_pairs']==1
    assert result['false_split_pairs']==5
    assert result['pair_precision']==.5 and result['pair_recall']==pytest.approx(1/7)


def test_run_comparison_ignores_uuid_renaming_and_distinguishes_unprocessed():
    records=truth()
    good={str(i):'one' if i<4 else 'two' for i in range(6)}
    renamed={key:'renamed-'+value for key,value in good.items()}
    changed=dict(good,**{'0':None,'1':'extra'})
    subset={key:good[key] for key in ('0','1')}
    report=compare_identity_runs(records,dict(full=good,renamed=renamed,changed=changed,subset=subset))
    pair=next(p for p in report['comparisons'] if p['left']=='full' and p['right']=='renamed')
    assert pair['equivalent'] and pair['common_observations']==6
    pair=next(p for p in report['comparisons'] if p['left']=='changed' and p['right']=='full')
    assert pair['coverage_changes']==1 and pair['partition_changes']>0 and not pair['equivalent']
    assert report['runs']['subset']['observations']==2 and report['runs']['subset']['withheld']==0


def test_withheld_samples_are_not_a_merged_identity():
    result=evaluate_identities([dict(row,predicted=None) for row in truth()])
    assert result['predicted_identities']==0 and result['merged_identities']==0
    assert result['withheld']==6 and result['pair_precision'] is None
    assert result['pair_recall']==0


@pytest.mark.parametrize('records',[
    [dict(observation='a',scene='1',truth='A',predicted='x')]*2,
    [dict(observation='a',scene='1',predicted='x')],
    [dict(observation='a',scene='1',truth='A',predicted=False)],
])
def test_identity_metrics_require_explicit_unique_ground_truth(records):
    with pytest.raises(ValueError):
        evaluate_identities(records)


def test_run_comparison_refuses_unknown_observations():
    with pytest.raises(ValueError):
        compare_identity_runs(truth(),{'bad':{'unknown':'person'}})


def test_complete_replay_metrics_compare_separate_and_joint_runs(tmp_path):
    import numpy as np
    from stash_ai.recognition.database import FaceDatabase
    from stash_ai.recognition.rebuild import rebuild_selected
    from tests.test_recognition_anchor_views import config_for,A,X,C
    from tests.test_recognition_clustering import scan
    from stash_ai.recognition.clustering import matching_policy
    config=config_for(tmp_path,view_prototypes_enabled=True)
    selections=[]
    with FaceDatabase(config.database) as db:
        for scene,vectors in [('1',[A]*3+[X]*3+[np.array([0.,0.,1.,0.])]*3),('2',[C]*3)]:
            sid,_=scan(db,config,scene,vectors,count=len(vectors))
            item=dict(scene=scene,scan=sid);selections.append(item)
            rebuild_selected(db,[item],config)
        records=[dict(observation=row['id'],scene=row['scene'],truth='other' if np.frombuffer(row['embedding'],dtype='<f4')[2]>.5 else 'one')
            for row in db.connection.execute('SELECT o.id,o.embedding,s.scene FROM observations o JOIN scans s ON s.id=o.scan')]
        def assignments():
            return {row[0]:row[1] for row in db.connection.execute("SELECT json_extract(payload,'$.observation'),CASE WHEN status='accepted' THEN person END FROM candidates WHERE policy=?",
                (matching_policy(config),))}
        separate=assignments()
        rebuild_selected(db,list(reversed(selections)),config)
        report=compare_identity_runs(records,{'separate':separate,'joint':assignments()})
        assert report['comparisons'][0]['equivalent']
        assert all(run['extra_fragments']==0 and run['false_merge_pairs']==0 and run['withheld']==0 for run in report['runs'].values())
