import gc
import weakref

import numpy as np
import pytest

from stash_ai.recognition.clustering import Group, _groups, vector
from stash_ai.recognition.database import FaceDatabase
from stash_ai.recognition.evidence_cache import evidence_scope
from stash_ai.recognition.identity_merges import _compatible, _supporting_cues, _merge_support_policy
from tests.test_recognition_clustering import setup, scan
from tests.test_recognition_identity_merges import fragments, changed


def test_negative_directional_pair_proofs_are_reused_and_release_inputs(tmp_path, monkeypatch):
    from stash_ai.recognition import identity_merges as merges
    config = setup(tmp_path)
    with FaceDatabase(config.database) as state, evidence_scope(state.connection, config):
        a = np.frombuffer(np.array([1.,0,0,0],dtype='<f4').tobytes(),dtype='<f4')
        b = np.frombuffer(np.array([0.,1,0,0],dtype='<f4').tobytes(),dtype='<f4')
        first, second = Group('a',(a,),True), Group('b',(b,),True)
        original, calls = merges._cross, []
        def compute(*args):
            calls.append(1)
            return original(*args)
        monkeypatch.setattr(merges,'_cross',compute)
        assert _compatible([first],[second],config) is None
        assert _compatible([first],[second],config) is None
        assert len(calls) == 1
        assert _compatible([second],[first],config) is None
        assert len(calls) == 2
        _compatible([first],[second],changed(config,recognition={'match_threshold':.9}))
        assert len(calls) == 3
        ref = weakref.ref(a)
        del first, a
        gc.collect()
        assert ref() is None


@pytest.mark.parametrize('minimum',[2,4])
def test_owned_incremental_votes_equal_ordered_revalidation(tmp_path, minimum):
    config = setup(tmp_path)
    with FaceDatabase(config.database) as state:
        people, _ = fragments(state,config)
        groups = _groups(state.connection,'scope','ns',config.data['clustering'])
        components = {g.identifier:[g] for g in groups}
        source, target = people[1],people[0]
        policy = _merge_support_policy(config)
        bridge = np.array([1.8,.6,0,0]); bridge /= np.linalg.norm(bridge)
        selected,_ = scan(state,config,'10',[bridge]*5,count=5)
        state.connection.execute('UPDATE observations SET seconds=0 WHERE scan=? AND cue=1',(selected,))
        state.connection.commit()
        head = state.connection.execute('SELECT *,rowid AS scan_order FROM scans WHERE id=?',(selected,)).fetchone()
        rows = state.connection.execute('SELECT * FROM observations WHERE scan=? ORDER BY cue',(selected,)).fetchall()
        with evidence_scope(state.connection,config):
            for index in [4,2,1,0,3]:
                row = rows[index]
                inserted = state.connection.execute('INSERT OR IGNORE INTO merge_support VALUES (?,?,?,?)',
                    (source,target,row['id'],policy))
                actual = _supporting_cues(state.connection,head,policy,source,target,components,config,minimum,
                    nomination=(bool(inserted.rowcount),row))
                expected = _supporting_cues(state.connection,head,policy,source,target,components,config,minimum)
                assert actual == expected
            statements=[]
            state.connection.set_trace_callback(statements.append)
            assert _supporting_cues(state.connection,head,policy,source,target,components,config,minimum,
                nomination=(False,rows[-1])) == expected
            assert not statements
            state.connection.set_trace_callback(None)


def test_exact_score_cache_does_not_own_obsolete_representatives(tmp_path, monkeypatch):
    from stash_ai.recognition import clustering
    config = setup(tmp_path)
    with FaceDatabase(config.database) as state, evidence_scope(state.connection,config) as cache:
        query = np.frombuffer(np.array([1.,0,0,0],dtype='<f4').tobytes(),dtype='<f4')
        sample = query.copy()
        sample = np.frombuffer(sample.tobytes(),dtype='<f4')
        group = Group('person',(sample,),True)
        assert cache.scores(group,query,'observation') == (1.,1.)
        with monkeypatch.context() as patch:
            patch.setattr(clustering,'_similarities',lambda *args: pytest.fail('exact score repeated'))
            assert cache.scores(group,query,'observation') == (1.,1.)
        ref = weakref.ref(sample)
        del sample,group
        gc.collect()
        assert ref() is None
