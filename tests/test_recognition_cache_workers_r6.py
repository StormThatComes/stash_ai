from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np

from tests.test_recognition_compute import configured
from stash_ai.recognition.run_cache import scope as cache_scope
from stash_ai.recognition.solver_shared import Arrays,scope as shared_scope


def test_scalar_dots_do_not_read_or_create_shared_files(tmp_path):
    from stash_ai.recognition.body_numeric import dot
    Arrays.prepare(tmp_path)
    vector=np.ones(4096,dtype='<f4')
    with shared_scope(tmp_path,1024**2) as shared:
        with patch.object(shared,'get',side_effect=AssertionError('disk lookup for cheap dot')):
            assert dot(vector,vector,lambda:None)==4096.
    assert not list(tmp_path.glob('*.npy'))


def test_shared_expensive_reduction_is_promoted_to_ram(tmp_path):
    from stash_ai.recognition.body_numeric import lookup,publish
    from stash_ai.recognition.body_parallel import _compare
    rng=np.random.default_rng(7)
    banks=[tuple(r for r in rng.normal(size=(16,4096)).astype('<f4')) for _ in range(2)]
    Arrays.prepare(tmp_path)
    with shared_scope(tmp_path,8*1024**2):
        identity,found=lookup(*banks,'body');assert found is None
        expected=publish(identity,_compare(banks))
    with shared_scope(tmp_path,8*1024**2) as shared,cache_scope(configured()):
        with patch.object(shared,'get',wraps=shared.get) as get:
            for _ in range(10):assert lookup(*banks,'body')[1]==expected
        assert get.call_count==1


def test_warm_and_mixed_matrix_tiles_skip_cached_executor_payload(monkeypatch):
    from stash_ai.recognition.compute import compute_scope
    from stash_ai.recognition import parallel_numeric
    monkeypatch.setattr('stash_ai.recognition.compute.available_cpus',lambda:4)
    config=configured(rebuild_workers=2)
    a=np.arange(12,dtype='float64').reshape(3,4);b=np.ones((2,4));c=b*2
    with cache_scope(config),compute_scope(config) as pool:
        expected=list(parallel_numeric.matrix_tiles([('one',[(a,b)])],kind='tiles'))[0][1][0]
        with patch.object(pool,'ordered',side_effect=AssertionError('cached executor job')):
            assert np.array_equal(list(parallel_numeric.matrix_tiles([('one',[(a,b)])],kind='tiles'))[0][1][0],expected)
        original=parallel_numeric._matrices;received=[]
        def compute(payload):received.append(len(payload));return original(payload)
        monkeypatch.setattr(parallel_numeric,'_matrices',compute)
        values=list(parallel_numeric.matrix_tiles([('two',[(a,b),(a,c)])],kind='tiles'))[0][1]
        assert received==[1]
        assert np.array_equal(values[0],expected) and np.array_equal(values[1],a@c.T)


def test_worker_reuses_only_pure_cache_and_same_goal_index(tmp_path):
    from tests.test_recognition_merge_report import seeded
    from stash_ai.recognition.solver_worker_reuse import Reuse
    from stash_ai.recognition.solver_execution import plan
    from stash_ai.recognition import run_cache,merge_goal_index
    from stash_ai.recognition.merge_report_policy import patched
    config,_=seeded(tmp_path)
    request=SimpleNamespace(config=config,execution=plan(config,dict(cpus=1)),seed_path='immutable-seed',
        goals=dict(groups=[]),shared_directory=None,shared_bytes=0)
    reuse=Reuse()
    try:
        assert not reuse.prepare(request)
        cache=run_cache.current();index=merge_goal_index.current()
        left=np.ones((2,4));right=np.ones((3,4))
        expected=run_cache.products(left,right)
        cache.put(('trial-membership',object()),{'owner':'old'},1024)
        reuse.completed()
        request.config=patched(config,{'recognition.match_threshold':.99})
        assert reuse.prepare(request)
        assert run_cache.current() is cache and merge_goal_index.current() is index
        with merge_goal_index.prepared(request.goals):assert merge_goal_index.current() is index
        assert all(key[0]!='trial-membership' for key in cache.entries)
        assert run_cache.products(left,right) is expected
        request.goals=dict(groups=[])
        assert not reuse.prepare(request) and merge_goal_index.current() is not index
    finally:reuse.close()
    assert run_cache.current() is None and merge_goal_index.current() is None
