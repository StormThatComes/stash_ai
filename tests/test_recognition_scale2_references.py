import numpy as np
import pytest
from stash_ai.recognition import paged_references, compute, parallel_scores, run_cache
from stash_ai.recognition.clustering import _similarities
from tests.test_recognition_compute import configured
from tests.test_recognition_score_blocks import groups


def test_paged_order_duplicates_and_missing():
    pool = paged_references.Pool(configured())
    try:
        vectors=[np.arange(32,dtype=np.float32)+i%7 for i in range(600)]
        samples=pool.pack(vectors)
        queries=[];pool.db.set_trace_callback(queries.append)
        assert all(np.array_equal(a,b) for a,b in zip(samples,vectors))
        assert len([q for q in queries if q.startswith('SELECT')])==3
        assert np.array_equal(samples[-1],vectors[-1])
        assert all(np.array_equal(a,b) for a,b in zip(samples[::-3],vectors[::-3]))
        pool.db.execute('DELETE FROM vectors WHERE id=?',(samples.keys[0],))
        with pytest.raises(ValueError,match='page is missing'):list(samples)
    finally:pool.close()


def test_frozen_paged_scores_reuse_matrix_without_page_reads(monkeypatch):
    monkeypatch.setattr(compute,'available_cpus',lambda:24)
    config=configured(rebuild_workers=4,rebuild_queue_bytes=1048576)
    references,query=groups(512)
    expected={i:_similarities(g,query) for i,g in enumerate(references)}
    pool=paged_references.Pool(config)
    try:
        for group in references:group.samples=pool.pack(group.samples)
        queries=[];pool.db.set_trace_callback(queries.append)
        with compute.compute_scope(config),run_cache.scope(config):
            plan=parallel_scores.FrozenScores(references)
            assert plan.eligible()
            assert not queries
            first=plan.score((query,),float('-inf'))
            assert {p:(m,l) for p,m,l in first[0]}==expected
            queries.clear()
            assert plan.score((query,),float('-inf'))==first
            assert not [q for q in queries if q.startswith('SELECT')]
    finally:pool.close()
