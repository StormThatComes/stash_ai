"""Run cache transparency across content, ownership and transaction changes."""
import json
import sqlite3
import numpy as np
import pytest

from stash_ai.recognition import run_cache, scoring
from stash_ai.recognition.clustering import Group, vector
from stash_ai.recognition.database import FaceDatabase
from tests.test_recognition_clustering import setup
from tests.test_recognition_identity_merges import changed
from tests.test_recognition_score_blocks import groups, immutable


def test_vectors_validate_once_per_content_not_observation_label(tmp_path, monkeypatch):
    config = setup(tmp_path)
    row = dict(id='same', dimensions=4, embedding=immutable([1,0,0,0]).base)
    with run_cache.scope(config) as cache:
        first = vector(row)
        assert not first.flags.writeable
        monkeypatch.setattr(np.linalg, 'norm', lambda a: pytest.fail('revalidated cached content'))
        assert vector(dict(row, id='other')) is first
        assert cache.size <= cache.limit
    monkeypatch.undo()
    with run_cache.scope(config):
        assert not np.array_equal(vector(row), vector(dict(row, embedding=immutable([0,1,0,0]).base)))
        with pytest.raises(ValueError):
            vector(dict(row, embedding=immutable([2,0,0,0]).base))
        with pytest.raises(ValueError):
            vector(dict(row, dimensions=True))


def test_metadata_is_deeply_immutable_and_visibility_preserved(tmp_path):
    from stash_ai.recognition.visibility import CONTRACT_ID, cached
    raw = json.dumps(dict(visible_face_fraction=.9, visibility=dict(contract=CONTRACT_ID), box=[1,2]))
    with run_cache.scope(setup(tmp_path)):
        result = run_cache.metadata(raw)
        assert run_cache.metadata(raw) is result
        assert cached(result)
        with pytest.raises(TypeError):
            result['visibility']['contract'] = 'bad'
        with pytest.raises(TypeError):
            result['box'][0] = 0


def test_packed_banks_reuse_equal_content_and_invalidate_new_content(tmp_path, monkeypatch):
    references, query = groups(128)
    with run_cache.scope(setup(tmp_path)):
        first = scoring._prepared(references, query)
        rebuilt = [Group('new'+g.identifier, tuple(immutable(v.copy()) for v in g.samples), True) for g in references]
        assert scoring._prepared(rebuilt, query) is not None
        assert scoring._prepared(rebuilt, query)[0] is first[0]
        assert not first[0].flags.writeable
        rebuilt[0].samples = (immutable(-query),) + rebuilt[0].samples[1:]
        assert scoring._prepared(rebuilt, query)[0] is not first[0]
        rebuilt[0].samples = tuple(v.copy() for v in rebuilt[0].samples)
        assert scoring._prepared(rebuilt, query)[0] is not scoring._prepared(rebuilt, query)[0]


def test_joint_budget_matrix_subcap_and_zero_cache(tmp_path):
    cache = run_cache.Cache(2000, 800)
    cache.put(('matrix', 1), 1, 200)
    cache.put(('matrix', 2), 2, 200)
    assert cache.get(('matrix', 1)) is run_cache.MISSING
    cache.put(('vector', 1), 1, 800)
    assert cache.size <= cache.limit and cache.peak <= cache.limit
    with run_cache.scope(changed(setup(tmp_path), execution={'rebuild_cache_bytes': 0})) as cache:
        assert run_cache.metadata('{"a":1}')['a'] == 1
        assert not cache.entries
    assert run_cache.current() is None


def test_copresence_cache_handles_updates_external_writes_and_rollback(tmp_path, monkeypatch):
    from stash_ai.recognition import global_views
    from tests.test_recognition_global_views import config_for, seed, A, X
    config = config_for(tmp_path)
    with FaceDatabase(config.database) as state:
        db = state.connection
        first, left = seed(state, config, '1', A)
        second, right = seed(state, config, '2', X)
        db.commit()
        observation = db.execute('SELECT id FROM observations WHERE scan=? LIMIT 1', (second,)).fetchone()[0]
        original, calls = global_views._cooccurs_sql, []
        def counted(*args):
            calls.append(1)
            return original(*args)
        monkeypatch.setattr(global_views, '_cooccurs_sql', counted)
        with run_cache.scope(config):
            assert not global_views._cooccurs(db, left, right)
            assert not global_views._cooccurs(db, left, right)
            assert len(calls) == 1
            db.execute('UPDATE observations SET scan=?,cue=0 WHERE id=?', (first, observation))
            assert global_views._cooccurs(db, left, right)
            db.rollback()
            assert not global_views._cooccurs(db, left, right)
            with sqlite3.connect(config.database) as external:
                external.execute('UPDATE observations SET scan=?,cue=0 WHERE id=?', (first, observation))
            assert global_views._cooccurs(db, left, right)


def test_pair_cache_keeps_original_reduction_and_bypasses_mutable_inputs(tmp_path):
    left, right = [immutable([1,0]), immutable([0,1])], [immutable([.6,.8])]
    with run_cache.scope(setup(tmp_path)) as cache:
        expected = run_cache.cross_scores(left, right)
        assert run_cache.cross_scores([immutable(a.copy()) for a in left], right) == expected
        assert any(k[0] == 'pair' for k in cache.entries)
        right = [right[0].copy()]
        right[0][:] = -right[0]
        assert run_cache.cross_scores(left, right)[0] < 0
