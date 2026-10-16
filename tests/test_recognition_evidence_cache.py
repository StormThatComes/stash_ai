import numpy as np
import pytest

from stash_ai.recognition.clustering import Group, co_presence_allows, vector
from stash_ai.recognition.database import FaceDatabase
from stash_ai.recognition.evidence_cache import evidence_scope, latest_rows
from tests.test_recognition_clustering import setup, scan
from tests.test_recognition_identity_merges import changed


@pytest.mark.parametrize('tiny', [False, True])
def test_latest_heads_match_correlated_sql_including_newer_partial_policy(tmp_path, tiny):
    config = setup(tmp_path)
    with FaceDatabase(config.database) as state:
        old, _ = scan(state, config, '1')
        newer, _ = scan(state, changed(config, detection={'min_sharpness': 1}), '1')
        state.connection.execute("UPDATE scans SET status='partial' WHERE id=?", (newer,))
        state.connection.commit()
        selected = state.connection.execute('SELECT * FROM scans WHERE id=?', (old,)).fetchone()
        sql = ("SELECT o.id,s.id AS scan,s.scene FROM observations o JOIN scans s ON s.id=o.scan "
               "WHERE s.scope=? AND s.namespace=? AND s.status='complete' AND s.policy_sha=? {latest}")
        args = selected['scope'], selected['namespace'], selected['policy_sha']
        assert list(latest_rows(state.connection, sql, args, selected)) == []
        with evidence_scope(state.connection, config) as cache:
            if tiny:
                cache.cache.limit = 1
            assert list(latest_rows(state.connection, sql, args, selected)) == []
            assert list(latest_rows(state.connection, sql, args, selected)) == []


def test_cue_rows_and_exact_peer_scores_are_reused_without_eager_bad_vector_validation(tmp_path, monkeypatch):
    from stash_ai.recognition import clustering
    config = setup(tmp_path)
    a, b = np.array([1.,0,0,0]), np.array([.8,.6,0,0])
    with FaceDatabase(config.database) as state:
        identifier, _ = scan(state, config, '1', [a,b,a])
        rows = state.connection.execute('SELECT * FROM observations ORDER BY cue').fetchall()
        names = ['10000000-0000-0000-0000-000000000000', '00000000-0000-0000-0000-000000000000',
                 'ffffffff-ffff-ffff-ffff-ffffffffffff']
        for row, name in zip(rows, names):
            state.connection.execute('UPDATE observations SET id=?,cue=0 WHERE id=?', (name,row['id']))
        state.connection.execute('UPDATE observations SET embedding=? WHERE id=?', (b'bad',names[-1]))
        state.connection.commit()
        query = state.connection.execute('SELECT * FROM observations WHERE id=?',(names[0],)).fetchone()
        group = Group('p', (vector(query),), True)
        expected = co_presence_allows(state.connection,query,group,[group],vector(query),config)
        assert expected is False  # The early peer conflict precedes the bad vector.
        with evidence_scope(state.connection, config):
            assert co_presence_allows(state.connection,query,group,[group],vector(query),config) is expected
            calls = []
            state.connection.set_trace_callback(calls.append)
            monkeypatch.setattr(clustering,'_similarities',lambda *a: pytest.fail('unchanged score recomputed'))
            assert co_presence_allows(state.connection,query,group,[group],vector(query),config) is expected
            assert not calls
            state.connection.set_trace_callback(None)


def test_cache_respects_changed_provisional_samples_and_byte_budget(tmp_path):
    from stash_ai.recognition.evidence_cache import MemoryCache, MISSING
    config = setup(tmp_path)
    with FaceDatabase(config.database) as state:
        identifier, _ = scan(state, config, '1')
        row = state.connection.execute('SELECT * FROM observations WHERE scan=? LIMIT 1',(identifier,)).fetchone()
        query = vector(row)
        group = Group('p',[query],False)
        with evidence_scope(state.connection,config) as cache:
            assert cache.scores(group,query,row['id']) == (1.,1.)
            group.samples.append(np.frombuffer((-query).tobytes(),dtype='<f4'))
            assert cache.scores(group,query,row['id']) == (0.,-1.)
    cache=MemoryCache(512)
    cache.put('one',1,200)
    cache.put('two',2,200)
    assert cache.get('one') is MISSING and cache.get('two') == 2
    assert cache.size <= cache.limit
