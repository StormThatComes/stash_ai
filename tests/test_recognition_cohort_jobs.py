"""Scene jobs preserve streaming priorities and never mutate identities."""
from dataclasses import replace
import threading
import numpy as np
import pytest

from stash_ai.recognition import compute, cohort_jobs, local_cohorts
from stash_ai.recognition.database import FaceDatabase
from stash_ai.recognition.config import plain
from tests.test_recognition_anchor_views import config_for, A, X, C
from tests.test_recognition_clustering import scan


@pytest.mark.parametrize('backend', ['threads','processes'])
def test_scene_jobs_match_serial_priorities_with_ordered_coordinator_writes(tmp_path, monkeypatch, backend):
    monkeypatch.setattr(compute, 'available_cpus', lambda: 24)
    config = config_for(tmp_path)
    with FaceDatabase(config.database) as state:
        db = state.connection
        selected = [dict(scene=str(i), scan=scan(state, config, str(i), vectors, count=len(vectors))[0])
                    for i, vectors in enumerate(([A]*3+[X]*3+[C], [C]*3+[A], [X,A,X,C]),1)]
        local_cohorts.prepare(db, selected, config)
        expected = [tuple(r) for r in db.execute('SELECT * FROM local_core_order ORDER BY observation')]
        db.execute('DROP TABLE local_core_order')
        settings = plain(config.data)
        settings['execution'].update(rebuild_workers=3, rebuild_cohort_backend=backend, rebuild_queue_bytes=1048576)
        parallel = replace(config, data=settings)
        writes, owner = [], threading.get_ident()
        db.set_trace_callback(lambda sql: writes.append(threading.get_ident()) if 'INSERT INTO local_core_order' in sql else None)
        with compute.compute_scope(parallel) as pool:
            local_cohorts.prepare(db, selected, parallel)
            assert pool.applied == 3
            assert pool.peak_bytes <= pool.limit
            if backend == 'processes':
                assert pool.process_blocks == 3
        assert writes and set(writes) == {owner}
        assert [tuple(r) for r in db.execute('SELECT * FROM local_core_order ORDER BY observation')] == expected
        assert db.execute('SELECT COUNT(*) FROM people').fetchone()[0] == 0


def test_oversize_scene_uses_streaming_fallback_without_losing_observations(tmp_path, monkeypatch):
    monkeypatch.setattr(compute, 'available_cpus', lambda: 24)
    config = config_for(tmp_path)
    settings = plain(config.data)
    settings['execution'].update(rebuild_workers=2, rebuild_queue_bytes=65536)
    parallel = replace(config, data=settings)
    with FaceDatabase(config.database) as state:
        sid, _ = scan(state, config, '1', [A]*20, count=20)
        with compute.compute_scope(parallel) as pool:
            local_cohorts.prepare(state.connection, [dict(scan=sid,scene='1')], parallel)
            assert pool.fallbacks['cohort_scene_admission'] == 1
        assert state.connection.execute('SELECT COUNT(*) FROM local_core_order').fetchone()[0] == 20


def test_cohort_worker_exception_leaves_authoritative_membership_untouched(tmp_path, monkeypatch):
    monkeypatch.setattr(compute, 'available_cpus', lambda: 24)
    config = config_for(tmp_path)
    settings = plain(config.data); settings['execution']['rebuild_workers'] = 2
    with FaceDatabase(config.database) as state:
        sid, _ = scan(state, config, '1', [A]*3)
        state.connection.execute('UPDATE observations SET embedding=? WHERE scan=?', (b'bad', sid))
        with pytest.raises(ValueError, match='embedding'):
            with compute.compute_scope(replace(config,data=settings)):
                local_cohorts.prepare(state.connection, [dict(scan=sid,scene='1')], replace(config,data=settings))
        assert state.connection.execute('SELECT COUNT(*) FROM members').fetchone()[0] == 0


def test_batched_late_ranking_preserves_exact_proofs_and_duplicate_owner_views(tmp_path, monkeypatch):
    from stash_ai.recognition.historical_references import History, rank, rank_many
    from stash_ai.recognition.view_prototypes import ViewPrototype
    from tests.test_recognition_score_blocks import groups, immutable
    monkeypatch.setattr(compute, 'available_cpus', lambda: 24)
    references, query = groups(128)
    # Keep one winner, several distinct banks belonging to one owner, and rivals.
    references[1].samples = tuple(immutable(-v) for v in references[1].samples)
    references[2].samples = tuple(immutable(-v) for v in references[2].samples)
    views = tuple(ViewPrototype(g,'other',g.identifier,(g.identifier,),3,'strict') for g in references[3:10])
    views[1].group.identifier = views[0].group.identifier
    history = History(tuple(references),views,{},'frozen-test')
    config = config_for(tmp_path)
    rows = [dict(id=str(i), dimensions=128, embedding=q.tobytes(), geometry='[]')
            for i,q in enumerate((query, immutable(-query), references[25].samples[0]))]
    expected = [rank([row],history,config) for row in rows]
    data = plain(config.data); data['execution'].update(rebuild_workers=4,rebuild_queue_bytes=1048576)
    with compute.compute_scope(replace(config,data=data)) as pool:
        actual = [result for _,result in rank_many(rows,history,config)]
        assert pool.submitted > 0
    assert actual == expected


def test_full_rebuild_threaded_cohorts_preserve_memberships_and_proofs(tmp_path, monkeypatch):
    from stash_ai.recognition.rebuild import rebuild_selected
    from tests.test_recognition_parallel_scores import semantic_state
    monkeypatch.setattr(compute, 'available_cpus', lambda: 24)
    config = config_for(tmp_path)
    with FaceDatabase(config.database) as state:
        selected = [dict(scene=str(i),scan=scan(state,config,str(i),v,count=len(v))[0])
                    for i,v in enumerate(([A]*3+[X]*3,[C]*3,[A,C,A,C]),1)]
        import sqlite3
        state.connection.commit()
        original = sqlite3.connect(':memory:')
        state.connection.backup(original)
        rebuild_selected(state,selected,config)
        expected = semantic_state(state.connection)
        original.backup(state.connection)
        original.close()
        data = plain(config.data); data['execution'].update(rebuild_workers=3,profile_rebuild=True)
        rebuild_selected(state,selected,replace(config,data=data))
        assert semantic_state(state.connection) == expected
        assert state.last_rebuild_profile['cpu']['records']['local_cohort_scenes']['consumed'] == 3
