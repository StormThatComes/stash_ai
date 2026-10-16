"""Parallel native screening must preserve exact scalar identity decisions."""
from dataclasses import replace
from threading import Event, Lock, get_ident

import numpy as np
import pytest

from stash_ai.recognition import compute, parallel_scores
from stash_ai.recognition.clustering import Group, _similarities, cluster_scene, vector
from stash_ai.recognition.config import plain
from stash_ai.recognition.database import FaceDatabase
from stash_ai.recognition.rebuild import rebuild_selected
from tests.test_recognition_clustering import setup, scan
from tests.test_recognition_compute import configured
from tests.test_recognition_score_blocks import groups, immutable


@pytest.fixture(autouse=True)
def cpus(monkeypatch):
    monkeypatch.setattr(compute, 'available_cpus', lambda: 24)


@pytest.mark.parametrize('dimensions', [128, 512, 1024])
@pytest.mark.parametrize('threshold,cohesion', [(0., 0.), (.65, .6), (.99, .9)])
def test_batched_scores_never_discard_scalar_candidates(dimensions, threshold, cohesion):
    references, query = groups(dimensions)
    references[1].samples = list(references[0].samples)  # An exact identity tie.
    queries = (query, references[0].samples[0], immutable(-query))
    expected = [{g.identifier: _similarities(g, q) for g in references} for q in queries]
    with compute.compute_scope(configured(rebuild_workers=4, rebuild_queue_bytes=1048576)) as pool:
        plan = parallel_scores.FrozenScores(references)
        actual = plan.score(queries, threshold, cohesion)
        assert actual is not None and pool.submitted >= 2
        for index, values in enumerate(actual):
            found = {references[p].identifier: (mean, low) for p, mean, low in values}
            assert all(expected[index][name] == value for name, value in found.items())
            for name, (mean, low) in expected[index].items():
                if mean >= threshold and low >= cohesion:
                    assert found[name] == (mean, low)
        assert pool.peak_bytes <= pool.limit


def test_adjacent_thresholds_and_mutated_generation():
    references, query = groups(128)
    mean, low = _similarities(references[0], query)
    with compute.compute_scope(configured(rebuild_workers=4, rebuild_queue_bytes=1048576)):
        plan = parallel_scores.FrozenScores(references)
        for threshold in (np.nextafter(mean, -np.inf), mean, np.nextafter(mean, np.inf)):
            found = {p: (m, l) for p, m, l in plan.score((query,), threshold, low)[0]}
            if threshold <= mean:
                assert found[0] == (mean, low)
        references[0].samples[0] = immutable(-query)
        with pytest.raises(RuntimeError, match='generation changed'):
            plan.score((query,), mean)


def test_bad_vector_is_deferred_to_its_original_consumption():
    references, query = groups(128)
    rows = [dict(id='good', dimensions=128, embedding=query.tobytes()),
            dict(id='bad', dimensions=128, embedding=b'bad')]
    with compute.compute_scope(configured(rebuild_workers=4, rebuild_queue_bytes=1048576)):
        stream = parallel_scores.scored_rows(rows, references, .65)
        row, scores = next(stream)
        assert row['id'] == 'good' and scores is not None
        row, scores = next(stream)
        assert row['id'] == 'bad' and scores is None
        with pytest.raises(ValueError, match='embedding'):
            vector(row)


def semantic_state(db):
    return {
        'members': [tuple(r) for r in db.execute('SELECT * FROM members ORDER BY 1')],
        'people': [tuple(r) for r in db.execute('SELECT id,scope,namespace,performer FROM people ORDER BY id')],
        'candidates': [tuple(r) for r in db.execute('SELECT * FROM candidates ORDER BY id')],
        'merges': [tuple(r) for r in db.execute('SELECT source,target,evidence,status FROM identity_merges ORDER BY source')],
    }


def test_full_rebuild_matches_serial_with_local_growth_and_reversed_completion(tmp_path, monkeypatch):
    config = setup(tmp_path)
    # Enough immutable known groups to exercise actual worker jobs. Local groups
    # and weak observations then grow/change only on the coordinator.
    vectors = np.eye(128, dtype='float32')
    with FaceDatabase(config.database) as state:
        first, _ = scan(state, config, '1', [v for v in vectors[:18] for _ in range(3)], count=54)
        cluster_scene(state, first, config)
        sequence = [vectors[0]]*3 + [vectors[22]]*3 + [vectors[22]*.8+vectors[23]*.6]*3
        second, _ = scan(state, config, '2', sequence, count=len(sequence))
        selected = [dict(scene='1', scan=first), dict(scene='2', scan=second)]
        rebuild_selected(state, selected, config)
        expected = semantic_state(state.connection)
        state.connection.execute("DELETE FROM metadata WHERE key LIKE 'identity-rebuild%'")
        state.connection.commit()
        data = plain(config.data)
        data['execution'].update(rebuild_workers=4, rebuild_queue_bytes=65536, profile_rebuild=True)
        parallel = replace(config, data=data)
        original = parallel_scores._score
        arrived, lock = Event(), Lock()
        calls, threads = [], set()
        def reordered(payload):
            with lock:
                first_call = not calls
                calls.append(1); threads.add(get_ident())
            if first_call:
                assert arrived.wait(5)
            else:
                arrived.set()
            return original(payload)
        monkeypatch.setattr(parallel_scores, '_score', reordered)
        rebuild_selected(state, list(reversed(selected)), parallel)
        assert semantic_state(state.connection) == expected
        assert len(threads) >= 2
        assert state.last_rebuild_profile['counts']['parallel_score_blocks'] > 0
        # A different execution budget reuses semantic results without inference.
        rebuild_selected(state, selected, parallel)
        assert state.last_rebuild_profile['counts'].get('parallel_score_blocks', 0) == 0


def test_frozen_scores_combine_with_current_local_representatives():
    references, query = groups(128)
    with compute.compute_scope(configured(rebuild_workers=4, rebuild_queue_bytes=1048576)):
        values = parallel_scores.FrozenScores(references).score((query,), .65)[0]
        known = len(references)
        local = Group('local', [query], False)
        references.append(local)
        local.samples.append(immutable(-query))
        actual = list(parallel_scores.contributions(references, query, values, known, .65))
        result = next((mean, low) for g, mean, low in actual if g is local)
        assert result == _similarities(local, query)


def test_serial_input_stream_does_not_materialize_parallel_reference_owners(monkeypatch):
    def unexpected(*args):
        raise AssertionError('serial path packed the complete reference catalogue')
    monkeypatch.setattr(parallel_scores, 'FrozenScores', unexpected)
    with compute.compute_scope(configured()):
        source = iter([dict(id='one'), dict(id='two')])
        assert list(parallel_scores.scored_rows(source, [None]*20000, .65)) == [
            (dict(id='one'), None), (dict(id='two'), None)]
