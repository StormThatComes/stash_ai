"""Ordered support and contextual evidence over exact immutable CPU predicates."""
from threading import Event, get_ident

import numpy as np
import pytest

from stash_ai.recognition import compute, identity_merges as merges, parallel_merges
from stash_ai.recognition.clustering import _groups, vector
from stash_ai.recognition.database import FaceDatabase
from stash_ai.recognition.evidence_cache import evidence_scope, current as evidence
from tests.test_recognition_clustering import setup, scan
from tests.test_recognition_compute import configured
from tests.test_recognition_evidence_safety import pair_cues
from tests.test_recognition_identity_merges import changed, fragments
from tests.test_recognition_parallel_merges import references


@pytest.fixture(autouse=True)
def cpus(monkeypatch):
    monkeypatch.setattr(compute, 'available_cpus', lambda: 24)


def test_exact_masks_preserve_all_constituents_under_reversed_completion(tmp_path, monkeypatch):
    first, second = references()
    components = (first, second)
    config = changed(setup(tmp_path), recognition={'match_threshold': .98})
    rng = np.random.default_rng(98)
    queries = rng.normal(0, .025, (40, 128)).astype('float32')
    queries[:, 0] += 1
    queries /= np.linalg.norm(queries, axis=1)[:, None]
    queries[12] = -queries[0]
    rows = [dict(id=str(i), dimensions=128, embedding=q.tobytes()) for i, q in enumerate(queries)]
    expected = [sum(1 << i for i, component in enumerate(components)
                    if merges._matches(component, vector(row), config)) for row in rows]
    original, arrived, owner = parallel_merges._matches, Event(), get_ident()
    threads = set()
    def reordered(payload):
        assert evidence() is None and compute.current() is None
        threads.add(get_ident())
        if payload[1][0].tobytes() == rows[0]['embedding']:
            assert arrived.wait(5)
        else:
            arrived.set()
        return original(payload)
    monkeypatch.setattr(parallel_merges, '_matches', reordered)
    with compute.compute_scope(configured(rebuild_workers=4, rebuild_queue_bytes=65536)) as pool:
        actual = list(parallel_merges.matching_rows(rows, components, config))
        assert [row['id'] for row, _ in actual] == [row['id'] for row in rows]
        assert [mask for _, mask in actual] == expected
        assert pool.submitted > 1 and pool.peak_bytes <= pool.limit
    assert len(threads) > 1 and owner not in threads


def test_bad_vectors_are_validated_only_at_original_consumption(tmp_path):
    first, second = references()
    config = setup(tmp_path)
    good = first[0].samples[0]
    rows = [dict(id=str(i), dimensions=128, embedding=good.tobytes()) for i in range(32)]
    rows[4]['embedding'] = b'bad'
    with compute.compute_scope(configured(rebuild_workers=4, rebuild_queue_bytes=65536)) as pool:
        stream = parallel_merges.matching_rows(rows, (first, second), config)
        for _ in range(4):
            row, mask = next(stream)
            assert mask == 3
        row, mask = next(stream)
        assert row['id'] == '4' and mask is None
        with pytest.raises(ValueError, match='embedding'):
            vector(row)
        stream.close()
        # Closing an evidence prefix leaves no in-flight stream or poisoned token.
        assert not pool.pending
        assert list(parallel_merges.matching_rows(rows[:4], (first, second), config))


@pytest.mark.parametrize('superseded', [False, True])
def test_parallel_support_keeps_duplicate_times_latest_scans_and_exact_vote_prefix(tmp_path, superseded):
    config = setup(tmp_path)
    bridge = np.array([1.8, .6, 0, 0]); bridge /= np.linalg.norm(bridge)
    with FaceDatabase(config.database) as state:
        people, _ = fragments(state, config)
        groups = _groups(state.connection, 'scope', 'ns', config.data['clustering'])
        components = {g.identifier: [g] for g in groups}
        source, target = people[1], people[0]
        policy = merges._merge_support_policy(config)
        old, _ = scan(state, config, '10', [bridge]*32, count=32)
        selected, _ = scan(state, config, '11', [bridge]*32, count=32)
        for identifier in (old, selected):
            state.connection.execute('UPDATE observations SET seconds=CAST(cue/2 AS INT) WHERE scan=?', (identifier,))
            state.connection.execute('UPDATE observations SET embedding=? WHERE scan=? AND cue=1', (b'bad', identifier))
            state.connection.execute('INSERT INTO merge_support SELECT ?,?,id,? FROM observations WHERE scan=?',
                                     (source, target, policy, identifier))
        if superseded:
            scan(state, changed(config, detection={'min_sharpness': 1}), '10', [bridge]*3)
        state.connection.commit()
        head = state.connection.execute('SELECT *,rowid AS scan_order FROM scans WHERE id=?', (selected,)).fetchone()
        expected = merges._supporting_cues(state.connection, head, policy, source, target, components, config, 12)
        assert len(expected) == 12
        assert {vote['scene'] for vote in expected} == {'11' if superseded else '10'}
        with evidence_scope(state.connection, config), compute.compute_scope(
                configured(rebuild_workers=4, rebuild_queue_bytes=65536)) as pool:
            actual = merges._supporting_cues(state.connection, head, policy, source, target, components, config, 12)
            assert actual == expected
            assert pool.submitted > 0 and not pool.pending


@pytest.mark.parametrize('kind', ['distinct', 'duplicates', 'superseded'])
def test_contextual_unassigned_pairs_preserve_signatures_and_live_membership(tmp_path, monkeypatch, kind):
    config = setup(tmp_path)
    a, b = np.array([1., 0, 0, 0]), np.array([.8, .6, 0, 0])
    bridge = (a+b)/np.linalg.norm(a+b)
    with FaceDatabase(config.database) as state:
        people, old = fragments(state, config, [a, b])
        groups = _groups(state.connection, 'scope', 'ns', config.data['clustering'])
        components = {g.identifier: [g] for g in groups}
        values = [bridge]*40 if kind == 'duplicates' else [a, b]*20
        paired, _ = scan(state, config, '10', values, count=40)
        pair_cues(state, paired)
        if kind == 'superseded':
            scan(state, config, '10', [bridge], count=1)
        harmless, _ = scan(state, config, '11', [bridge]*40, count=40)
        pair_cues(state, harmless)
        head = state.connection.execute('SELECT * FROM scans WHERE id=?', (harmless,)).fetchone()
        expected = merges._cooccurs(state.connection, *people, components, head, config)
        assert expected is (kind == 'distinct')
        with evidence_scope(state.connection, config), compute.compute_scope(
                configured(rebuild_workers=4, rebuild_queue_bytes=65536)) as pool:
            assert merges._cooccurs(state.connection, *people, components, head, config) is expected
            assert pool.submitted > 0 and not pool.pending
            # Mutable membership must be checked before any contextual worker/cache.
            state.connection.execute('UPDATE observations SET scan=? WHERE scan=?', (old[0], old[1]))
            monkeypatch.setattr(parallel_merges, '_matches', lambda _: pytest.fail('live membership check skipped'))
            assert merges._cooccurs(state.connection, *people, components, head, config) is True


@pytest.mark.parametrize('failure', [RuntimeError, KeyboardInterrupt])
def test_evidence_worker_failure_releases_the_stream(tmp_path, monkeypatch, failure):
    first, second = references()
    rows = [dict(id=str(i), dimensions=128, embedding=first[0].samples[0].tobytes()) for i in range(32)]
    def fail(_):
        raise failure('evidence worker failed')
    monkeypatch.setattr(parallel_merges, '_matches', fail)
    with compute.compute_scope(configured(rebuild_workers=4, rebuild_queue_bytes=65536)) as pool:
        with pytest.raises(failure, match='evidence worker failed'):
            list(parallel_merges.matching_rows(rows, (first, second), setup(tmp_path)))
    assert not pool.pending and pool.native is None
