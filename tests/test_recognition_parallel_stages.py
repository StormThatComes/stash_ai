"""Parallel numerical stages keep merge/context/binding decisions serial."""
from dataclasses import replace
from types import SimpleNamespace
import math
import uuid

import numpy as np
import pytest

from stash_ai.recognition import compute, rebuild
from stash_ai.recognition.clustering import cluster_scene
from stash_ai.recognition.config import plain
from stash_ai.recognition.database import FaceDatabase
from stash_ai.recognition.replay_checkpoint import ReplayClock
from stash_ai.recognition.replay_memory import memory_replay
from stash_ai.recognition.scoring import score_scope
from tests.test_recognition_clustering import setup, scan
from tests.test_recognition_parallel_scores import semantic_state


def library(state, config):
    basis = np.eye(128, dtype='float32')
    face = lambda x: basis[0]*x + basis[1]*math.sqrt(1-x*x)
    weak = basis[20]*.63 + basis[21]*math.sqrt(1-.63**2)
    vectors = [[v for v in basis[2:20] for _ in range(3)], [face(1.)]*3,
               [face(.4)]+[face(.75)]*4, [face(.9)]*3, [basis[20]]*3,
               [weak]+[basis[20]]*3, [weak]*3]
    selected = []
    for index, values in enumerate(vectors, 1):
        identifier, _ = scan(state, config, str(index), values, count=len(values))
        selected.append(dict(scene=str(index), scan=identifier))
    return selected


@pytest.mark.parametrize('workers', [2, 4])
def test_parallel_merge_context_and_binding_equal_serial(tmp_path, monkeypatch, workers):
    monkeypatch.setattr(compute, 'available_cpus', lambda: 24)
    config = setup(tmp_path, {'cohesion_threshold': .35})
    with FaceDatabase(config.database) as state:
        selected = library(state, config)
        rebuild.rebuild_selected(state, selected, config)
        expected = semantic_state(state.connection)
        # The fixture actually merges the fragment scenes and attaches the
        # borderline face only where a strict same-scene core exists.
        people = {r[0] for r in state.connection.execute(
            'SELECT m.person FROM members m JOIN observations o ON o.id=m.observation '
            'JOIN scans s ON s.id=o.scan WHERE s.scene IN (\'2\',\'3\',\'4\')')}
        assert len(people) == 1
        assert any('same_scene_identity' in row[-1] for row in expected['candidates'])
        state.connection.execute("DELETE FROM metadata WHERE key LIKE 'identity-rebuild%'")
        state.connection.commit()
        data = plain(config.data)
        data['execution'].update(rebuild_workers=workers, rebuild_queue_bytes=65536, profile_rebuild=True)
        parallel = replace(config, data=data)
        rebuild.rebuild_selected(state, selected, parallel)
        assert semantic_state(state.connection) == expected
        assert state.last_rebuild_profile['counts']['merge_pairs'] > 0
        assert state.last_rebuild_profile['counts']['parallel_score_blocks'] > 0
        # Remove vote origins in a disposable replay to exercise the optional
        # numerical historical-label fallback rather than just the vote path.
        with memory_replay(config) as (replay, _):
            rebuild._copy_selection(state, replay, selected, 'scope', 'ns')
            temporary = SimpleNamespace(connection=replay, identity_clock=ReplayClock(1),
                installation=str(uuid.uuid5(uuid.NAMESPACE_URL, 'parallel-binding-fixture')))
            for item in selected:
                cluster_scene(temporary, item['scan'], config, _cold_replay=True)
            replay.execute('DELETE FROM origins')
            replay.commit()
            baseline = rebuild._bind(state, replay, config, 'scope', 'ns')
            replay.execute('DROP TABLE votes')
            with compute.compute_scope(parallel) as pool, score_scope():
                actual = rebuild._bind(state, replay, parallel, 'scope', 'ns')
                assert pool.submitted > 0
            assert actual == baseline
