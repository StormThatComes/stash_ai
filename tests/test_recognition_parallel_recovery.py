"""Only applied strict scenes survive interruption or CPU worker failure."""
from dataclasses import replace

import pytest

from stash_ai.recognition import compute, parallel_scores, rebuild
from stash_ai.recognition.config import plain
from stash_ai.recognition.database import FaceDatabase
from tests.test_recognition_clustering import setup
from tests.test_recognition_parallel_scores import semantic_state
from tests.test_recognition_parallel_stages import library


@pytest.mark.parametrize('failure', ['interrupt', 'worker'])
def test_resume_after_parallel_failure_with_different_worker_budget(tmp_path, monkeypatch, failure):
    monkeypatch.setattr(compute, 'available_cpus', lambda: 24)
    original = setup(tmp_path, {'cohesion_threshold': .35})
    data = plain(original.data)
    data['execution'].update(rebuild_workers=4, rebuild_queue_bytes=65536,
                             rebuild_checkpoint_scenes=1, profile_rebuild=True)
    config = replace(original, data=data)
    with FaceDatabase(config.database) as state:
        selected = library(state, config)
        before = semantic_state(state.connection)
        cluster, score = rebuild.cluster_scene, parallel_scores._score
        calls = []
        def failing_cluster(*args, **kwargs):
            calls.append(args[1])
            if len(calls) == 4 and failure == 'interrupt':
                raise KeyboardInterrupt('interrupted after strict prefix')
            return cluster(*args, **kwargs)
        def failing_score(payload):
            if len(calls) == 4:
                raise ValueError('failed CPU worker after strict prefix')
            return score(payload)
        with monkeypatch.context() as patch:
            patch.setattr(rebuild, 'cluster_scene', failing_cluster)
            if failure == 'worker':
                patch.setattr(parallel_scores, '_score', failing_score)
            with pytest.raises(KeyboardInterrupt if failure == 'interrupt' else ValueError):
                rebuild.rebuild_selected(state, selected, config)
        assert semantic_state(state.connection) == before
        # Execution knobs do not invalidate the semantic checkpoint descriptor.
        data = plain(config.data)
        data['execution']['rebuild_workers'] = 2
        resumed = replace(config, data=data)
        rebuild.rebuild_selected(state, selected, resumed)
        assert state.last_rebuild_profile['counts']['checkpoint_resumed_scenes'] == 3
        cpu = state.last_rebuild_profile['cpu']
        assert cpu['effective_workers'] == 2
        assert cpu['submitted_blocks'] == cpu['applied_blocks'] > 0
        assert 0 < cpu['peak_queued_bytes'] <= cpu['queue_limit_bytes']
        assert cpu['successful_worker_wall_seconds'] > 0
        assert cpu['ordered_wait_seconds'] >= 0
        expected = semantic_state(state.connection)
        state.connection.execute("DELETE FROM metadata WHERE key LIKE 'identity-rebuild%'")
        state.connection.commit()
        rebuild.rebuild_selected(state, selected, original)
        assert semantic_state(state.connection) == expected
