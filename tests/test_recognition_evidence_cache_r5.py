from contextlib import closing
import numpy as np


def test_valid_numpy_file_with_wrong_values_is_not_a_cache_hit(tmp_path):
    from stash_ai.recognition.solver_shared import Arrays, scope, key, numerical
    Arrays.prepare(tmp_path)
    source = np.asarray([1.,2.], dtype='<f4')
    with scope(tmp_path, 10000):
        assert list(numerical('proof', (source,), lambda: source*3)) == [3.,6.]
    np.save(tmp_path/(key('proof', (source,))+'.npy'), np.asarray([9.,9.], dtype='<f4'))
    with scope(tmp_path, 10000) as shared:
        assert list(numerical('proof', (source,), lambda: source*3)) == [3.,6.]
        assert shared.disabled


def test_exact_body_reductions_are_reused_by_a_fresh_evidence_reader(tmp_path, monkeypatch):
    from tests.test_recognition_body_scaling import fixture
    from stash_ai.recognition.body_global_search import Evidence
    from stash_ai.recognition.run_cache import scope
    from stash_ai.recognition import body_parallel
    config, state = fixture(tmp_path, [[1,0],[.8,.6]])
    try:
        rows = [dict(r) for r in state.connection.execute('SELECT id FROM body_observations ORDER BY scan,cue')]
        def reader():
            stats = dict(body_witness_comparisons=0, face_witness_comparisons=0,
                         candidate_pairs=0, score_cache_hits=0, embedding_reads=0)
            return Evidence(state.connection, config, 8*1024**2, stats)
        with scope(config):
            expected = reader().compare(rows[:3], rows[3:])
            def unexpected(*args):raise AssertionError('exact reduction repeated')
            monkeypatch.setattr(body_parallel, '_compare', unexpected)
            assert reader().compare(rows[:3], rows[3:]) == expected
    finally:
        state.__exit__(None,None,None)


def test_crop_artifacts_preserve_matching_generations_but_quality_invalidates(tmp_path):
    from tests.test_recognition_body_identity import configuration, scene
    from stash_ai.recognition.database import FaceDatabase
    from stash_ai.recognition.dependencies import version
    config = configuration(tmp_path)
    with FaceDatabase(config.database) as state:
        db = state.connection
        scene(db, config); db.commit()
        before = version(db)
        db.execute("UPDATE body_observations SET crop=?,quality=json_set(quality,'$.crop_status','retained','$.crop_attempt_limit',100)", (b'image',))
        assert version(db) == before
        db.execute("UPDATE body_observations SET quality=json_set(quality,'$.quality.keypoints',3)")
        assert version(db) > before


def test_exact_old_triggers_upgrade_and_still_invalidate_real_changes(tmp_path):
    from tests.test_recognition_body_identity import configuration, scene
    from stash_ai.recognition.database import FaceDatabase
    from stash_ai.recognition.dependency_schema import LEGACY_ARTIFACT_TRIGGERS, validate_triggers
    from stash_ai.recognition.dependencies import version
    config = configuration(tmp_path)
    with FaceDatabase(config.database) as state:
        scene(state.connection, config)
        for name, sql in LEGACY_ARTIFACT_TRIGGERS.items():
            state.connection.execute('DROP TRIGGER '+name)
            state.connection.execute(sql)
        state.connection.commit()
    with FaceDatabase(config.database) as state:
        validate_triggers(state.connection, required=True)
        before = version(state.connection)
        state.connection.execute('UPDATE body_observations SET crop=?', (b'crop',))
        assert version(state.connection) == before
        state.connection.execute('UPDATE body_observations SET embedding=zeroblob(16384)')
        assert version(state.connection) > before


def test_portrait_configuration_does_not_change_body_rebuild_key(tmp_path):
    from tests.test_recognition_body_identity import configuration
    from stash_ai.recognition.body_cache import key
    config = configuration(tmp_path)
    before = key([],config)
    config.data['portraits']['anchor_only'] = not config.data['portraits']['anchor_only']
    assert key([],config) == before
    config.data['sapiensid']['identities']['match_threshold'] -= .1
    assert key([],config) != before
