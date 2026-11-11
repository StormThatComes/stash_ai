"""Profile real thread/SQLite boundaries without changing recognition ownership."""
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
import json
from threading import Barrier

import pytest

from stash_ai.recognition.profiling import (
    pipeline_bind, pipeline_count, pipeline_scope, pipeline_snapshot, pipeline_stage,
)


def test_disabled_profile_does_not_read_clock_or_capture_tasks(monkeypatch):
    from stash_ai.recognition import profiling
    @pipeline_stage('work')
    def work():
        pipeline_count('items')
        return 7
    monkeypatch.setattr(profiling.time, 'perf_counter', lambda: pytest.fail('disabled profiler read clock'))
    with pipeline_scope(False):
        assert pipeline_bind(work) is work
        assert work() == 7
        assert pipeline_snapshot() is None


def test_thread_measurements_are_isolated_atomic_and_snapshots_detached():
    barrier = Barrier(3)
    @pipeline_stage('work')
    def work():
        barrier.wait(timeout=5)
        for _ in range(500):
            pipeline_count('items')
        return 7
    with ThreadPoolExecutor(2) as pool:
        with pipeline_scope(True):
            jobs = [pool.submit(pipeline_bind(work)) for _ in range(2)]
            barrier.wait(timeout=5)
            assert [job.result(timeout=5) for job in jobs] == [7, 7]
            with pipeline_scope(True):
                assert pipeline_snapshot()['counts'] == {}
            snapshot = pipeline_snapshot()
            assert snapshot['stages'] == {}
            assert snapshot['worker_stages']['work']['calls'] == 2
            assert snapshot['worker_stages']['work']['errors'] == 0
            assert snapshot['counts']['items'] == 1000
            snapshot['counts']['items'] = -1
            snapshot['worker_stages']['work']['errors'] = 100
            assert pipeline_snapshot()['counts']['items'] == 1000
            assert pipeline_snapshot()['worker_stages']['work']['errors'] == 0
            assert pool.submit(pipeline_snapshot).result(timeout=5) is None
        assert pipeline_snapshot() is None


@pytest.mark.parametrize('fail_at', [None, 2])
def test_prefetched_iterator_end_is_not_an_error_but_source_failure_is(fail_at):
    from stash_ai.recognition.prefetch import FramePrefetch
    from tests.test_recognition_prefetch import Frames
    source = Frames(3, fail_at=fail_at)
    with pipeline_scope(True):
        stream = FramePrefetch(source, 0, 2).launch()
        pixels = []
        def consume():
            for index, _, image in stream:
                with image:
                    pixels.append(image.getpixel((0, 0))[0])
        try:
            if fail_at is None:
                consume()
            else:
                with pytest.raises(ValueError, match='source failed'):
                    consume()
        finally:
            stream.close()
        metrics = pipeline_snapshot()
    assert pixels == list(range(3 if fail_at is None else fail_at))
    assert metrics['stages']['frame_input_wait']['errors'] == int(fail_at is not None)
    assert metrics['worker_stages']['frame_queue_put']['calls'] == len(pixels)
    assert source.closed.is_set()


@pytest.mark.parametrize('retention_outcome', ['success', 'failed', 'interrupted'])
def test_run_history_contains_worker_io_and_final_retention_profile(tmp_path, monkeypatch, retention_outcome):
    from stash_ai.recognition.config import plain
    from stash_ai.recognition.database import FaceDatabase
    from stash_ai.recognition import db_retention
    from tests.test_recognition_run import setup, run
    config, _, services, _ = setup(tmp_path, monkeypatch)
    data = plain(config.data)
    data['execution'].update(profile_rebuild=True, scan_cpu_workers=2, prefetch_frames=2)
    data['storage']['retention']['auto_prune'] = True
    config = replace(config, data=data)
    if retention_outcome != 'success':
        def fail(*args, **kwargs):
            if retention_outcome == 'interrupted':
                raise KeyboardInterrupt
            raise ValueError('synthetic retention failure')
        monkeypatch.setattr(db_retention, 'on_connection', fail)
    if retention_outcome == 'interrupted':
        with pytest.raises(KeyboardInterrupt):
            run(config, services, '1')
    else:
        assert run(config, services, '1') == int(retention_outcome == 'failed')
    with FaceDatabase(config.database) as state:
        row = state.connection.execute('SELECT status,summary FROM runs').fetchone()
        assert row['status'] == 'success'  # Retention has a separate outcome.
        summary = json.loads(row['summary'])
        profile = summary['pipeline_profile']
        assert profile['format'] == 2
        assert profile['cutoff'] == 'after_auto_prune_before_profile_write'
        assert profile['retention_outcome'] == retention_outcome
        assert profile['stages']['auto_prune']['errors'] == int(retention_outcome != 'success')
        assert profile['elapsed_seconds'] >= profile['stages']['auto_prune']['seconds']
        assert profile['stages']['frame_input_wait']['errors'] == 0
        assert profile['stages']['face_detection']['calls'] == 3
        assert profile['stages']['face_model_load']['calls'] == 1
        assert profile['worker_stages']['frame_materialize']['calls'] == 3
        assert profile['worker_stages']['sprite_decode']['calls'] > 0
        assert profile['worker_stages']['media_read']['calls'] > 0
        assert profile['worker_stages']['scan_sqlite_write']['calls'] > 0
        assert profile['worker_stages']['pixel_prepare']['calls'] > 0
        assert profile['counts']['media_http_requests'] > 0
        assert profile['counts']['media_downloaded_bytes'] > 0
        assert 'http://' not in json.dumps(profile)
        assert state.connection.execute('SELECT COUNT(*) FROM observations').fetchone()[0] == 3
    assert pipeline_snapshot() is None


def test_body_enrichment_times_direct_frame_input_and_atomic_checkpoint(tmp_path):
    from types import SimpleNamespace
    from stash_ai.recognition.person_collection import PersonCollector
    from tests.test_recognition_scanning import config as settings
    from stash_ai.recognition.config import plain
    from stash_ai.recognition.database import FaceDatabase
    from tests.test_recognition_clustering import scan
    from tests.test_recognition_prefetch import Frames
    config = settings(tmp_path)
    data = plain(config.data)
    data['sapiensid']['enabled'] = True
    config = replace(config, data=data)
    collector = PersonCollector(config, {}, factory=lambda *args: SimpleNamespace(close=lambda: None))
    collector.missing_cues = lambda *args: {0, 1, 2}
    # Exercise the actual direct frame loop and complete-cue checkpoint without models.
    collector.collect_records = lambda db,scan,index,image,rows,**kwargs: None
    source = Frames(3)
    source.cues = [None]*3
    with FaceDatabase(config.database) as state:
        from tests.test_recognition_clustering import setup
        face_config = setup(tmp_path)
        identifier, _ = scan(state, face_config, '1')
        # Auxiliary recovery/crop work is independent of the empty synthetic collector.
        from unittest.mock import patch
        with patch('stash_ai.recognition.future_crowd.recover'), patch('stash_ai.recognition.person_crop_backfill.backfill'):
            with pipeline_scope(True):
                assert collector.enrich(state.connection, identifier, source) == 3
                profile = pipeline_snapshot()
        assert profile['stages']['body_enrichment']['calls'] == 1
        assert profile['stages']['body_sqlite_checkpoint']['calls'] == 1
    collector.close()


def test_sam_and_pose_timings_preserve_masks_points_and_predictor_cleanup():
    pytest.importorskip('torch')
    import numpy as np
    from PIL import Image
    from stash_ai.recognition.sam_pose2seg import Segmenter
    from stash_ai.recognition.pose_refinement import RefinedPMPose
    from stash_ai.recognition.pose_refinement_config import defaults
    from tests.test_recognition_pose_refinement import BOX, Pose
    class Predictor:
        def __init__(self):
            self.active = False
        def set_image(self, pixels):
            self.shape = pixels.shape[:2]
            self.active = True
        def predict(self, **kwargs):
            assert self.active
            mask = np.zeros((1, *self.shape), dtype=bool)
            mask[:, 10:90, 20:80] = True
            return mask, np.array([.95]), None
        def reset_predictor(self):
            self.active = False
    predictor = Predictor()
    model = RefinedPMPose(Pose(), Segmenter(predictor, 'cpu'), defaults())
    with Image.new('RGB', (160, 100)) as image:
        expected = model.predict(image, [BOX])
        with pipeline_scope(True):
            actual = model.predict(image, [BOX])
            profile = pipeline_snapshot()
    for index in (0, 1):
        np.testing.assert_array_equal(actual[index], expected[index])
    for key in expected[2]:
        np.testing.assert_array_equal(actual[2][key], expected[2][key])
    assert actual[3] == expected[3]
    assert not predictor.active
    for name in ('pmpose_first_pass', 'sam_image_embedding', 'sam_mask_prediction',
                 'mask_validation', 'pmpose_masked_second_pass'):
        assert profile['stages'][name]['calls'] == 1
        assert profile['stages'][name]['errors'] == 0
