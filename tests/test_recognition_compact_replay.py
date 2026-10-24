import json
import sqlite3

from stash_ai.recognition.clustering import cluster_scene, matching_policy
from stash_ai.recognition.database import FaceDatabase
from stash_ai.recognition.rebuild import _copy_selection, rebuild_selected, sample_key
from tests.test_recognition_clustering import setup, scan


def test_scratch_drops_unused_quality_and_pixels_without_changing_sample_identity(tmp_path):
    config = setup(tmp_path)
    with FaceDatabase(config.database) as database:
        identifier, _ = scan(database, config, '1')
        selected = [{'scene': '1', 'scan': identifier}]
        original = database.connection.execute('SELECT o.*,s.scope,s.namespace,s.scene,s.input_sha '
            'FROM observations o JOIN scans s ON s.id=o.scan ORDER BY o.id').fetchall()
        with sqlite3.connect(':memory:') as replay:
            replay.row_factory = sqlite3.Row
            _copy_selection(database, replay, selected, 'scope', 'ns')
            for row in replay.execute('SELECT * FROM observations ORDER BY id'):
                source = next(item for item in original if item['id'] == row['id'])
                assert set(json.loads(row['quality'])) == {'pose_deviation','pixels','sharpness','roll_deviation','alignment_valid'}
                assert row['crop'] is None
                assert row['source'] == '' and row['rect'] == '[]'
                assert row['geometry'] == source['geometry'] and row['embedding'] == source['embedding']
                assert replay.execute('SELECT sample_key FROM sample_keys WHERE observation=?',
                                      (row['id'],)).fetchone()[0] == sample_key(source)
        rebuild_selected(database, selected, config)
        current = database.connection.execute('SELECT * FROM observations ORDER BY id').fetchall()
        assert [row['quality'] for row in current] == [row['quality'] for row in original]
        payloads = [json.loads(row[0]) for row in database.connection.execute('SELECT payload FROM candidates')]
        assert payloads and all(payload['reference_context'] for payload in payloads)


def test_pose_aware_replay_retains_only_pose_not_full_quality(tmp_path):
    config = setup(tmp_path, {'max_anchor_pose_deviation': 0.2})
    with FaceDatabase(config.database) as database:
        identifier, _ = scan(database, config, '1')
        selected = [{'scene': '1', 'scan': identifier}]
        with sqlite3.connect(':memory:') as replay:
            replay.row_factory = sqlite3.Row
            _copy_selection(database, replay, selected, 'scope', 'ns', retain_pose=True)
            rows = replay.execute('SELECT quality,crop FROM observations').fetchall()
            assert rows
            assert all(json.loads(row['quality'])['pose_deviation'] == 0.0
                       and set(json.loads(row['quality'])) == {'pose_deviation','pixels','sharpness','roll_deviation','alignment_valid'}
                       and row['crop'] is None for row in rows)


def test_normal_incremental_matching_still_records_reusable_context(tmp_path):
    config = setup(tmp_path)
    with FaceDatabase(config.database) as database:
        identifier, _ = scan(database, config, '1')
        result = cluster_scene(database, identifier, config)
        before = [tuple(row) for row in database.connection.execute('SELECT * FROM candidates')]
        assert cluster_scene(database, identifier, config) == result
        assert before == [tuple(row) for row in database.connection.execute('SELECT * FROM candidates')]
        assert database.connection.execute('SELECT COUNT(*) FROM candidates WHERE policy=? '
            "AND json_extract(payload,'$.reference_context') IS NOT NULL", (matching_policy(config),)).fetchone()[0] == 3
