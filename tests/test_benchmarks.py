"""Every documented offline entry point must run on small current evidence."""
import json
from pathlib import Path
import subprocess
import sys

import pytest


@pytest.mark.parametrize('name,arguments', [
    ('performance', ['--frames', '4', '--labels', '2', '--repeats', '2']),
    ('merge_pipeline', ['--frames', '4', '--labels', '2', '--repeats', '2']),
    ('marker_matching', ['--markers', '4', '--labels', '1', '--repeats', '2']),
    ('retained_merge', ['--frames', '4', '--labels', '2', '--weak-labels', '1', '--repeats', '2']),
    ('occurrences', ['--frames', '4', '--labels', '2', '--repeats', '2']),
    ('cleanup_sql', ['--markers', '20', '--legacy-time-budget', '.01']),
    ('tag_cascade', ['--markers', '20', '--batch-size', '7', '--repeats', '2']),
    ('rethreshold', ['--scenes', '2', '--frames', '4', '--labels', '2', '--repeats', '2']),
])
def test_offline_entry_point(name, arguments):
    result = subprocess.run([sys.executable, '-m', 'benchmarks.benchmark_' + name, *arguments],
                            cwd=Path(__file__).resolve().parents[1], capture_output=True,
                            text=True, timeout=60)
    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    assert payload
    if name == 'rethreshold':
        assert all(trial['evidence_unchanged'] and trial['evidence_writes'] == 0
                   for trial in payload['trials'])
    if name == 'tag_cascade':
        assert payload['identical_state']
        for trial in payload['trials']:
            assert trial['cascade']['state_sha256'] == trial['detach']['state_sha256']
            assert trial['cascade']['native_mutations'] == 1
            assert trial['detach']['native_mutations'] == 5
            assert trial['cascade']['remaining_rows'] == {
                'tags': 2, 'tags_image': 1, 'scenes': 1, 'scene_markers': 20,
                'images': 20, 'scene_markers_tags': 20, 'scenes_tags': 1, 'images_tags': 20}
