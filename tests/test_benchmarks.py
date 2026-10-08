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
    ('run_history', ['--frames', '8', '--labels', '2', '--repeats', '1']),
    ('pinned_proofs', ['--frames', '4', '--labels', '2', '--repeats', '1']),
    ('scoped_reports', ['--scenes', '20', '--selected', '2', '--repeats', '2']),
    ('state_compression', ['--frames', '4', '--labels', '2', '--repeats', '2']),
])
def test_offline_entry_point(name, arguments):
    result = subprocess.run([sys.executable, '-m', 'benchmarks.benchmark_' + name, *arguments],
                            cwd=Path(__file__).resolve().parents[1], capture_output=True,
                            text=True, timeout=60)
    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    assert payload
    if name == 'pinned_proofs':
        assert payload['identical_state']
        assert all(row['proofs_unchanged'] and row['archived_records'] == 8 for row in payload['trials'])
        assert all(row['source_unchanged'] for row in payload['trials'] if row['preview'])
        assert {row['decoded_snapshots'] for row in payload['trials'] if row['variant'] == 'baseline'} == {48}
        assert {row['decoded_snapshots'] for row in payload['trials'] if row['variant'] == 'candidate'} == {24}
    if name == 'state_compression':
        assert payload['identical_evidence']
        assert {(row['kind'], row['level']) for row in payload['groups']} == {
            (kind, level) for kind in ('scene', 'markers') for level in (1, 6)}
        for kind in ('scene', 'markers'):
            assert len({row['logical_sha256'] for row in payload['trials'] if row['kind'] == kind}) == 1
    if name == 'rethreshold':
        assert all(trial['evidence_unchanged'] and trial['evidence_writes'] == 0
                   for trial in payload['trials'])
    if name == 'run_history':
        assert payload['identical_evidence']
        assert all(row['history_bytes'] == 0 for row in payload['trials'] if not row['history'])
        assert all(row['retention']['removed_runs'] == 40 and row['retention']['proofs_unchanged']
                   for row in payload['trials'] if row['history'])
    if name == 'scoped_reports':
        assert all(trial['evidence_unchanged'] and trial['decoded_payloads'] == 0
                   for rows in payload['trials'].values() for trial in rows)
    if name == 'tag_cascade':
        assert payload['identical_state']
        for trial in payload['trials']:
            assert trial['cascade']['state_sha256'] == trial['detach']['state_sha256']
            assert trial['cascade']['native_mutations'] == 1
            assert trial['detach']['native_mutations'] == 5
            assert trial['cascade']['remaining_rows'] == {
                'tags': 2, 'tags_image': 1, 'scenes': 1, 'scene_markers': 20,
                'images': 20, 'scene_markers_tags': 20, 'scenes_tags': 1, 'images_tags': 20}
