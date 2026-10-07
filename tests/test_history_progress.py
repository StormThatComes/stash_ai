"""Comparable selections, code provenance and bounded partial snapshots."""
from dataclasses import replace
import sqlite3
from threading import Thread

import pytest

from stash_ai.commands.arguments import parse_args
from stash_ai.history import lifecycle, provenance
from stash_ai.history.lifecycle import RunHistory, configuration
from stash_ai.history.store import HistoryStore, FORMAT
from stash_ai.reports.runs import build_report
from stash_ai.storage.store import scope_for
from tests.helpers.runtime import runtime as app

pytestmark = pytest.mark.usefixtures('reset_temporal')


def test_selection_order_and_duplicates_do_not_change_identity_but_other_ids_do():
    def config(ids):
        args = parse_args(app, ['--scene-id', ids[0]] + [value for identifier in ids[1:] for value in ('--scene-id', identifier)])
        return configuration(app, args)
    a = config(['1', '2', '1'])
    assert a['fingerprint'] == config(['2', '1'])['fingerprint']
    assert a['fingerprint'] != config(['3', '4'])['fingerprint']
    assert a['selected_scene_count'] == 2 and a['selection']['mode'] == 'explicit'
    assert configuration(app, parse_args(app, []))['selection']['mode'] == 'scope'


def test_code_and_package_provenance_change_comparison_identity(tmp_path, monkeypatch):
    root = tmp_path / 'app'
    (root / 'stash_ai').mkdir(parents=True)
    code = root / 'stash_ai' / 'module.py'
    code.write_text('VALUE = 1\n')
    local = replace(app, INSTALL_ROOT=root)
    args = parse_args(app, ['--merge-only'])
    monkeypatch.setattr(provenance.metadata, 'version', lambda name: '1')
    first = configuration(local, args)
    code.write_text('VALUE = 2\n')
    second = configuration(local, args)
    assert first['fingerprint'] != second['fingerprint']
    monkeypatch.setattr(provenance.metadata, 'version', lambda name: '2')
    assert second['fingerprint'] != configuration(local, args)['fingerprint']
    assert first['provenance']['revision'] is None


def test_partial_progress_is_bounded_and_survives_a_missing_finish(tmp_path, monkeypatch):
    path = tmp_path / 'state.sqlite3'
    args = parse_args(app, ['--merge-only', '--state-db', str(path)])
    ticks = [100.]
    monkeypatch.setattr(lifecycle.time, 'perf_counter', lambda: ticks[0])
    history = RunHistory(app, args).__enter__()
    try:
        history.progress()
        assert history.store.connection.execute('SELECT progress_seq FROM runs').fetchone()[0] == 0
        args.report.counts['processed'] = 32
        history.progress()
        args.report.counts['processed'] = 33
        history.progress()
        assert history.store.connection.execute('SELECT progress_seq FROM runs').fetchone()[0] == 1
        ticks[0] += 31
        history.progress()
        with HistoryStore(history.store.path, readonly=True) as reader:
            row = list(reader.rows(scope_for(app.STASH_URL)))[0]
        assert row['status'] == 'running' and row['partial'] and row['finished_at'] is None
        assert row['last_progress_at'] and row['progress_seq'] == 2
        assert row['counts']['processed'] == 33
    finally:
        history.close()  # Simulate missing process exit; never mark it finished.


def test_worker_threads_cannot_write_parent_progress(tmp_path, monkeypatch):
    args = parse_args(app, ['--merge-only', '--state-db', str(tmp_path / 'state.sqlite3')])
    with RunHistory(app, args) as history:
        history.progress_seconds = 0
        writes = []
        monkeypatch.setattr(history.store, 'progress', lambda *a, **k: writes.append(True))
        thread = Thread(target=history.progress)
        thread.start(); thread.join()
        assert not writes
        history.progress()
        assert writes == [True]


def test_progress_failure_remains_nonfatal(pipeline, monkeypatch):
    monkeypatch.setattr(RunHistory, 'progress_seconds', 0)
    monkeypatch.setattr(HistoryStore, 'progress', lambda *a, **k: (_ for _ in ()).throw(OSError('full')))
    assert app.main(pipeline.args) == 0
    assert pipeline.loaded and pipeline.stash.scene['scene_markers']


def test_load_failure_records_every_blocked_scene(pipeline, monkeypatch):
    from stash_ai.history.store import history_path
    scene = pipeline.stash.scene
    def scenes(*args, **kwargs):
        for identifier in ('1', '2', '3'):
            yield dict(scene, id=identifier)
    monkeypatch.setattr(pipeline.stash, 'iter_scenes', scenes)
    monkeypatch.setattr(app.services, 'load_tagger', lambda *a: (_ for _ in ()).throw(RuntimeError('no model')))
    args = [value for value in pipeline.args]
    # Remove the explicit single-scene selection.
    i = args.index('--scene-id'); del args[i:i+2]
    assert app.main(args) != 0
    parsed = parse_args(app, args)
    with HistoryStore(history_path(parsed), readonly=True) as store:
        rows = store.decisions(scope_for(app.STASH_URL), limit=1)
    assert {row['scene'] for row in rows} == {'1', '2', '3'}
    assert all(row['action'] == 'blocked' and 'model_load_failed' in row['reasons'] for row in rows)


def test_legacy_history_is_readable_without_changes_then_upgraded(tmp_path):
    path = tmp_path / 'history.sqlite3'
    with HistoryStore(path) as store:
        store.start('one', 'scope', 'merge_only', 'now', {'fingerprint': 'legacy'})
    with sqlite3.connect(path) as db:
        for column in ('last_progress_at', 'progress_seq', 'workload'):
            db.execute('ALTER TABLE runs DROP COLUMN '+column)
        db.execute('PRAGMA user_version=2')
    before = path.read_bytes()
    with HistoryStore(path, readonly=True) as store:
        row = list(store.rows('scope'))[0]
        assert row['workload'] == {} and row['progress_seq'] == 0 and row['partial']
    assert path.read_bytes() == before
    with HistoryStore(path) as store:
        assert store.connection.execute('PRAGMA user_version').fetchone()[0] == FORMAT


def test_partial_and_different_observed_workloads_have_separate_groups():
    def row(partial, cues):
        return dict(command='inference', status='running', partial=partial, configuration={'fingerprint': 'same'},
                    models={}, counts={}, outcomes={}, metrics=[], wall_seconds=1, cpu_seconds=1,
                    workload={'verified_cues': cues})
    assert len(build_report([row(True, 10), row(True, 20), row(False, 10)])['groups']) == 3
