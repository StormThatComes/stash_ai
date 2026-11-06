"""Auto retention stays visible through blocking SQL and atomic commit."""
from dataclasses import replace
from io import StringIO
import json
from threading import Event, current_thread

import pytest

from stash_ai.recognition import db_retention as retention
from stash_ai.recognition.config import plain
from stash_ai.recognition.database import FaceDatabase
from stash_ai.recognition.progress import observe_stages, observe_updates, stage
from stash_ai.recognition.run import execute
from stash_ai.runtime.console import ConsolePresenter, current_report, present
from stash_ai.runtime.report import RunReport
from tests.test_recognition_phase_progress import terminal
from tests.test_recognition_progress import args
from tests.test_recognition_retention import logical, scans, settings, successful
from tests.test_recognition_run import setup


def test_auto_prune_owns_live_rows_until_commit(tmp_path, monkeypatch, terminal):
    config, _, services, _ = setup(tmp_path, monkeypatch)
    config = replace(config, data=plain(config.data))
    config.data['storage']['retention']['auto_prune'] = True
    bars, owners = terminal
    stages, commits = [], []
    original = retention.on_connection

    def prune(db, *a, **kw):
        console = current_report().console
        assert console.phase_name == 'Recognition auto prune'
        assert not console.finalized
        result = original(db, *a, **kw)
        assert db.in_transaction
        assert next(b for b in bars if b.desc == 'Auto prune recognition').n == 2
        # Observe the real COMMIT, on the SQLite owner, before it finishes.
        def trace(sql):
            if sql == 'COMMIT':
                commits.append((next(b for b in bars if b.desc == 'Auto prune recognition').n,
                                console.state.get('section'), console.finalized))
        db.set_trace_callback(trace)
        return result

    monkeypatch.setattr(retention, 'on_connection', prune)
    with observe_stages(stages.append):
        assert execute(args('1', '2'), config, services=services) == 0
    rows = {b.desc: b for b in bars}
    assert commits == [(2, 'Commit automatic retention', False)]
    assert rows['Auto prune recognition'].n == rows['Auto prune recognition'].total == 3
    assert rows['Inspect retention scenes'].n == 2
    assert 'Prune obsolete recognition evidence' not in rows  # Empty plans skip SQL deletion passes.
    assert 'Protect references: publications.payload' in stages
    assert 'Clear obsolete crops: observations' not in stages
    assert all(b.disable for b in bars) and owners == {'stash-console'}
    with FaceDatabase(config.database) as db:
        saved = db.connection.execute("SELECT data FROM progress WHERE phase='retention'").fetchone()
        assert json.loads(saved[0])['applied']


@pytest.mark.parametrize('style', ['auto', 'compact', 'verbose'])
def test_redirected_run_heartbeats_while_sqlite_is_blocked(tmp_path, monkeypatch, style):
    config, _, services, _ = setup(tmp_path, monkeypatch)
    config = replace(config, data=plain(config.data))
    config.data['storage']['retention']['auto_prune'] = True
    stream, heartbeat = StringIO(), Event()
    monkeypatch.setattr('sys.stderr', stream)
    original, write = retention._apply, ConsolePresenter._write_status
    snapshots = []

    def status(self, text):
        write(self, text)
        if text.startswith('Still working:') and 'Waiting inside retention SQL' in text:
            snapshots.append((current_thread().name, text))
            heartbeat.set()

    def apply(db):
        console = current_report().console
        assert not console.finalized
        console.refresh = .01
        console.heartbeat = .01
        def wait():
            # A SQLite call is still in progress; only the console thread can
            # emit the heartbeat which releases this bounded wait.
            assert heartbeat.wait(3), 'No heartbeat during automatic retention SQL'
            return 1
        db.create_function('retention_wait', 0, wait)
        try:
            with stage('Waiting inside retention SQL'):
                assert db.execute('SELECT retention_wait()').fetchone()[0] == 1
        finally:
            db.create_function('retention_wait', 0, None)
        original(db)

    monkeypatch.setattr(ConsolePresenter, '_write_status', status)
    monkeypatch.setattr(retention, '_apply', apply)
    assert execute(args('1', style=style), config, services=services) == 0
    assert snapshots and all(owner == 'stash-console' for owner, _ in snapshots)
    assert all('Recognition auto prune' in text and 'Auto prune recognition: 1/3' in text
               for _, text in snapshots)
    assert '\x1b[' not in stream.getvalue()


@pytest.mark.parametrize('failure', [RuntimeError, KeyboardInterrupt])
@pytest.mark.parametrize('point', ['delete', 'commit'])
def test_failed_retention_rolls_back_without_completing_progress(tmp_path, failure, point):
    config = settings(tmp_path)
    config.data['storage']['retention']['auto_prune'] = True
    events = []
    report = RunReport(write=lambda _: None)
    with FaceDatabase(config.database) as state:
        scans(state, config)
        run = successful(state, config, 's3')
        before = logical(config.database)
        def observe(label):
            if label == ('Delete obsolete scans' if point == 'delete' else 'Commit automatic retention'):
                raise failure('injected retention failure')
        with present(report, 'auto'), observe_updates(lambda *event: events.append(event)), observe_stages(observe):
            with pytest.raises(failure, match='injected retention failure'):
                retention.after_run(state, config, run)
            assert not state.connection.in_transaction
            assert not state.connection.execute("SELECT 1 FROM sqlite_temp_master WHERE name LIKE 'retention_%'").fetchone()
            assert not report.console._bars
            assert 'work' not in report.console.state and 'detail' not in report.console.state
            assert report.console.phase_outcomes['Recognition auto prune'] == (
                'failed' if failure is RuntimeError else 'interrupted')
        progress = [data for kind, data in events if kind == 'work' and data['label'] == 'Auto prune recognition']
        assert progress[-1]['completed'] == (1 if point == 'delete' else 2)
        assert all(data['completed'] < data['total'] for data in progress)
        after=logical(config.database)
        saved=after['progress'].pop()
        assert saved[:2]==(run,'retention')
        detail=json.loads(saved[2])
        assert detail['outcome']==('failed' if failure is RuntimeError else 'interrupted')
        assert detail['applied'] is False and detail['error_type']==failure.__name__
        assert after==before


def test_progress_off_and_json_stay_quiet(tmp_path, monkeypatch, capsys):
    config, _, services, _ = setup(tmp_path, monkeypatch)
    config = replace(config, data=plain(config.data))
    config.data['storage']['retention']['auto_prune'] = True
    assert execute(args('1', style='off'), config, services=services) == 0
    output = capsys.readouterr()
    assert output.err == '' and 'Recognition retention: applied' in output.out
    from types import SimpleNamespace
    assert retention.execute(SimpleNamespace(progress_style='verbose', json=True, apply=False,
                                             scope=None, scene_id=None), config) == 0
    output = capsys.readouterr()
    assert output.err == '' and not json.loads(output.out)['applied']


def test_disabled_auto_prune_does_not_open_maintenance_rows(tmp_path, monkeypatch, terminal):
    config, _, services, _ = setup(tmp_path, monkeypatch)
    assert execute(args('1'), config, services=services) == 0
    assert not any(b.desc == 'Auto prune recognition' for b in terminal[0])
