"""Export failures preserve command results, reservations and truthful phases."""
from contextlib import contextmanager
import json
import subprocess
import sys

import pytest

from stash_ai.reports import io
from tests.helpers.runtime import runtime as app

pytestmark = pytest.mark.usefixtures('reset_progress')


@pytest.mark.parametrize('failure', ['serialize', 'invalid_json', 'write', 'flush'])
def test_failed_summary_export_leaves_no_final_and_same_path_can_be_retried(pipeline, tmp_path, monkeypatch, failure):
    destination = tmp_path/'outcome.json'
    original_dump, original_temp = io.json.dumps, io.tempfile.NamedTemporaryFile
    with monkeypatch.context() as patch:
        if failure in ('serialize', 'invalid_json'):
            def dumps(value, **kwargs):
                if isinstance(value, dict) and value.get('schema_version') == 1 and 'confirmed' in value:
                    if failure == 'serialize': raise OSError('serialization failed')
                    return '{"schema_version":'
                return original_dump(value, **kwargs)
            patch.setattr(io.json, 'dumps', dumps)
        else:
            @contextmanager
            def temporary(**kwargs):
                with original_temp(**kwargs) as stream:
                    write = stream.write
                    if failure == 'write':
                        def broken(payload):
                            write(payload[:8]); raise OSError('write failed')
                        stream.write = broken
                    else:
                        stream.flush = lambda: (_ for _ in ()).throw(OSError('flush failed'))
                    yield stream
            patch.setattr(io.tempfile, 'NamedTemporaryFile', temporary)
        assert app.main([*pipeline.args, '--run-summary-json', str(destination)]) == 0
    assert not destination.exists() and not list(tmp_path.glob('.outcome.json.*.tmp'))
    assert app.main([*pipeline.args, '--run-summary-json', str(destination)]) == 0
    assert json.loads(destination.read_text())['status'] == 'completed'


def test_original_exception_survives_export_failure(pipeline, tmp_path, monkeypatch):
    pipeline.fail['wd'] = 4
    destination = tmp_path/'interrupted.json'
    monkeypatch.setattr(io.ReservedJsonReport, 'publish', lambda *args: (_ for _ in ()).throw(OSError('export failed')))
    with pytest.raises(KeyboardInterrupt) as raised:
        app.main([*pipeline.args, '--run-summary-json', str(destination)])
    assert any('Outcome reporting failed' in note for note in raised.value.__notes__)
    assert not destination.exists()
    with io.ReservedJsonReport(destination): pass


def test_competing_reserved_or_created_destinations_are_preserved(tmp_path, monkeypatch):
    destination = tmp_path/'outcome.json'
    with io.ReservedJsonReport(destination) as export:
        with pytest.raises(RuntimeError, match='reserves'):
            with io.ReservedJsonReport(destination): pass
        original = io.os.link
        def race(source, target):
            destination.write_text('other writer')
            return original(source, target)
        monkeypatch.setattr(io.os, 'link', race)
        with pytest.raises(FileExistsError): export.publish({'status': 'completed'})
    assert destination.read_text() == 'other writer'
    assert not list(tmp_path.glob('.outcome.json.*.tmp'))


def test_crashed_reservation_does_not_leave_a_final_or_block_next_run(tmp_path):
    destination = tmp_path/'outcome.json'
    result = subprocess.run([sys.executable, '-c',
        'import os,sys; from stash_ai.reports.io import ReservedJsonReport; '
        'ReservedJsonReport(sys.argv[1]).__enter__(); os._exit(7)', str(destination)],
        timeout=10, capture_output=True)
    assert result.returncode == 7 and not destination.exists()
    with io.ReservedJsonReport(destination) as export:
        export.publish({'status': 'completed'})
    assert json.loads(destination.read_text()) == {'status': 'completed'}


@pytest.mark.parametrize('service,expected', [('load_model_settings', 'setup'), ('load_vocabulary', 'vocabulary setup'),
                                            ('create_stash_client', 'setup')])
def test_initialization_failures_report_setup_instead_of_finalize(pipeline, tmp_path, monkeypatch, service, expected):
    destination = tmp_path/'setup.json'
    def broken(*args, **kwargs): raise ValueError('setup failed')
    monkeypatch.setattr(app.services, service, broken)
    with pytest.raises(ValueError, match='setup failed'):
        app.main([*pipeline.args, '--run-summary-json', str(destination)])
    row = json.loads(destination.read_text())
    assert row['phase'] == expected and row['phases'][expected] == 'failed'
    assert 'finalize' not in row['phases'] and row['confirmed']['published_scenes'] == 0


@pytest.mark.parametrize('style', ['auto', 'off'])
def test_skipped_phases_have_bounded_explicit_reasons_in_both_styles(pipeline, tmp_path, style):
    destination = tmp_path/'skipped.json'
    assert app.main([*pipeline.args, '--merge-only', '--progress-style', style,
                     '--run-summary-json', str(destination)]) == 0
    row = json.loads(destination.read_text())
    assert row['phases']['inference'] == 'skipped'
    assert row['skipped_phase_reasons']['inference'] == 'merge-only'
    assert row['phases']['threshold update'] == 'skipped'
    assert row['skipped_phase_reasons']['threshold update'] == 'disabled'


def test_offline_review_redacts_error_details_without_changing_skip_counts(tmp_path, monkeypatch):
    from stash_ai.evidence import pipeline
    from stash_ai.reports.review.report import build_report
    from stash_ai.storage.store import StateStore
    from tests.test_result_thresholds import outputs
    def failed(*args, **kwargs): raise RuntimeError('https://host.test/?apikey=PRIVATE_TEST_TOKEN')
    monkeypatch.setattr(pipeline, 'evidence', failed)
    with StateStore(tmp_path/'state.sqlite', 'scope') as store:
        store.put('1', 'wd', outputs())
        row = build_report(app, store, ['1'], {}, {})
        assert 'PRIVATE_TEST_TOKEN' not in json.dumps(row)
        assert row['skipped_by_reason'] == {'unusable_evidence': 1}
        assert '[redacted]' in row['skipped_scene_examples'][0]['message']


def test_offline_report_console_failure_redacts_active_key(capsys, monkeypatch):
    from stash_ai.reports.runtime import offline_report
    monkeypatch.setattr(app, 'STASH_API_KEY', 'PRIVATE_TEST_TOKEN')
    @offline_report('Offline')
    def fail(app): raise ValueError('Authorization PRIVATE_TEST_TOKEN')
    assert fail(app) == 1
    assert 'PRIVATE_TEST_TOKEN' not in capsys.readouterr().out


def test_unavailable_console_still_exports_primary_interruption(pipeline, tmp_path, monkeypatch):
    from stash_ai.runtime.console import ConsolePresenter
    original = ConsolePresenter.message
    def message(self, text, **kwargs):
        if text.startswith('Outcome:'):
            self._abandon()
        return original(self, text, **kwargs)
    monkeypatch.setattr(ConsolePresenter, 'message', message)
    pipeline.fail['wd'] = 4
    destination = tmp_path/'console-unavailable.json'
    with pytest.raises(KeyboardInterrupt):
        app.main([*pipeline.args, '--run-summary-json', str(destination)])
    row = json.loads(destination.read_text())
    assert row['status'] == 'interrupted' and row['confirmed']['inference_frames'] == 4
    assert row['counts']['console_critical_unconfirmed'] >= 1
    assert row['counts']['outcome_console_failures'] >= 1
