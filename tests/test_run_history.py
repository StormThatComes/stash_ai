"""Durable telemetry, comparable averages and unchanged inference/preview behavior."""
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import sqlite3

import pytest

from stash_ai.history.store import HistoryStore
from stash_ai.runtime.profile import ProfileMetrics
from stash_ai.runtime.report import RunReport
from stash_ai.storage.store import scope_for
from tests.helpers.runtime import runtime as app

pytestmark = pytest.mark.usefixtures('reset_progress')


def history_rows(path):
    with HistoryStore(Path(str(path) + '.runs.sqlite3'), readonly=True) as store:
        return list(store.rows(scope_for(app.STASH_URL)))


def test_fresh_and_cached_runs_are_durable_and_report_is_entirely_offline(pipeline, monkeypatch, tmp_path):
    assert app.main(pipeline.args) == 0
    loaded = len(pipeline.loaded)
    assert app.main(pipeline.args) == 0
    assert len(pipeline.loaded) == loaded
    cached, fresh = history_rows(pipeline.path)
    assert fresh['outcomes']['wd']['fresh'] == 1
    assert cached['outcomes']['wd']['reused'] == 1
    assert fresh['counts']['processed'] == cached['counts']['reused'] == 1
    assert fresh['models']['wd']['revision']
    batches = [row for row in fresh['metrics'] if row['name'] == 'inference_batch']
    assert sum(row['units'] for row in batches) == len(pipeline.cues)
    assert {row['work'] for row in batches} == {'fresh'}
    assert not any(row['name'] == 'inference_batch' for row in cached['metrics'])
    assert any(row['name'] == 'model_scene' and row['work'] == 'reused' for row in cached['metrics'])
    before = Path(str(pipeline.path) + '.runs.sqlite3').read_bytes()
    for service in ('create_stash_client', 'load_tagger', 'load_vocabulary', 'load_model_settings'):
        monkeypatch.setattr(app.services, service, lambda *a, **k: pytest.fail('history report accessed Stash/models'))
    destination = tmp_path / 'history.json'
    assert app.main(['--run-report', str(destination), '--state-db', str(pipeline.path), '--run-model', 'wd']) == 0
    report = json.loads(destination.read_text())
    assert len(report['runs']) == 2
    assert report['groups']
    assert Path(str(pipeline.path) + '.runs.sqlite3').read_bytes() == before


def test_interruption_then_checkpoint_resume_keeps_separate_profiles(pipeline):
    pipeline.fail['wd'] = 4
    with pytest.raises(KeyboardInterrupt):
        app.main(pipeline.args)
    assert history_rows(pipeline.path)[0]['status'] == 'interrupted'
    pipeline.fail.clear()
    assert app.main(pipeline.args) == 0
    resumed, interrupted = history_rows(pipeline.path)
    assert resumed['outcomes']['wd']['checkpoint_resumed'] == 1
    rows = [row for row in resumed['metrics'] if row['name'] == 'inference_batch']
    assert {row['work'] for row in rows} == {'checkpoint_resumed'}
    assert sum(row['units'] for row in rows) == len(pipeline.cues) - 4
    assert any(row['name'] == 'inference_batch' and row['outcome'] == 'error' for row in interrupted['metrics'])


@pytest.mark.parametrize('flag', ['--dry-run', '--no-run-history'])
def test_preview_and_opt_out_do_not_create_history(pipeline, flag):
    assert app.main([*pipeline.args, flag]) == 0
    assert not Path(str(pipeline.path) + '.runs.sqlite3').exists()


def test_programmatic_preview_even_with_record_run_never_creates_history(pipeline, monkeypatch):
    monkeypatch.setattr(app, 'DRY_RUN', True)
    assert app.main([*pipeline.args, '--record-run']) == 0
    assert not Path(str(pipeline.path) + '.runs.sqlite3').exists()


@pytest.mark.parametrize('result,exception,status', [(0, None, 'finished'), (1, None, 'error'),
                                                   (None, ValueError, 'error'), (None, KeyboardInterrupt, 'interrupted')])
def test_command_start_is_committed_before_work_and_failures_are_finished(tmp_path, monkeypatch, result, exception, status):
    from stash_ai import cli
    path = tmp_path / 'state.sqlite3'
    def execute(runtime, args):
        assert history_rows(path)[0]['status'] == 'running'
        args.report.issue('read', 'wd', '123', ValueError(), 'SECRET_PAYLOAD')
        if exception:
            raise exception('SECRET_PAYLOAD')
        return result
    monkeypatch.setattr(cli, '_execute', execute)
    if exception:
        with pytest.raises(exception):
            app.main(['--state-db', str(path)])
    else:
        assert app.main(['--state-db', str(path)]) == result
    row = history_rows(path)[0]
    assert row['status'] == status and row['finished_at']
    assert row['issues']['read/wd/ValueError'] == 1
    assert 'SECRET_PAYLOAD' not in json.dumps(row)


def test_unavailable_history_does_not_break_successful_inference(pipeline, monkeypatch):
    from stash_ai.history import lifecycle
    monkeypatch.setattr(lifecycle, 'HistoryStore', lambda *a, **k: (_ for _ in ()).throw(sqlite3.OperationalError('full')))
    assert app.main(pipeline.args) == 0
    assert pipeline.loaded and pipeline.stash.scene['scene_markers']


def test_onnx_cuda_metadata_failure_does_not_fail_model_loading(pipeline, monkeypatch):
    import sys
    from types import SimpleNamespace
    original = app.services.load_tagger
    def load(*args):
        tagger = original(*args)
        tagger.device = 'cuda:0'
        tagger.session = SimpleNamespace(get_providers=lambda: ['CUDAExecutionProvider'])
        return tagger
    monkeypatch.setattr(app.services, 'load_tagger', load)
    monkeypatch.setitem(sys.modules, 'torch', SimpleNamespace(cuda=SimpleNamespace(
        get_device_name=lambda *a: (_ for _ in ()).throw(RuntimeError('CUDA unavailable in Torch')))))
    assert app.main(pipeline.args) == 0
    model = history_rows(pipeline.path)[0]['models']['wd']
    assert model['effective_device'] == 'cuda:0'
    assert model['providers'] == ['CUDAExecutionProvider']
    assert model['gpu_metadata_error'] == 'RuntimeError'
    assert pipeline.stash.scene['scene_markers']


def test_finish_write_failure_does_not_mask_command_error(pipeline, monkeypatch):
    monkeypatch.setattr(HistoryStore, 'finish', lambda *a, **k: (_ for _ in ()).throw(OSError('full')))
    pipeline.fail['wd'] = 4
    with pytest.raises(KeyboardInterrupt):
        app.main(pipeline.args)
    assert history_rows(pipeline.path)[0]['status'] == 'running'


@pytest.mark.parametrize('operation', ['start', 'finish', 'record_decision', 'close'])
@pytest.mark.parametrize('interrupted', [False, True])
def test_history_lifecycle_failures_preserve_inference_and_interrupt(pipeline, monkeypatch, operation, interrupted):
    original_close = HistoryStore.close
    def failing_close(store):
        original_close(store)
        raise OSError('private close payload')
    def fail(*args, **kwargs):
        raise OSError('private write payload')
    monkeypatch.setattr(HistoryStore, operation, failing_close if operation == 'close' else fail)
    if operation in ('start', 'record_decision'):
        monkeypatch.setattr(HistoryStore, 'close', failing_close)
    if interrupted:
        pipeline.fail['wd'] = 4
        with pytest.raises(KeyboardInterrupt):
            app.main(pipeline.args)
    else:
        assert app.main(pipeline.args) == 0
        loaded = len(pipeline.loaded)
        assert app.main(pipeline.args) == 0
        assert len(pipeline.loaded) == loaded


def test_history_close_and_warning_failures_preserve_original_exception(tmp_path, monkeypatch):
    from stash_ai.commands.arguments import parse_args
    from stash_ai.history.lifecycle import RunHistory
    args = parse_args(app, ['--state-db', str(tmp_path / 'state.sqlite3')])
    with pytest.raises(ValueError, match='original command failure'):
        with RunHistory(app, args) as history:
            monkeypatch.setattr(history.store, 'close', lambda: (_ for _ in ()).throw(OSError()))
            monkeypatch.setattr(args.report, 'write', lambda *a: (_ for _ in ()).throw(OSError()))
            raise ValueError('original command failure')


def test_read_only_commands_record_only_when_explicitly_requested(tmp_path, monkeypatch):
    from stash_ai import cli
    path = tmp_path / 'absent-state.sqlite3'
    monkeypatch.setattr(cli, '_execute', lambda *a: 0)
    arguments = ['--db-report', '--state-db', str(path)]
    assert app.main(arguments) == 0
    assert not Path(str(path) + '.runs.sqlite3').exists()
    assert app.main([*arguments, '--record-run']) == 0
    assert history_rows(path)[0]['command'] == 'db_report'
    assert not path.exists()


def test_missing_history_report_does_not_create_db_or_output(tmp_path):
    destination = tmp_path / 'report.json'
    before = set(tmp_path.iterdir())
    with pytest.raises(sqlite3.OperationalError):
        app.main(['--run-report', str(destination), '--state-db', str(tmp_path / 'missing.sqlite3')])
    assert set(tmp_path.iterdir()) == before


def test_foreign_history_db_is_not_modified(tmp_path):
    path = tmp_path / 'foreign.sqlite3'
    with sqlite3.connect(path) as db:
        db.execute('CREATE TABLE user_data(value TEXT)')
        db.execute("INSERT INTO user_data VALUES ('keep')")
    before = path.read_bytes()
    with pytest.raises(ValueError, match='history DB'):
        HistoryStore(path)
    assert path.read_bytes() == before


@pytest.mark.parametrize('extra', [['--run-limit', '0'], ['--run-limit', '10001'], ['--scene-id', '1'],
                                  ['--record-run'], ['--models', 'wd'], ['--restart'], ['--dry-run']])
def test_history_report_rejects_inapplicable_options(tmp_path, extra):
    with pytest.raises(SystemExit):
        app.main(['--run-report', str(tmp_path / 'new.json'), *extra])


def test_history_cannot_use_state_db_as_destination(tmp_path):
    path = tmp_path / 'state.sqlite3'
    with pytest.raises(SystemExit):
        app.main(['--state-db', str(path), '--run-history-db', str(path)])
    assert not path.exists()


def test_weighted_means_and_status_and_configuration_groups():
    from stash_ai.reports.runs import build_report
    def row(samples, *, status='finished', config='one'):
        profile = ProfileMetrics()
        for value in samples:
            profile.record('stage', 'inference_batch', value, value, model='wd', work='fresh', units=4, unit='frames')
        return dict(command='inference', status=status, configuration={'fingerprint': config}, models={'wd': {}},
                    wall_seconds=sum(samples), cpu_seconds=0.0, counts={}, outcomes={}, metrics=profile.export())
    report = build_report([row([1]), row([3, 5, 7]), row([100], status='error'), row([40], config='two')])
    assert len(report['groups']) == 3
    group = report['groups'][0]
    assert group['runs'] == 2 and group['mean_wall_seconds'] == 8
    metric = group['metrics'][0]
    assert metric['calls'] == 4 and metric['mean_seconds'] == 4
    assert metric['units_per_second'] == 1
    assert 1 <= metric['p50_seconds'] <= metric['p95_seconds'] <= 7


def test_latency_aggregates_are_thread_safe_and_bounded():
    profile = ProfileMetrics()
    def worker(_):
        for _ in range(1000):
            profile.record('stage', 'image_prepare', .001, .001, model='wd', units=1, unit='frames')
    with ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(worker, range(4)))
    rows = profile.export()
    assert len(rows) == 1
    assert rows[0]['calls'] == rows[0]['units'] == sum(rows[0]['histogram']) == 4000
    assert len(rows[0]['histogram']) == 64


def test_parallel_merge_profiles_real_workers_without_state_writes(tmp_path):
    from stash_ai.storage.store import StateStore
    from stash_ai.publication.parallel import iter_parallel_evidence
    from tests.helpers.evidence import empty_policies
    from tests.test_merge_parallel import _outputs
    path, scope = tmp_path / 'state.sqlite3', scope_for(app.STASH_URL)
    policies = empty_policies()
    with StateStore(path, scope) as store:
        store.put('1', 'wd', _outputs('wd', 'one'))
        store.put('2', 'wd', _outputs('wd', 'two'))
    before = path.read_bytes()
    results = list(iter_parallel_evidence(app, path, scope, ['1', '2'], policies, policies,
        workers=2, cache_bytes=0, memory_budget_bytes=128 * 2**20, profile=True))
    profile = ProfileMetrics()
    for result in results:
        assert result.error is None
        profile.ingest(result.metrics)
    metrics = profile.export()
    assert any(row['name'] == 'merge_worker_compute' and row['calls'] == 2 for row in metrics)
    assert any(row['kind'] == 'worker_cpu' for row in metrics)
    assert path.read_bytes() == before
