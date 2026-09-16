"""Diagnostic accuracy, payload privacy, bounded memory and unchanged merging."""
from copy import deepcopy
import json
from types import SimpleNamespace

import pytest

import stash_ai as app
import stash_merge as merge
from stash_diagnostics import MergeDiagnostics
from stash_runtime import RunReport
from test_progress import pipeline, reset


def records(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def test_nested_timings_are_exclusive_and_failures_are_recorded(tmp_path, monkeypatch):
    clock = SimpleNamespace(value=0.0)
    monkeypatch.setattr('stash_runtime.time.perf_counter', lambda: clock.value)
    monkeypatch.setattr('stash_runtime.time.process_time', lambda: clock.value / 2)
    report = RunReport(write=lambda _: None)
    path = tmp_path/'profile.jsonl'
    with MergeDiagnostics(report, path) as diagnostics:
        with diagnostics.scene('12'):
            with report.measure('merge_plan'):
                clock.value = 1
                with pytest.raises(ValueError):
                    with report.measure('nested'):
                        clock.value = 4
                        raise ValueError('not part of the report')
                clock.value = 10
    rows = records(path)
    scene = rows[1]
    assert scene['wall_seconds'] == 10 and scene['cpu_seconds'] == 5
    assert scene['stages']['merge_plan']['seconds'] == 7
    assert scene['stages']['merge_plan']['inclusive_seconds'] == 10
    assert scene['stages']['nested']['seconds'] == 3 and scene['stages']['nested']['errors'] == 1
    assert scene['unmeasured_seconds'] == 0
    assert rows[-1]['scene_wall_seconds'] == 10


@pytest.mark.parametrize('failure', [False, True])
def test_api_records_named_operation_bytes_and_errors_without_payloads(tmp_path, failure):
    path = tmp_path/'api.jsonl'
    report = RunReport(write=lambda _: None)
    client = app.StashClient(False); client.report = report
    secret = 'PAYLOAD_MUST_NEVER_ENTER_PROFILE'
    payload = {'errors': [{'message': secret}]} if failure else {'data': {'sceneMarkerCreate': {'id': '7'}}}
    body = json.dumps(payload).encode()
    client.api.close()
    client.api = SimpleNamespace(close=lambda: None, post=lambda *a, **kw: SimpleNamespace(
        content=body, status_code=200, raise_for_status=lambda: None, json=lambda: payload))
    try:
        with MergeDiagnostics(report, path) as diagnostics:
            with diagnostics.scene('1'):
                def call():
                    return client.gql('mutation StashAICreateMarker($input: SceneMarkerCreateInput!) { sceneMarkerCreate(input:$input) { id } }',
                                      {'input': {'title': secret, 'tag_ids': ['9'], 'seconds': 5}})
                if failure:
                    with pytest.raises(RuntimeError): call()
                else:
                    call()
    finally:
        client.close()
    text = path.read_text()
    assert secret not in text and 'tag_ids' not in text and 'graphql' not in text
    api = records(path)[1]['api']['StashAICreateMarker']
    assert api['calls'] == 1 and api['errors'] == int(failure) and api['response_bytes'] == len(body)
    assert api['max_seconds'] == api['seconds']
    stages = records(path)[1]['stages']
    assert stages['stash_http']['calls'] == stages['stash_json_decode']['calls'] == 1


def test_codec_measures_bytes_without_changing_persisted_format(tmp_path):
    report = RunReport(write=lambda _: None)
    payload = {'contributions': [{'tag': 'tag', 'extra_tags': ['unicode-ą']} for _ in range(10)]}
    with MergeDiagnostics(report, tmp_path/'codec.jsonl') as diagnostics:
        with diagnostics.scene('1'):
            baseline = merge.encode(payload)
            encoded = merge.encode(payload, report)
            assert baseline == encoded and merge.decode(encoded, report) == payload
    scene = records(tmp_path/'codec.jsonl')[1]
    assert scene['counts']['state_blob_bytes_read'] == scene['counts']['state_blob_bytes_written'] == len(encoded)
    assert scene['counts']['state_json_bytes_read'] == scene['counts']['state_json_bytes_written'] > len(encoded)
    assert set(scene['stages']) == {'state_json_encode', 'state_compress', 'state_decompress', 'state_json_decode'}


def test_interrupted_profile_keeps_completed_rows_and_closes_file(tmp_path):
    report = RunReport(write=lambda _: None); path = tmp_path/'interrupted.jsonl'
    with pytest.raises(KeyboardInterrupt):
        with MergeDiagnostics(report, path) as diagnostics:
            with diagnostics.scene('1'):
                pass
            with diagnostics.scene('2'):
                raise KeyboardInterrupt()
    rows = records(path)
    assert [row['status'] for row in rows[1:]] == ['ok', 'interrupted', 'interrupted']
    assert report.diagnostics is None and diagnostics.stream is None


def test_memory_is_bounded_and_existing_files_are_not_overwritten(tmp_path):
    report = RunReport(write=lambda _: None); path = tmp_path/'bounded.jsonl'
    with MergeDiagnostics(report, path) as diagnostics:
        for i in range(1500):
            with diagnostics.scene(i):
                report.record_counts(markers=100)
        assert len(diagnostics.slowest) == 3
        assert diagnostics.counts['markers'] == 150000
    before = path.read_bytes()
    with pytest.raises(FileExistsError):
        with MergeDiagnostics(report, path): pass
    assert path.read_bytes() == before


def test_profile_io_failure_does_not_interrupt_merge(tmp_path):
    messages = []; report = RunReport(write=messages.append)
    class FailedFile:
        def write(self, _): raise OSError('full')
        def close(self): raise OSError('full')
    with MergeDiagnostics(report) as diagnostics:
        diagnostics.stream = FailedFile()
        with diagnostics.scene('1'):
            pass
        assert diagnostics.stream is None
    assert any('write failed' in line for line in messages)


def test_profile_and_force_merge_preserve_results_and_do_not_run_models(pipeline, monkeypatch, tmp_path):
    p = pipeline
    assert app.main(p.args) == 0
    expected = deepcopy(p.stash.scene)
    p.stash.writes.clear()
    monkeypatch.setattr(app, 'load_tagger', lambda *a: pytest.fail('profile loaded a model'))
    monkeypatch.setattr(app, 'fetch_cues', lambda *a: pytest.fail('profile fetched media'))
    for force in (False, True):
        path = tmp_path/f'profile-{force}.jsonl'
        args = [*p.args, '--merge-only', '--profile-merge', str(path)] + (['--force-merge'] if force else [])
        assert app.main(args) == 0
        assert p.stash.scene == expected and not p.stash.writes
        scene = next(row for row in records(path) if row['type'] == 'scene')
        assert scene['reason'] == ('forced' if force else 'reused')
        assert scene['status'] == ('ok' if force else 'reused')
        if force:
            assert scene['counts']['retained_detections'] > 0
            assert scene['counts']['state_json_bytes_written'] > 0
            assert scene['stages']['marker_matching']['calls'] == 1
            assert scene['stages']['state_json_encode']['calls'] == 2
        assert sum(v['seconds'] for v in scene['stages'].values()) <= scene['wall_seconds'] * 1.01


def test_forced_dry_profile_preserves_stash_and_source_database(pipeline, tmp_path):
    p = pipeline
    assert app.main(p.args) == 0
    before = p.path.read_bytes(), deepcopy(p.stash.scene)
    p.stash.dry_run = True
    path = tmp_path/'dry.jsonl'
    assert app.main([*p.args, '--merge-only', '--force-merge', '--dry-run', '--profile-merge', str(path)]) == 0
    assert before == (p.path.read_bytes(), p.stash.scene)
    assert records(path)[-1]['dry_run']


@pytest.mark.parametrize('arguments', [['--force-merge'], ['--output-mode', 'separate', '--profile-merge', 'x.jsonl'],
                                    ['--merge-only', '--profile-merge', 'state.sqlite3']])
def test_invalid_profiling_flags_fail_before_stash_access(pipeline, arguments):
    with pytest.raises(SystemExit) as exc:
        app.main([*pipeline.args, *arguments])
    assert exc.value.code == 2 and not pipeline.stash.writes and not pipeline.loaded
