"""Malformed history values stay unverified and do not hide other usable runs."""
from copy import deepcopy
import json

import pytest

from stash_ai.history.store import HistoryStore
from stash_ai.history.reconcile import reconcile
from stash_ai.reports.runs import build_report
from stash_ai.reports.io import write_json_report
from stash_ai.runtime.profile import ProfileMetrics


@pytest.mark.parametrize('configuration', ['null', '[]', '123', '"text"', 'broken-json'])
@pytest.mark.parametrize('dry_run', [True, False])
def test_unverified_configuration_does_not_block_other_selected_run(tmp_path, monkeypatch, configuration, dry_run):
    import stash_ai.history.reconcile as module
    monkeypatch.setattr(module, 'owner_is_dead', lambda owner: True if owner == {'proof': 'dead'} else None)
    with HistoryStore(tmp_path/'history.sqlite') as store:
        store.start('bad', 'scope', 'inference', '2026-10-13T03:00:00+00:00', {})
        store.start('good', 'scope', 'inference', '2026-10-13T03:00:00+00:00', {'process_owner': {'proof': 'dead'}})
        with store.connection:
            store.connection.execute("UPDATE runs SET configuration=? WHERE id='bad'", (configuration,))
        before = store.connection.execute("SELECT * FROM runs WHERE id='bad'").fetchone()
        result = reconcile(store, 'scope', ['bad', 'good'], dry_run=dry_run)
        assert result['rows'] == [{'id': 'bad', 'action': 'unverified'}, {'id': 'good', 'action': 'orphaned'}]
        assert store.connection.execute("SELECT * FROM runs WHERE id='bad'").fetchone() == before
        assert store.connection.execute("SELECT status FROM runs WHERE id='good'").fetchone()[0] == (
            'running' if dry_run else 'interrupted')


def metric():
    values = ProfileMetrics()
    values.record('stage', 'inference_batch', 2., 2., model='wd', work='fresh', units=4, unit='frames')
    return values.export()[0]


def row(configuration=None, metrics=None):
    return dict(command='inference', status='finished', configuration={} if configuration is None else configuration,
                models={'wd': {}}, wall_seconds=3., cpu_seconds=1., counts={}, outcomes={},
                metrics=[metric()] if metrics is None else metrics)


@pytest.mark.parametrize('configuration', [{}, None, [], {'fingerprint': ''}, {'fingerprint': 123}])
def test_unknown_fingerprints_are_isolated_and_never_mixed_with_verified_groups(configuration):
    unknown = row(); unknown['configuration'] = configuration
    report = build_report([unknown, deepcopy(unknown), row({'fingerprint': 'known'}), row({'fingerprint': 'known'})])
    assert len(report['runs']) == 4 and len(report['groups']) == 3
    legacy = [group for group in report['groups'] if not group['configuration_comparable']]
    assert len(legacy) == 2 and all(group['runs'] == 1 and group['configuration_fingerprint'] is None for group in legacy)
    known = next(group for group in report['groups'] if group['configuration_comparable'])
    assert known['runs'] == 2 and known['metrics'][0]['calls'] == 2
    assert known['metrics'][0]['mean_seconds'] == 2
    assert report['warning_counts'] == {'unknown_configuration_fingerprint': 2}


@pytest.mark.parametrize('bad', [{}, 1, None, {'work': []}])
def test_invalid_metric_entries_keep_run_and_valid_weighted_metrics(tmp_path, bad):
    good = metric()
    original = row({'fingerprint': 'known'}, [bad, good])
    report = build_report([original])
    group = report['groups'][0]
    assert group['metrics'][0]['calls'] == 1 and group['metrics'][0]['mean_seconds'] == 2
    assert group['mean_wall_seconds'] == 3 and group['invalid_metrics'] == 1
    assert report['runs'][0]['metrics'] == [good] and original['metrics'] == [bad, good]
    assert report['warning_counts'] == {'invalid_metric': 1}
    write_json_report(report, tmp_path/'report.json')


@pytest.mark.parametrize('field,value', [('calls', True), ('errors', 2), ('histogram', []),
    ('histogram', [0] * 64), ('max_seconds', float('inf')), ('seconds', -1),
    ('seconds', 10 ** 1000), ('min_seconds', None), ('model', []), ('units', 'four')])
def test_invalid_numeric_metric_fields_are_excluded_and_export_stays_available(tmp_path, field, value):
    bad = metric(); bad[field] = value
    report = build_report([row({'fingerprint': 'known'}, [bad, metric()])])
    assert report['warning_counts'] == {'invalid_metric': 1}
    assert report['groups'][0]['metrics'][0]['calls'] == 1
    destination = write_json_report(report, tmp_path/'report.json')
    assert json.loads(destination.read_text())['runs'][0]['invalid_metrics'] == 1


def test_invalid_metric_warning_examples_are_bounded_and_nullable_keys_sort():
    nullable = metric(); nullable['work'] = None
    empty = metric(); empty['work'] = ''
    report = build_report([row({'fingerprint': 'known'}, [None] * 100 + [nullable, empty])])
    assert report['warning_counts'] == {'invalid_metric': 100} and len(report['warnings']) == 20
    assert len(report['groups'][0]['metrics']) == 2
