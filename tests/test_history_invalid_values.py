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


@pytest.mark.parametrize('field,value',[
    ('counts',None),('counts',[]),('counts',1),('counts',{'x':'bad'}),('counts',{'x':True}),
    ('counts',{'x':-1}),('counts',{'x':1.5}),('counts',{'x':10**1000}),
    ('outcomes',None),('outcomes',[]),('outcomes',{'wd':[]}),('outcomes',{'wd':{'fresh':False}}),
    ('outcomes',{'wd':{'fresh':'one'}}),('wall_seconds','bad'),('wall_seconds',True),
    ('wall_seconds',-1),('wall_seconds',float('nan')),('cpu_seconds',float('inf')),
    ('models',[]),('workload',{'verified_cues':float('nan')}),('partial',1),
])
def test_invalid_nonmetric_fields_are_isolated_and_valid_aggregates_survive(tmp_path,field,value):
    bad=row({'fingerprint':'known'});bad[field]=value
    original=deepcopy(bad)
    good=row({'fingerprint':'known'})
    good['counts']={'x':2};good['outcomes']={'wd':{'fresh':1}}
    report=build_report([good,deepcopy(good),bad,deepcopy(bad)])
    assert len(report['runs'])==4
    comparable=[group for group in report['groups'] if group['configuration_comparable']]
    assert len(comparable)==1 and comparable[0]['runs']==2
    assert comparable[0]['counts']['x']==4 and comparable[0]['outcomes']['wd']['fresh']==2
    assert comparable[0]['mean_wall_seconds']==3 and comparable[0]['metrics'][0]['mean_seconds']==2
    assert all(group['runs']==1 for group in report['groups'] if not group['configuration_comparable'])
    assert report['warning_counts'] and len(report['warnings'])<=20
    assert bad==original  # Reporting never edits caller data.
    destination=write_json_report(report,tmp_path/'mixed.json')
    assert len(json.loads(destination.read_text())['runs'])==4


def test_invalid_elapsed_values_are_unknown_and_large_valid_sums_do_not_overflow(tmp_path):
    bad=row({'fingerprint':'known'});bad['wall_seconds']='bad';bad['cpu_seconds']=False
    report=build_report([bad])
    assert report['groups'][0]['mean_wall_seconds'] is None and report['groups'][0]['mean_cpu_seconds'] is None
    huge=row({'fingerprint':'known'});huge['wall_seconds']=1e308
    report=build_report([huge,deepcopy(huge)])
    assert len(report['groups'])==2 and report['warning_counts']=={'elapsed_aggregate_overflow':1}
    write_json_report(report,tmp_path/'large.json')


def test_invalid_stored_nonmetric_json_does_not_hide_other_runs_or_rewrite_db(tmp_path):
    path=tmp_path/'history.sqlite'
    with HistoryStore(path) as store:
        for identifier in ('valid','bad'):
            store.start(identifier,'scope','inference','now',{'fingerprint':'same'})
        store.connection.execute("UPDATE runs SET counts='broken-json',outcomes='[]',wall_seconds='bad' WHERE id='bad'")
        store.connection.commit()
    before=path.read_bytes()
    with HistoryStore(path,readonly=True) as store:
        report=build_report(store.rows('scope'))
    assert path.read_bytes()==before and len(report['runs'])==2
    assert report['warning_counts']['invalid_counts']==1
    assert any(run.get('invalid_json_fields')==['counts'] for run in report['runs'])
    write_json_report(report,tmp_path/'report.json')


def test_warning_examples_for_invalid_counters_are_bounded():
    bad=row({'fingerprint':'known'});bad['counts']={str(index):'bad' for index in range(100)}
    report=build_report([bad])
    assert report['warning_counts']=={'invalid_counts_counter':100} and len(report['warnings'])==20
    assert report['groups'][0]['counts']=={}


def test_human_history_output_and_export_survive_invalid_elapsed_and_counters(tmp_path,monkeypatch,capsys):
    from types import SimpleNamespace
    from stash_ai.reports import runs
    from tests.helpers.runtime import runtime as app
    from stash_ai.storage.store import scope_for
    path=tmp_path/'history.sqlite'
    with HistoryStore(path) as store:
        store.start('bad',scope_for(app.STASH_URL),'inference','now',{'fingerprint':'same'})
        store.connection.execute("UPDATE runs SET counts='[]',wall_seconds='bad',status='finished',finished_at='later'")
        store.connection.commit()
    before=path.read_bytes()
    args=SimpleNamespace(run_history_db=path,state_db=tmp_path/'state.sqlite',run_limit=50,
                         run_command=None,run_status=None,run_model=None,run_report=tmp_path/'report.json')
    assert runs.run(app,args)==0
    output=capsys.readouterr().out
    assert 'elapsed unknown' in output and 'mean wall=unknown' in output
    assert path.read_bytes()==before and json.loads(args.run_report.read_text())['warnings']
