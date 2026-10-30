from collections import OrderedDict
from types import SimpleNamespace
import json
import pytest

from benchmarks.benchmark_merge_solver import fixture
from stash_ai.recognition.merge_solver_checkpoint import Journal
from stash_ai.recognition.solver_journal_events import load_document, log_path


def test_event_appends_do_not_rewrite_manifest_and_recover_torn_tail(tmp_path):
    config, _ = fixture(tmp_path/'input', 'face')
    path = tmp_path/'journal.json'
    journal = Journal(path, {'test':1})
    initial = path.read_bytes()
    for index in range(20):
        journal.task('trial-'+str(index), config, 'queued')
    assert path.read_bytes() == initial
    with log_path(path).open('ab') as stream:
        stream.write(b'{"sequence":21')
    journal.close()
    resumed = Journal(path, {'test':1}, resume=True)
    assert len(resumed.document['tasks']) == 20
    resumed.task('trial-20', config, 'running')
    resumed.flush_snapshot()
    assert len(load_document(path)['tasks']) == 21
    assert json.loads(path.read_text())['journal_applied'] == 21


def test_event_corruption_is_rejected(tmp_path):
    config, _ = fixture(tmp_path/'input', 'face')
    path = tmp_path/'journal.json'
    journal = Journal(path, {'test':1})
    journal.task('trial-1', config, 'queued')
    event = log_path(path)
    event.write_bytes(event.read_bytes().replace(b'queued', b'failed'))
    journal.close()
    with pytest.raises(ValueError, match='Corrupt'):
        Journal(path, {'test':1}, resume=True)


def test_applied_record_retires_raw_result_only_after_durable_event(tmp_path):
    config, _ = fixture(tmp_path/'input', 'face')
    path = tmp_path/'journal.json'
    journal = Journal(path, {'test':1})
    journal.save_raw_result('trial-1', config, {'prediction_delta':{'a':None}})
    raw = journal.directory/journal.document['raw_results']['trial-1']['file']
    assert raw.exists()
    journal.record('trial-1', config, {'failed':False}, {}, None)
    assert not raw.exists()
    journal.close()
    resumed = Journal(path, {'test':1}, resume=True)
    assert resumed.get('trial-1')['item'] == {'failed':False}
    assert resumed.raw_result('trial-1') is None


def test_ordered_scheduler_refills_after_each_retirement():
    from stash_ai.recognition.solver_pool import Trials
    class Pool:
        execution = SimpleNamespace(scheduling='batch', trials=2)
        history = ()
        def __init__(self):self.pending = OrderedDict(); self.events = []
        def dispatch(self, name, settings):
            self.events.append(('dispatch',name)); self.pending[name] = settings, None
        def retire(self, name):
            self.events.append(('retire',name)); settings, _ = self.pending.pop(name); return name, settings
    pool = Pool()
    assert list(Trials.entries(pool, [('one',{}),('two',{}),('three',{})])) == [('one',{}),('two',{}),('three',{})]
    assert pool.events[:5] == [('dispatch','one'),('dispatch','two'),('retire','one'),('dispatch','three'),('retire','two')]
