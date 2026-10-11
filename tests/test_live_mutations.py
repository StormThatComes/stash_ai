"""Confirmed writes survive later failures without claiming local publication."""
from collections import Counter
from types import SimpleNamespace

import pytest

from stash_ai.publication.merged import publish_sync_plan
from stash_ai.runtime.report import RunReport


@pytest.mark.parametrize('failure,outcome', [('dispatch', 'failed'), ('ack', 'unknown'), ('commit', None)])
def test_partial_mutations_and_local_commit_are_separate(failure, outcome):
    report, events = RunReport(), []
    plan = SimpleNamespace(creates=[{'id': 1}, {'id': 2}], updates=[], deletes=[],
                           scene_changed=False, snapshots={})
    def create(payload):
        events.append('create')
        if len(events) == 3 and failure != 'commit':
            error = RuntimeError('no acknowledgement')
            if failure == 'dispatch': error.write_outcome = 'failed'
            raise error
        return str(payload['id'])
    def commit(*args, **kwargs):
        events.append('commit')
        raise OSError('commit failed')
    store = SimpleNamespace(prepare=lambda *args: events.append('intent'), commit_prepared=commit)
    stash = SimpleNamespace(dry_run=False, create_marker=create)
    with pytest.raises((RuntimeError, OSError)):
        publish_sync_plan(SimpleNamespace(APPLY_SCENE_TAGS=False), stash,
            SimpleNamespace(missing=True), store, '1', {}, [], [], plan, Counter(),
            report=report, publication_key=None)
    assert report.counts['mutation_attempted'] == 2
    assert report.counts['mutation_confirmed'] == (2 if failure == 'commit' else 1)
    assert report.counts['mutation_failed'] == int(outcome == 'failed')
    assert report.counts['mutation_unknown'] == int(outcome == 'unknown')
    assert report.counts['publication_pending'] == 1
    assert report.counts['publication_committed'] == 0
    assert ('commit' in events) == (failure == 'commit')
