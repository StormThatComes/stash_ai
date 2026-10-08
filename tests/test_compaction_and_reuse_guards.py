"""Maintenance checks keep their lock; merge reuse observes current Stash state."""
from contextlib import contextmanager
from copy import deepcopy

import pytest

from tests.helpers.runtime import runtime as app
from tests.helpers.evidence import segment
from stash_ai.storage.store import StateStore
from stash_ai.db.checks import check_database

pytestmark = pytest.mark.usefixtures('reset_progress')


def test_compaction_verifies_before_unlocking_and_accepts_a_later_writer(tmp_path, monkeypatch):
    from stash_ai.db import compaction
    path = tmp_path/'state.db'
    with StateStore(path, 's'):
        pass
    original = compaction._writer_lock
    @contextmanager
    def lock(path):
        with original(path):
            yield
        # A valid write immediately after release is not compaction corruption.
        with StateStore(path, 's') as state:
            state.put('1', 'wd', {'markers': segment('wd', [[('a', .9)]])})
    monkeypatch.setattr(compaction, '_writer_lock', lock)
    before, after = compaction.compact_database(path, 's')
    assert compaction._same_logical_counts(before, after)
    assert check_database(path, 's')['ok']
    with StateStore(path, 's', True) as state:
        assert state.records('1', 'markers')


def test_compaction_final_inspection_still_excludes_other_writers(tmp_path, monkeypatch):
    from stash_ai.db import compaction
    path = tmp_path/'state.db'
    with StateStore(path, 's'):
        pass
    original = compaction.inspect_database
    calls = []
    def inspect(path, scope):
        calls.append(True)
        if len(calls) == 3:
            with pytest.raises(RuntimeError, match='Another merged-output process'):
                StateStore(path, scope)
        return original(path, scope)
    monkeypatch.setattr(compaction, 'inspect_database', inspect)
    compaction.compact_database(path, 's')
    assert len(calls) == 3


def test_merge_rechecks_automation_changes_after_initial_scan(pipeline, monkeypatch):
    from stash_ai.inference.inputs import RunInputs
    p = pipeline
    assert app.main(p.args) == 0
    expected = deepcopy(p.stash.scene['scene_markers'][0])
    original = RunInputs.inspect
    changed = []
    def inspect(self, *args):
        result = original(self, *args)
        if not changed:
            marker = p.stash.scene['scene_markers'][0]
            marker['title'] = '[AI] changed by another automation'
            marker['end_seconds'] += 5
            changed.append(marker['id'])
        return result
    monkeypatch.setattr(RunInputs, 'inspect', inspect)
    p.stash.writes.clear()
    assert app.main([*p.args, '--force-fetch-vtt']) == 0
    repaired = next(m for m in p.stash.scene['scene_markers'] if m['id'] == changed[0])
    assert repaired['title'] == expected['title']
    assert repaired['end_seconds'] == expected['end_seconds']
    assert any(operation == 'update' for operation, _ in p.stash.writes)
