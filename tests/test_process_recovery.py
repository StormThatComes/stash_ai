"""Abrupt death recovers durable prefixes and idempotent cleanup ownership."""
from copy import deepcopy
import json
from pathlib import Path
import sqlite3
import subprocess
import sys

import pytest

from stash_ai.storage.store import StateStore, scope_for
from tests.helpers.cleanup import FastStash, options, install
from tests.helpers.cleanup_sql import SQLStash
from tests.helpers.runtime import runtime as app
from tests.test_run_history import history_rows

pytestmark = pytest.mark.usefixtures('reset_progress')


def crash(module, tmp_path, *arguments):
    result = subprocess.run([sys.executable, '-m', module, str(tmp_path), *arguments],
        cwd=Path(__file__).resolve().parents[1], capture_output=True, text=True, timeout=30)
    assert result.returncode == 73, result.stdout + result.stderr


@pytest.mark.parametrize('cadence', [1, 3])
@pytest.mark.parametrize('stage', ['flush', 'before_finish', 'finish', 'before_history_finish', 'history_finish'])
def test_cli_recovers_only_committed_current_and_staged_work(pipeline, monkeypatch, tmp_path, stage, cadence):
    crash('tests.helpers.crash_inference', tmp_path, stage, str(cadence))
    with sqlite3.connect(pipeline.path) as db:
        prefixes = dict(db.execute("SELECT scene,processed FROM checkpoints"))
        completed = {row[0] for row in db.execute('SELECT DISTINCT scene FROM results')}
    if stage == 'flush':
        assert prefixes['1'] == cadence * 2 and not completed
    elif stage == 'before_finish':
        assert prefixes['1'] == 10 and not completed
    elif stage == 'finish':
        assert completed == {'1'} and '1' not in prefixes
    else:
        assert completed == {'1', '2'} and not prefixes
    rows = history_rows(pipeline.path)
    assert len(rows) == 1
    assert rows[0]['status'] == ('finished' if stage == 'history_finish' else 'running')
    stash = FastStash()
    stash.scenes['2'] = dict(deepcopy(stash.scene), id='2')
    monkeypatch.setattr(app.services, 'create_stash_client', lambda _: stash)
    args = ['--models', 'wd', '--no-hierarchy', '--state-db', str(pipeline.path), '--merge-workers', '1']
    assert app.main(args) == 0
    seen = [value for model in pipeline.loaded for value in model.seen]
    assert len(seen) == sum(10 - prefixes.get(scene, 0) for scene in ('1', '2') if scene not in completed)
    with StateStore(pipeline.path, scope_for(app.STASH_URL)) as store:
        assert not store.connection.execute('SELECT 1 FROM checkpoints').fetchone()
        assert not store.has_pending_publications()
        outputs = [store.results.model_record(scene, 'scene', 'wd') for scene in ('1', '2')]
        assert outputs[0]['items'] == outputs[1]['items']
    assert len(history_rows(pipeline.path)) == 2


@pytest.mark.parametrize('stage', ['mutation', 'finish'])
def test_cleanup_resumes_durable_intent_without_duplicate_mutations(tmp_path, monkeypatch, stage):
    crash('tests.helpers.crash_cleanup', tmp_path, stage)
    remote = json.loads((tmp_path/'remote.json').read_text())
    stash = SQLStash()
    stash.tags, stash.scene = remote['tags'], remote['scene']
    stash.scenes = {'1': stash.scene}
    install(monkeypatch, stash)
    with StateStore(tmp_path/'state.sqlite3', scope_for(app.STASH_URL), allow_cleanup=True) as store:
        assert bool(store.cleanup_pending) == (stage == 'mutation')
    assert app.main([*options(tmp_path, ['blocked']), '--cleanup-strategy', 'sql']) == 0
    with StateStore(tmp_path/'state.sqlite3', scope_for(app.STASH_URL), allow_cleanup=True) as store:
        assert not store.cleanup_pending and store.cleanup_generation == (1 if stage == 'mutation' else 2)
    assert not stash.scene['scene_markers']
    from stash_ai.evidence.labels import AI
    assert AI.prefix + 'blocked' not in stash.tags
    assert not any(op == 'delete' for op, *_ in stash.writes)
