"""Deferred verification preserves valid predictions and isolates unavailable scenes."""
from copy import deepcopy
from types import SimpleNamespace
import sqlite3

import pytest

from tests.helpers.runtime import runtime as app
from tests.helpers.evidence import empty_policies
from stash_ai.storage.store import StateStore, scope_for
from stash_ai.storage import input_checks, progress

pytestmark = pytest.mark.usefixtures('reset_merge')


def blobs(path):
    with sqlite3.connect(path) as db:
        return db.execute('SELECT * FROM results ORDER BY 1,2,3,4').fetchall()


def test_missing_media_defers_publication_and_restoration_reuses_predictions(pipeline, monkeypatch):
    p = pipeline
    assert app.main(p.args) == 0
    before = blobs(p.path)
    published = deepcopy(p.stash.scene['scene_markers'])
    loaded = len(p.loaded)
    vtt = p.stash.scene['paths'].pop('vtt')
    p.stash.writes.clear()
    monkeypatch.setattr(app, 'MARKER_MIN_HITS', 99)
    assert app.main([*p.args, '--resume']) == 1
    assert p.stash.scene['scene_markers'] == published
    assert not p.stash.writes and len(p.loaded) == loaded
    with StateStore(p.path, scope_for(app.STASH_URL)) as store:
        assert input_checks.pending(store, '1') == {'wd': 'missing_media'}
        assert not progress.pending_refreshes(store, ['scene', 'markers'])
    p.stash.scene['paths']['vtt'] = vtt
    assert app.main([*p.args, '--resume']) == 0
    assert len(p.loaded) == loaded and blobs(p.path) == before
    with StateStore(p.path, scope_for(app.STASH_URL)) as store:
        assert not input_checks.pending(store, '1')


@pytest.mark.parametrize('reason', ['missing_media', 'refresh'])
def test_report_excludes_unverified_or_refresh_blocked_results(pipeline, reason):
    from stash_ai.reports.model.report import build_report
    p = pipeline
    assert app.main(p.args) == 0
    with StateStore(p.path, scope_for(app.STASH_URL)) as store:
        if reason == 'refresh':
            progress.block_publication(store, ['1'], 'wd', ['scene', 'markers'])
        else:
            input_checks.defer(store, '1', 'wd')
        report = build_report(app, store, ['1'], empty_policies(), empty_policies())
        assert report['totals'].get('scenes', 0) == 0
        assert report['totals'].get('markers', 0) == 0
        assert report['warnings'][0]['scenes'] == 1


def test_media_read_error_does_not_request_fresh_predictions(pipeline, monkeypatch):
    from stash_ai.inference.inputs import RunInputs
    p = pipeline
    assert app.main(p.args) == 0
    loaded = len(p.loaded)
    original = RunInputs.inspect
    def fail(*args, **kwargs):
        raise TimeoutError('temporary media read failure')
    monkeypatch.setattr(RunInputs, 'inspect', fail)
    assert app.main([*p.args, '--resume']) == 1
    with StateStore(p.path, scope_for(app.STASH_URL)) as store:
        assert input_checks.pending(store, '1') == {'wd': 'read_error'}
        assert not progress.pending_refreshes(store, ['scene', 'markers'])
    monkeypatch.setattr(RunInputs, 'inspect', original)
    assert app.main([*p.args, '--resume']) == 0
    assert len(p.loaded) == loaded


@pytest.mark.parametrize('sql', [False, True])
def test_explicit_missing_scene_does_not_stop_following_ids(sql):
    from stash_ai.stash_api.scene_reads import SceneReads, MissingSceneError
    from stash_ai.runtime.report import RunReport
    visited = []
    def gql(query, variables):
        visited.append(variables['id'])
        return {'findScene': None if variables['id'] == '1' else {'id': '2'}}
    reads = SceneReads(SimpleNamespace(scene_page_size=100, gql=gql, scene_fields=lambda: 'id'))
    reads.report = RunReport()
    if sql:
        reads.merge_only = True
        reads.snapshots = lambda ids: {'2': {'id': '2'}}
    assert list(reads.iter_scenes(['1', '2'])) == [{'id': '2'}]
    assert isinstance(reads.report.unread_scenes['1'], MissingSceneError)
    if not sql:
        assert visited == ['1', '2']


def test_deleted_pending_init_scene_is_retired_and_healthy_scene_finishes(pipeline, monkeypatch):
    from tests.test_db_init import collection
    p = pipeline
    args = [*collection(p, monkeypatch), '--db-init']
    p.fail['wd'] = 4
    with pytest.raises(KeyboardInterrupt):
        app.main(args)
    p.fail.clear()
    del p.stash.scenes['2']
    assert app.main([x for x in args if x != '--db-init']) == 0
    with StateStore(p.path, scope_for(app.STASH_URL)) as store:
        assert not store.initialization.active
        assert store.initialization.removed == {'2'}
        assert store.records('1', 'scene')


def test_v25_upgrade_preserves_raw_results_and_publications(pipeline):
    from stash_ai.db.update import update_database
    from stash_ai.storage.schema import LATEST_SCHEMA_VERSION
    p = pipeline
    assert app.main(p.args) == 0
    before = blobs(p.path)
    with sqlite3.connect(p.path) as db:
        snapshots = db.execute('SELECT * FROM snapshots').fetchall()
        db.execute('DROP TABLE input_checks')
        db.execute('PRAGMA user_version=25')
    assert update_database(p.path) == (25, LATEST_SCHEMA_VERSION)
    assert blobs(p.path) == before
    with sqlite3.connect(p.path) as db:
        assert db.execute('SELECT * FROM snapshots').fetchall() == snapshots
