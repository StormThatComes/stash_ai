"""Reports reject mixed source generations and preview exposes only read methods."""
import sqlite3
from types import SimpleNamespace

import pytest

from stash_ai.db.inspection import inspect_database
from stash_ai.db.read_source import consistent_source
from stash_ai.storage.store import StateStore, scope_for
from stash_ai.cleanup.sql_preview import SQLPreview
from stash_ai.cleanup.groups import PreviewUsage
from tests.helpers.runtime import runtime as app


def test_db_report_remains_one_snapshot_across_writer_commit(tmp_path, monkeypatch):
    from stash_ai.db import inspection
    path = tmp_path/'state.sqlite3'
    with StateStore(path, 's') as store:
        store.connection.execute('PRAGMA journal_mode=WAL')
    original = inspection._payload_bytes
    called = []
    def read(*args):
        if not called:
            called.append(True)
            with sqlite3.connect(path) as writer:
                writer.execute("INSERT INTO refresh_requests VALUES ('s','1','wd','scene')")
        return original(*args)
    monkeypatch.setattr(inspection, '_payload_bytes', read)
    assert inspect_database(path, 's')['refresh_requests'] == 0
    assert inspect_database(path, 's')['refresh_requests'] == 1


def test_shared_source_guard_detects_commit_between_readers(tmp_path):
    path = tmp_path/'state.sqlite3'
    with StateStore(path, 's'):
        pass
    with pytest.raises(RuntimeError, match='changed during offline report'):
        with consistent_source(path):
            with sqlite3.connect(path) as writer:
                writer.execute("INSERT INTO refresh_requests VALUES ('s','1','wd','scene')")


def test_report_rejects_change_after_parent_scene_selection_before_worker_start(tmp_path, monkeypatch):
    from stash_ai.reports.model import report, workers
    from tests.test_model_report import inference_pair
    path, destination = tmp_path/'state.sqlite3', tmp_path/'report.json'
    with StateStore(path, scope_for(app.STASH_URL)) as store:
        for scene in ('1', '2'):
            store.put(scene, 'wd', inference_pair(scene))
    def build(*args, **kwargs):
        with sqlite3.connect(path) as writer:
            writer.execute("INSERT INTO refresh_requests VALUES (?,'1','wd','scene')", (scope_for(app.STASH_URL),))
        return {}
    monkeypatch.setattr(workers, 'build_report_parallel', build)
    assert app.main(['--state-db', str(path), '--model-report', str(destination), '--model-report-workers', '2']) == 1
    assert not destination.exists()


@pytest.mark.parametrize('wrapper', [lambda s: PreviewUsage(s, {}),
                                    lambda s: SQLPreview(s, app, set(), {})])
def test_preview_has_explicit_read_surface_and_blocks_mutation_query(wrapper):
    calls = []
    stash = SimpleNamespace(dry_run=False, secret_transport='secret', create_marker=lambda *_: None,
                            gql=lambda *a: calls.append(a) or {})
    view = wrapper(stash)
    assert view.dry_run
    assert not hasattr(view, 'create_marker') and not hasattr(view, 'secret_transport')
    with pytest.raises(ValueError, match='only named GraphQL read'):
        view.gql('mutation StashAIChange { change }')
    assert not calls
    assert view.gql('query StashAIRead { version }') == {}
