"""Point migration preserves ownership, manual metadata and durable retry proofs."""
from copy import deepcopy

import pytest

from stash_ai.tools.point_migration import plan
from stash_ai.publication.preconditions import require_native_ranges
from stash_ai.evidence.labels import AI
from stash_ai.evidence.marker_identity import OCCURRENCE_TAG
from stash_ai.storage.store import StateStore
from tests.helpers.evidence import span
from tests.helpers.stash import Stash

pytestmark = pytest.mark.usefixtures('reset_merge')


def scene(stash):
    marker = stash.marker(AI, 'common', 'old')
    marker.update(seconds=10, end_seconds=None)
    marker['tags'].append(stash.tag('human'))
    return stash.find_scene('1'), marker


@pytest.mark.parametrize('location', ['published', 'pending', 'recovery'])
def test_saved_extent_updates_native_end_without_changing_other_fields(tmp_path, location):
    stash = Stash()
    observed, marker = scene(stash)
    original = deepcopy(marker)
    with StateStore(tmp_path/'state.sqlite3', 'scope') as store:
        saved = {'contributions': [span('wd', 10, 20), span('dtq', 10.0004, 30)]}
        if location == 'published':
            store.publish('1', {'markers': saved})
        else:
            store.prepare('1', {'markers': saved})
            if location == 'recovery':
                from stash_ai.storage.recovery import abandon
                abandon(store, lambda _: None)
        before = store.connection.execute('SELECT * FROM snapshots').fetchall()
        updates = plan(store, observed)
        assert len(updates) == 1 and updates[0]['end_seconds'] == 30
        assert updates[0]['title'] == original['title']
        assert updates[0]['tag_ids'] == [str(tag['id']) for tag in original['tags']]
        with pytest.raises(RuntimeError, match='stash_ai.tools.point_migration'):
            require_native_ranges(observed)
        stash.update_marker(updates[0])
        assert plan(store, stash.find_scene('1')) == []
        assert store.connection.execute('SELECT * FROM snapshots').fetchall() == before
        assert not store.has_pending_publications() or location == 'pending'
    assert marker['id'] == original['id'] and marker['seconds'] == 10


def test_occurrence_and_manual_points_are_untouched(tmp_path):
    stash = Stash()
    observed, marker = scene(stash)
    marker['tags'].append(stash.tag(OCCURRENCE_TAG))
    manual = stash.marker(AI, 'other', 'manual', manual=True)
    manual.update(end_seconds=None)
    observed = stash.find_scene('1')
    with StateStore(tmp_path/'state.sqlite3', 'scope') as store:
        assert plan(store, observed) == []
        require_native_ranges(observed)


def test_missing_extent_fails_before_any_marker_can_be_updated(tmp_path):
    stash = Stash()
    observed, _ = scene(stash)
    unknown = stash.marker(AI, 'other', 'unknown')
    unknown.update(seconds=12, end_seconds=None)
    with StateStore(tmp_path/'state.sqlite3', 'scope') as store:
        store.publish('1', {'markers': {'contributions': [span('wd', 10, 20)]}})
        with pytest.raises(RuntimeError, match='No saved extent'):
            plan(store, stash.find_scene('1'))
        assert not stash.writes



def test_ordinary_unknown_duration_requires_a_verified_extent(tmp_path):
    stash = Stash()
    observed, _ = scene(stash)
    imported = span('wd', 10, None)
    imported['meta']['origin'] = 'legacy'
    with StateStore(tmp_path/'state.sqlite3', 'scope') as store:
        store.publish('1', {'markers': {'contributions': [imported]}})
        with pytest.raises(RuntimeError, match='No saved extent'):
            plan(store, observed)
        with pytest.raises(RuntimeError, match='stash_ai.tools.point_migration'):
            require_native_ranges(observed)


@pytest.mark.parametrize('dry_run', [False, True])
def test_point_tool_cli_preserves_state_and_manual_metadata(pipeline, monkeypatch, dry_run):
    from stash_ai.tools.point_migration import main
    from stash_ai.storage.store import scope_for
    from tests.helpers.runtime import runtime as app
    p = pipeline
    _, marker = scene(p.stash)
    with StateStore(p.path, scope_for(app.STASH_URL)) as store:
        store.publish('1', {'markers': {'contributions': [span('wd', 10, 20)]}})
    before_db, before = p.path.read_bytes(), deepcopy(p.stash.scene)
    p.stash.dry_run = dry_run
    monkeypatch.setattr(app.services, 'load_tagger', lambda *args: pytest.fail('point tool loaded model'))
    args = ['--state-db', str(p.path), '--scene-id', '1'] + (['--dry-run'] if dry_run else [])
    assert main(args, runtime=app) == 0
    assert p.path.read_bytes() == before_db
    if dry_run:
        assert p.stash.scene == before and not p.stash.writes
    else:
        marker = p.stash.scene['scene_markers'][0]
        assert marker['end_seconds'] == 20
        assert marker['title'] == before['scene_markers'][0]['title']
        assert marker['tags'] == before['scene_markers'][0]['tags']


def test_ordinary_run_rejects_ai_points_before_inference_and_stash_writes(pipeline, monkeypatch, capsys):
    from tests.helpers.runtime import runtime as app
    p = pipeline
    assert app.main(p.args) == 0
    for marker in p.stash.scene['scene_markers']:
        marker['end_seconds'] = None
    before = deepcopy(p.stash.scene)
    p.stash.writes.clear()
    monkeypatch.setattr(app.services, 'load_tagger', lambda *args: pytest.fail('model loaded before range check'))
    assert app.main(p.args) == 1
    assert 'stash_ai.tools.point_migration' in capsys.readouterr().out
    assert p.stash.scene == before and not p.stash.writes
