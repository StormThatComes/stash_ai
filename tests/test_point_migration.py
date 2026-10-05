"""Point migration preserves ownership, manual metadata and durable retry proofs."""
from copy import deepcopy

import pytest

from stash_ai.commands.point_migration import plan, require_native_ranges
from stash_ai.evidence.labels import AI
from stash_ai.evidence.marker_identity import OCCURRENCE_TAG
from stash_ai.storage.store import StateStore
from tests.helpers.evidence import span
from tests.helpers.stash import Stash


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
        with pytest.raises(RuntimeError, match='migrate-point-markers-only'):
            require_native_ranges(observed, [])
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
        require_native_ranges(observed, [])


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
        with pytest.raises(RuntimeError, match='migrate-point-markers-only'):
            require_native_ranges(observed, [imported])
