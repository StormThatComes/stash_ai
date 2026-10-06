"""Retired proofs preserve retry geometry; cleanup and recovery cannot guess ranges."""
from stash_ai.storage.schema import LATEST_SCHEMA_VERSION
from copy import deepcopy
import sqlite3

import pytest

from stash_ai.db.checks import check_database
from stash_ai.db.update import update_database
from stash_ai.evidence.labels import AI
from stash_ai.evidence.temporal.policy import build_options
from stash_ai.models import SOURCES
from stash_ai.storage.store import StateStore
from stash_ai.storage.codec import encode
from stash_ai.storage.snapshots import compact, require_rebased
from stash_ai.publication.recovery import reconcile_marker_snapshot
from stash_ai.publication.preconditions import require_native_ranges
from tests.helpers.runtime import runtime as app
from tests.helpers.evidence import span, segment
from tests.helpers.stash import Stash

pytestmark = pytest.mark.usefixtures('reset_temporal')


def old_publication():
    value = span('wd', 10, 20, extras=['extra'])
    value['meta'] = {**segment('wd', [])['meta'], 'analysis_format': 3}
    value['evidence_signature'] = 'old-proof'
    value['build_options'] = {**build_options(app), 'algorithm': 'historical',
        'core_exclusions': {s: [] for s in SOURCES}, 'tag_exclusions': {s: [] for s in SOURCES}}
    return {'owned_ids': ['manual-owner'], 'contributions': [value]}


@pytest.mark.parametrize('location', ['published', 'pending', 'recovery'])
def test_offline_rebase_keeps_geometry_ownership_and_retry_proofs_across_scopes(tmp_path, location):
    path = tmp_path/'state.sqlite3'
    value = old_publication()
    # An older writer, outside the current frozen-engine validation boundary.
    with StateStore(path, 'scope-a'):
        pass
    with sqlite3.connect(path) as connection:
        packed = encode({'__stash_ai_snapshot__': 4, 'snapshot': value, 'proofs': []})
        for scope in ('scope-a', 'scope-b'):
            connection.execute('INSERT INTO snapshots VALUES (?,?,?,?,?)',
                (scope, '1', 'markers', packed if location == 'published' else None,
                 packed if location == 'pending' else None))
            if location == 'recovery':
                connection.execute('INSERT INTO recovery_publications VALUES (?,?,?,?,?)',
                                   (scope, '1', 'markers', 'lost-response', packed))
        connection.execute('PRAGMA user_version=18')
    before = path.read_bytes()
    assert update_database(path, dry_run=True) == (18, LATEST_SCHEMA_VERSION)
    assert path.read_bytes() == before
    assert update_database(path) == (18, LATEST_SCHEMA_VERSION)
    for scope in ('scope-a', 'scope-b'):
        with StateStore(path, scope) as state:
            from stash_ai.storage.recovery import publications
            saved = (publications(state, '1', 'markers')[0] if location == 'recovery'
                     else state.snapshot('1', 'markers')[location == 'pending'])
            item = saved['contributions'][0]
            assert saved['owned_ids'] == value['owned_ids']
            assert 'retired_build_options' not in item and 'build_options' not in item
            assert item['evidence_signature'] == 'old-proof' and item['requires_rebase']
            assert (item['start'], item['end'], item['extra_tags']) == (10, 20, ['extra'])
            with pytest.raises(RuntimeError, match='fresh inference'):
                require_rebased(saved['contributions'])
            stash = Stash()
            marker = stash.marker(AI, 'common', 'old')
            marker.update(seconds=10, end_seconds=None)
            marker['tags'].append(stash.tag('human'))
            before_marker = deepcopy(marker)
            with pytest.raises(RuntimeError, match='migration tools from commit'):
                require_native_ranges(stash.find_scene('1'))
            assert marker == before_marker and not stash.writes
    with sqlite3.connect(path) as connection:
        migrated = [connection.execute('SELECT * FROM ' + table).fetchall()
                    for table in ('snapshots', 'recovery_publications')]
    assert update_database(path) == (LATEST_SCHEMA_VERSION, LATEST_SCHEMA_VERSION)
    with sqlite3.connect(path) as connection:
        assert [connection.execute('SELECT * FROM ' + table).fetchall()
                for table in ('snapshots', 'recovery_publications')] == migrated
    assert check_database(path, 'scope-a')['ok']


@pytest.mark.parametrize('observed_end,saved_end,match', [
    (None, 20, False), (20, None, False), (None, None, True),
    (20, 20, True), (21, 20, False),
])
def test_recovery_requires_exact_point_or_range_geometry(observed_end, saved_end, match):
    stash = Stash()
    marker = stash.marker(AI, 'common', 'old')
    marker.update(seconds=10, end_seconds=observed_end)
    saved = {'contributions': [span('wd', 10, saved_end)]}
    before = deepcopy(saved)
    if match:
        result = reconcile_marker_snapshot(stash.find_scene('1'), [saved])
        assert result['contributions']
    else:
        with pytest.raises(RuntimeError, match='Cannot identify surviving marker'):
            reconcile_marker_snapshot(stash.find_scene('1'), [saved])
    assert saved == before and not stash.writes


def test_current_temporal_publication_remains_replayable(tmp_path):
    value = old_publication()
    item = value['contributions'][0]
    item['meta']['analysis_format'] = 5
    item['build_options']['algorithm'] = 'temporal'
    item['extra_support'] = {}
    path = tmp_path/'state.sqlite3'
    with StateStore(path, 'scope') as state:
        state.connection.execute('INSERT INTO snapshots VALUES (?,?,?,?,NULL)',
            ('scope', '1', 'markers', encode({'__stash_ai_snapshot__': 5, 'snapshot': value, 'proofs': []})))
        state.connection.execute('PRAGMA user_version=18')
        state.connection.commit()
    update_database(path)
    value['contributions'][0]['build_options'].pop('algorithm')
    with StateStore(path, 'scope') as state:
        assert state.snapshot('1', 'markers')[0] == value
        require_rebased(value['contributions'])


@pytest.mark.parametrize('damage', ['algorithm', 'format'])
def test_unknown_publication_contract_rolls_back_migration(tmp_path, damage):
    path = tmp_path/'state.sqlite3'
    with StateStore(path, 'scope'):
        pass
    value = old_publication()
    if damage == 'algorithm':
        value['contributions'][0]['build_options']['algorithm'] = 'future-engine'
    else:
        value['contributions'][0]['meta']['analysis_format'] = 99
    packed = encode({'__stash_ai_snapshot__': 4, 'snapshot': value, 'proofs': []})
    with sqlite3.connect(path) as connection:
        connection.execute('INSERT INTO snapshots VALUES (?,?,?,?,NULL)', ('scope', '1', 'markers', packed))
        connection.execute('PRAGMA user_version=18')
    with pytest.raises(ValueError, match='Unsupported'):
        update_database(path)
    with sqlite3.connect(path) as connection:
        assert connection.execute('PRAGMA user_version').fetchone() == (18,)
        assert connection.execute('SELECT published FROM snapshots').fetchone()[0] == packed
