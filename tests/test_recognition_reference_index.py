import sqlite3
from contextlib import closing
from dataclasses import replace

import numpy as np
import pytest

from stash_ai.recognition.clustering import _groups, cluster_scene, reference_context
from stash_ai.recognition.database import FaceDatabase
from stash_ai.recognition.reference_index import _active, references
from tests.test_recognition_clustering import setup, scan


@pytest.mark.parametrize('storage', ['memory', 'disk'])
def test_lazy_references_initialize_while_prefetch_holds_schema_read_lock(tmp_path, storage):
    from stash_ai.recognition.config import plain
    from stash_ai.recognition.rebuild import REPLAY_SCHEMA
    from stash_ai.recognition.reference_index import current
    from stash_ai.recognition.replay_memory import memory_replay

    config = setup(tmp_path)
    data = plain(config.data)
    data['execution']['rebuild_storage'] = storage
    config = replace(config, data=data)
    with memory_replay(config) as (db, uri):
        db.executescript(REPLAY_SCHEMA)
        with references(db, 'scope', 'ns', config) as lazy:
            assert lazy.value is None
            with closing(sqlite3.connect(uri, uri=True)) as reader:
                # Keep a real cursor open, making the race deterministic without
                # sleeps: shared-cache readers prohibit main-schema writes.
                with closing(reader.execute('SELECT name FROM sqlite_master')) as rows:
                    assert rows.fetchone() is not None
                    index = current(db)
                    assert index is not None and index is current(db)


def baseline(db, config):
    token = _active.set(None)
    try:
        return _groups(db, 'scope', 'ns', config.data['clustering'])
    finally:
        _active.reset(token)


def test_incremental_references_match_full_loader_after_move_retract_and_alias(tmp_path):
    config = setup(tmp_path, {'max_samples_per_person': 2})
    with FaceDatabase(config.database) as state:
        first, _ = scan(state, config, '10')
        second, _ = scan(state, config, '2', [np.array([0., 1., 0, 0])]*3)
        cluster_scene(state, first, config)
        cluster_scene(state, second, config)
        db = state.connection
        people = [row[0] for row in db.execute('SELECT id FROM people ORDER BY created,id')]
        with references(db, 'scope', 'ns', config):
            def compare():
                full = baseline(db, config)
                cached = _groups(db, 'scope', 'ns', config.data['clustering'])
                assert reference_context(cached) == reference_context(full)
                assert [g.identifier for g in cached] == [g.identifier for g in full]
            compare()
            db.execute('UPDATE members SET person=? WHERE person=?', (people[0], people[1]))
            compare()
            observation = db.execute('SELECT observation FROM members ORDER BY observation LIMIT 1').fetchone()[0]
            db.execute('DELETE FROM members WHERE observation=?', (observation,))
            compare()
            db.execute("INSERT INTO identity_merges VALUES (?,?,1,'{}','local')", (people[1], people[0]))
            compare()
        assert not list(db.execute("SELECT name FROM sqlite_temp_master WHERE type='trigger'"))


def test_unchanged_snapshot_does_not_read_or_validate_history_again(tmp_path, monkeypatch):
    from stash_ai.recognition import clustering
    config = setup(tmp_path)
    with FaceDatabase(config.database) as state:
        identifier, _ = scan(state, config, '1')
        cluster_scene(state, identifier, config)
        with references(state.connection, 'scope', 'ns', config):
            before = _groups(state.connection, 'scope', 'ns', config.data['clustering'])
            calls = []
            state.connection.set_trace_callback(calls.append)
            monkeypatch.setattr(clustering, 'vector', lambda _: pytest.fail('unchanged vector revalidated'))
            after = _groups(state.connection, 'scope', 'ns', config.data['clustering'])
            assert reference_context(before) == reference_context(after)
            assert not calls
            state.connection.set_trace_callback(None)


def test_invalid_new_membership_is_rejected_and_cleanup_preserves_primary(tmp_path):
    config = setup(tmp_path)
    with FaceDatabase(config.database) as state:
        identifier, _ = scan(state, config, '1')
        cluster_scene(state, identifier, config)
        state.connection.execute("UPDATE observations SET rejected='bad'")
        state.connection.commit()
        with pytest.raises(ValueError, match='incompatible'):
            with references(state.connection, 'scope', 'ns', config):
                _groups(state.connection, 'scope', 'ns', config.data['clustering'])
        assert not list(state.connection.execute("SELECT name FROM sqlite_temp_master WHERE type='trigger'"))
