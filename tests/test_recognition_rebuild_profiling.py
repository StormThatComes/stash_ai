from dataclasses import replace

import pytest

from stash_ai.recognition.config import plain
from stash_ai.recognition.database import FaceDatabase
from stash_ai.recognition.rebuild import rebuild_selected
from tests.test_recognition_clustering import setup, scan


def profiled(config):
    data = plain(config.data)
    data['execution']['profile_rebuild'] = True
    return replace(config, data=data)


def test_profile_preserves_cold_replay_and_borrowed_sql_trace(tmp_path):
    config = setup(tmp_path)
    with FaceDatabase(config.database) as state:
        first, _ = scan(state, config, '1')
        second, _ = scan(state, config, '2')
        selected = [dict(scene='1', scan=first), dict(scene='2', scan=second)]
        expected = rebuild_selected(state, selected, config)
        members = [tuple(row) for row in state.connection.execute('SELECT * FROM members ORDER BY observation')]
        state.connection.execute("DELETE FROM metadata WHERE key LIKE 'identity-rebuild:%'")
        state.connection.commit()
        trace, raw = [], state.connection
        raw.set_trace_callback(trace.append)
        assert rebuild_selected(state, selected, profiled(config)) == expected
        assert state.connection is raw
        assert members == [tuple(row) for row in raw.execute('SELECT * FROM members ORDER BY observation')]
        metrics = state.last_rebuild_profile
        assert metrics['counts']['validated_vectors'] > 0
        assert metrics['counts']['sample_comparisons'] > 0
        assert metrics['stages']['scene']['calls'] == 2
        assert metrics['sql'] and len(metrics['sql']) <= 50
        assert trace and not any(first in key or second in key for key in metrics['sql'])
        trace.clear()
        raw.execute('SELECT 1')
        assert trace == ['SELECT 1']


def test_profile_restores_connection_and_records_primary_failure(tmp_path):
    config = profiled(setup(tmp_path))
    with FaceDatabase(config.database) as state:
        identifier, _ = scan(state, config, '1')
        state.connection.execute('UPDATE observations SET embedding=?', (b'bad',))
        state.connection.commit()
        raw = state.connection
        with pytest.raises(ValueError, match='embedding'):
            rebuild_selected(state, [dict(scene='1', scan=identifier)], config)
        assert state.connection is raw
        assert state.last_rebuild_profile['stages']['rebuild']['errors'] == 1
        assert raw.execute('SELECT COUNT(*) FROM members').fetchone()[0] == 0


def test_existing_state_acquires_indexes_without_changing_evidence(tmp_path):
    config = setup(tmp_path)
    with FaceDatabase(config.database) as state:
        identifier, _ = scan(state, config, '1')
        before = [tuple(row) for row in state.connection.execute('SELECT * FROM observations')]
        state.connection.execute('DROP INDEX candidates_scan_policy')
        state.connection.execute('DROP INDEX candidates_person')
        state.connection.commit()
    with FaceDatabase(config.database) as state:
        assert before == [tuple(row) for row in state.connection.execute('SELECT * FROM observations')]
        plan = ' '.join(row[3] for row in state.connection.execute(
            'EXPLAIN QUERY PLAN SELECT person,status,COUNT(*) FROM candidates '
            'WHERE scan=? AND policy=? GROUP BY person,status', (identifier, 'p')))
        assert 'SEARCH candidates USING' in plan and 'candidates_scan_policy' in plan
