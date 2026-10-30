from contextlib import closing
from dataclasses import replace
import sqlite3

from benchmarks.benchmark_merge_solver import fixture
from stash_ai.recognition.config import plain
from stash_ai.recognition.fragmentation_report import _open_readonly
from stash_ai.recognition.offline_evaluation import _ram
from stash_ai.recognition.replay_simulation import prepared_history
from stash_ai.recognition.trial_inputs import prepare


def test_preparation_has_minimal_private_rows_and_no_vector_statistics_scan(tmp_path):
    config, goals = fixture(tmp_path, 'body_modes')
    with closing(_open_readonly(config.database)) as source:
        installation = source.execute("SELECT value FROM metadata WHERE key='installation'").fetchone()[0]
        with prepared_history(source, config, goals['scope'], goals['namespace'], installation) as seed:
            queries = []
            seed.set_trace_callback(queries.append)
            with closing(_ram(config)) as target:
                before = list(target.execute("SELECT name,sql FROM sqlite_master WHERE type='trigger' ORDER BY name"))
                prepare(target, seed)
                assert all(r[0] == '{}' for r in target.execute('SELECT quality FROM main.observations'))
                assert any(r[0] != '{}' for r in target.execute('SELECT quality FROM observations'))
                assert not target.execute('PRAGMA foreign_key_check').fetchall()
                assert list(target.execute("SELECT name,sql FROM sqlite_master WHERE type='trigger' ORDER BY name")) == before
                assert not any('SUM(length(embedding))' in q for q in queries)
            with prepared_history(seed, config, goals['scope'], goals['namespace'], installation) as reused:
                assert reused is seed


def test_disk_snapshot_does_not_consume_private_ram_limit(tmp_path):
    config, goals = fixture(tmp_path, 'face')
    data = plain(config.data); data['execution']['rebuild_memory_bytes'] = 1024
    config = replace(config, data=data)
    with closing(_open_readonly(config.database)) as source:
        installation = source.execute("SELECT value FROM metadata WHERE key='installation'").fetchone()[0]
        with prepared_history(source, config, goals['scope'], goals['namespace'], installation) as seed:
            assert seed.execute('PRAGMA page_count').fetchone()[0] * 4096 > 1024


def test_input_admission_bounds_references_but_keeps_both_private_database_limits(tmp_path):
    from stash_ai.recognition.solver_execution import plan
    config, goals = fixture(tmp_path, 'face')
    ceiling = plan(config)
    with closing(_open_readonly(config.database)) as source:
        measured = plan(config, source=source, scope=goals['scope'], namespace=goals['namespace'])
    assert measured.shared_input_bytes > 0
    assert 2*config.data['execution']['rebuild_memory_bytes'] < measured.trial_bytes < ceiling.trial_bytes


def test_copy_filters_relationships_in_sql(tmp_path):
    from stash_ai.recognition.evaluation_scenarios import copy_history
    config, goals = fixture(tmp_path, 'face')
    with closing(_open_readonly(config.database)) as source, closing(_ram(config)) as target:
        queries = []; source.set_trace_callback(queries.append)
        result = copy_history(source, target, [], goals['scope'], goals['namespace'])
        assert result['observations'] == 6
        for table in ('members','candidates','identity_origins','identity_merges','merge_support'):
            assert any(q.startswith('SELECT * FROM '+table+' WHERE ') for q in queries)
            assert not any(q == 'SELECT * FROM '+table+' ORDER BY rowid' for q in queries)
