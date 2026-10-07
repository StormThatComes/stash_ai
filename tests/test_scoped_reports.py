"""Small selections stay indexed and preserve sparse preview semantics."""
import pytest

from stash_ai.storage.store import StateStore
from stash_ai.reports import cache
from stash_ai.reports.review import report as review
from stash_ai.reports.model import report as model
from tests.test_result_thresholds import outputs


def seed(store, count=1600):
    store.put('selected', 'wd', outputs())
    with store.connection:
        for table in ('results', 'result_metadata', 'result_index'):
            row = store.connection.execute('SELECT * FROM ' + table).fetchone()
            sql = 'INSERT INTO ' + table + ' VALUES (' + ','.join('?' for _ in row) + ')'
            store.connection.executemany(sql, ((row[0], str(i), *row[2:]) for i in range(count)))


@pytest.mark.parametrize('preview', [False, True])
def test_metadata_batch_uses_full_keys_without_scanning_unselected_rows(tmp_path, preview):
    path = tmp_path / 'state.sqlite3'
    with StateStore(path, 'scope') as store:
        seed(store)
    with StateStore(path, 'scope', dry_run=preview) as store:
        plans = []
        read = store.read_rows
        def inspect(sql, params):
            plans.extend(row[3] for row in store.connection.execute('EXPLAIN QUERY PLAN ' + sql, params))
            return read(sql, params)
        store.read_rows = inspect
        with store.metadata.batch([('selected', 'wd'), ('selected', 'wd'), ('missing', 'wd')]):
            expected = store.result_metadata('selected', 'scene', 'wd')
            assert expected is not None
            assert store.result_metadata('missing', 'scene', 'wd') is None
            with store.metadata.batch([]):
                assert store.result_metadata('selected', 'scene', 'wd') is None
            assert store.result_metadata('selected', 'scene', 'wd') == expected
        assert any('scope=? AND scene=? AND model=?' in plan for plan in plans)
        assert not any(plan.startswith(('SCAN s', 'SCAN main.')) for plan in plans)
        # The preview overlay and its deletion mask remain authoritative.
        if preview:
            store.set_result_threshold('selected', 'wd', 'scene', .7, outputs()['scene'])
            with store.metadata.batch([('selected', 'wd')]):
                from stash_ai.storage.thresholds import effective
                assert effective(store.result_metadata('selected', 'scene', 'wd')) == .7
            from stash_ai.storage.preview import delete
            with store.connection:
                delete(store, 'results', 'selected', 'wd', 'scene')
            with store.metadata.batch([('selected', 'wd')]):
                assert store.result_metadata('selected', 'scene', 'wd') is None


def test_report_selection_queries_are_bounded_and_preserve_explicit_order(tmp_path):
    with StateStore(tmp_path / 'state.sqlite3', 'scope') as store:
        seed(store)
        statements = []
        store.connection.set_trace_callback(statements.append)
        ids = [str(i) for i in range(1100)] + ['selected', 'missing', '2']
        expected = list(dict.fromkeys(ids[:-2]))
        assert model._scene_ids(store, ids) == expected
        assert len([sql for sql in statements if sql.startswith('SELECT DISTINCT scene')]) == 2
        assert all('scene IN (' in sql for sql in statements if sql.startswith('SELECT DISTINCT scene'))
        statements.clear()
        assert review._scene_ids(store, ['selected', '2', 'selected']) == ['selected', '2']
        assert len([sql for sql in statements if sql.startswith('SELECT DISTINCT scene')]) == 1
        with pytest.raises(ValueError, match='missing'):
            review._scene_ids(store, ['selected', 'missing'])
        statements.clear()
        assert cache._keys(store, 'result_index', 'scene,model,kind', ['selected'])
        assert 'scene IN (' in statements[-1]
        assert not cache._keys(store, 'result_index', 'scene,model,kind', [])


def test_one_pass_scene_iterator_still_blocks_pending_publication(tmp_path):
    from tests.test_result_thresholds import rethreshold, blobs
    with StateStore(tmp_path / 'state.sqlite3', 'scope') as store:
        store.put('1', 'wd', outputs())
        store.prepare('1', {'scene': {'contributions': []}})
        before = blobs(store)
        assert rethreshold(store, .8, scenes=iter(['1'])) == {'errors': 1}
        assert blobs(store) == before
