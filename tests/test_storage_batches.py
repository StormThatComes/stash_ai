"""Large selections preserve cache, preflight and bounded SQL/application."""
import pytest

from stash_ai.rethreshold.execution import run
from stash_ai.runtime.report import RunReport
from stash_ai.storage import progress
from stash_ai.storage.store import StateStore
from tests.helpers.runtime import runtime as app
from tests.test_result_thresholds import blobs, config, outputs


@pytest.mark.parametrize('invalid_last', [False, True])
def test_batched_rethreshold_keeps_full_preflight_and_raw_blobs(tmp_path, invalid_last):
    with StateStore(tmp_path / 'state.sqlite3', 'scope', cache_bytes=2**20) as store:
        scenes = [str(i) for i in range(260)]
        for scene in scenes:
            value = outputs()
            if invalid_last and scene == scenes[-1]:
                value['scene']['meta']['scene_score_floor'] = .95
            store.put(scene, 'wd', value)
        before = blobs(store)
        statements = []
        store.connection.set_trace_callback(statements.append)
        result = run(app, store, scenes, [config(.8)], RunReport(write=lambda _: None),
                     scenes=True, markers=True)
        store.connection.set_trace_callback(None)
        assert result == ({'errors': 1} if invalid_last else {'scene_thresholds_updated': 260})
        assert blobs(store) == before
        reads = [sql for sql in statements if sql.startswith('SELECT r.scene,r.model,r.kind,m.format')]
        assert len(reads) == 3
        assert not any('SELECT data FROM results' in sql for sql in statements)
        writes = [sql for sql in statements if sql.startswith('INSERT INTO result_thresholds')]
        assert len(set(writes)) == (0 if invalid_last else 520)
        assert sum(sql == 'COMMIT' for sql in statements) <= 6


def test_selected_queries_and_preloads_do_not_include_other_scenes(tmp_path):
    with StateStore(tmp_path / 'state.sqlite3', 'scope', cache_bytes=2**20) as store:
        for scene in ('1', '2'):
            store.put(scene, 'wd', outputs())
            progress.block_publication(store, [scene], 'wd', ['scene'])
        assert store.result_presence(['wd'], ['1']) == {('1', 'wd', 'scene'), ('1', 'wd', 'markers')}
        assert store.result_index_presence(['wd'], []) == set()
        assert progress.pending_refreshes(store, ['scene'], ['1']) == {'1': [('wd', 'scene')]}
        store.cache.clear()
        store.preload_sources(['1'])
        progress.preload_status(store, ['1'])
        assert store.cached(('sources', '1', 'scene')) == {'wd'}
        assert store.cached(('sources', '2', 'scene')) is None
        assert store.cached(('status', '2', 'wd')) is None
        # More than a conservative SQLite parameter limit, with duplicate IDs.
        selection = [str(i) for i in range(1700)] + ['1']
        assert len(store.result_presence(['wd'], selection)) == 4
