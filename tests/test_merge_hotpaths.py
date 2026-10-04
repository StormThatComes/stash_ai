"""Range-query equivalence, native snapshot reads and useful phase timings."""
from bisect import bisect_left
from collections import Counter
from dataclasses import replace
import json
import random
import sqlite3
from types import SimpleNamespace

import pytest

from stash_ai import cli as app

from stash_ai.evidence.observations import ObservationIndex
from stash_ai.stash_api.sql import SceneReader, OPERATION, SQL, is_read_only_merge
from stash_ai.performance import scene_fingerprint
from stash_ai.runtime import RamCache, RunReport, retained_size
from stash_ai.evidence.temporal.policy import TemporalPolicy
from stash_ai.evidence.timeline import support_size

pytestmark = pytest.mark.usefixtures("supported_stash_version")


@pytest.mark.parametrize('overlap', [False, True])
@pytest.mark.parametrize('short', [False, True])
def test_index_matches_reference_scanner_across_ranges_sources_and_gaps(overlap, short):
    rng = random.Random(513)
    events, start = [], 0.0
    for i in range(60):
        width = rng.choice([.25, 1., 2.])
        extras = {}
        for tag in ('a', 'b', 'c', 'd'):
            sources = {source: rng.choice([.8, .85, .97]) for source in ('wd','dtq','oppai',None) if rng.random() < .4}
            if sources:
                extras[tag] = sources
        events.append(dict(start=start, end=start+width, extras=extras))
        start += width + rng.choice([0., 0., .0005, .01, 1.])
    if overlap:
        events[0]['end'] = events[-1]['end'] + 1
    index = ObservationIndex(events, {s:[s+'-sig'] for s in ('wd','dtq','oppai')}, 'timeline', RamCache(10000000))
    assert index.disjoint is not overlap
    for _ in range(80):
        start, end = sorted(rng.uniform(-1, events[-1]['end']+1) for _ in range(2))
        policy = replace(TemporalPolicy(), short_marker_extras=short, extra_min_hits=rng.choice([1,2,4,50]))
        allowed = rng.choice([None, frozenset(), frozenset({'wd'}), frozenset({'wd','dtq'}), frozenset({None})])
        expected = index.scan_support(bisect_left(index.starts,start),bisect_left(index.starts,end),end,policy,allowed)
        actual = index.support(start,end,policy,allowed)
        assert actual == expected
        assert support_size(actual) >= retained_size(actual)


def test_exact_boundaries_skip_empty_queries_and_reuse_qualification():
    events = [dict(start=i, end=i+1, extras={'a': {'wd': .9}}) for i in range(10)]
    index = ObservationIndex(events, {'wd':['sig']}, 'timeline', RamCache(1000000))
    policy = TemporalPolicy()
    assert index.support(3,3,policy) == {}
    result = index.support(3,5,policy)['a']
    assert result['hits'] == 2 and result['intervals'] == [[3,5]]
    assert result['runs'] == [[3,5,3,5]]
    first = index.qualified(0,10,policy)
    assert index.qualified(0,10,policy) is first
    assert index.support(0,10,policy)['a']['hits'] == 10
    cache = index.cache
    cache.put('unrelated', 'retained')
    proof = index.support(0,10,policy)
    index.close()
    assert list(cache.entries) == ['unrelated']
    assert proof['a']['intervals'] == [[0,10]]


SCHEMA = '''
CREATE TABLE scenes(id INTEGER PRIMARY KEY,title TEXT);
CREATE TABLE tags(id INTEGER PRIMARY KEY,name TEXT);
CREATE TABLE scenes_tags(scene_id INTEGER,tag_id INTEGER);
CREATE TABLE scene_markers(id INTEGER PRIMARY KEY,scene_id INTEGER,title TEXT,seconds REAL,end_seconds REAL,primary_tag_id INTEGER);
CREATE TABLE scene_markers_tags(scene_marker_id INTEGER,tag_id INTEGER);
CREATE INDEX st_scene ON scenes_tags(scene_id);
CREATE INDEX m_scene ON scene_markers(scene_id);
CREATE INDEX mt_marker ON scene_markers_tags(scene_marker_id);
INSERT INTO scenes VALUES (1,'first'),(2,'second');
INSERT INTO tags VALUES (1,'ai:core'),(2,'ai:detail'),(3,'manual'),(4,'ai:marker:stash-taggers');
INSERT INTO scenes_tags VALUES (1,1),(1,3);
'''


@pytest.fixture
def database():
    db = sqlite3.connect(':memory:')
    db.executescript(SCHEMA)
    with db:
        for i in range(1,301):
            db.execute('INSERT INTO scene_markers VALUES (?,?,?,?,?,?)',(i,1,'[AI] core',i+.25,i+1.5,1))
            db.executemany('INSERT INTO scene_markers_tags VALUES (?,?)',[(i,2),(i,3),(i,4)])
    yield db
    db.close()


def native_scene(db, identifier, markers=True, end=True, full=True):
    row = db.execute('SELECT id,title FROM scenes WHERE id=?',(identifier,)).fetchone()
    if row is None:
        return None
    scene = dict(id=str(row[0]),title=row[1],paths=dict(sprite='/sprite',vtt='/vtt'))
    if not full:
        return scene
    tags = {str(i):dict(id=str(i),name=n) for i,n in db.execute('SELECT id,name FROM tags')}
    scene['tags'] = [tags[str(i)] for i, in db.execute('SELECT tag_id FROM scenes_tags WHERE scene_id=?',(identifier,))]
    if markers:
        scene['scene_markers'] = [dict(id=str(i),title=t,seconds=s,end_seconds=e if end else None,
            primary_tag=tags[str(p)],tags=[tags[str(v)] for v, in db.execute('SELECT tag_id FROM scene_markers_tags WHERE scene_marker_id=?',(i,))])
            for i,t,s,e,p in db.execute('SELECT id,title,seconds,end_seconds,primary_tag_id FROM scene_markers WHERE scene_id=?',(identifier,))]
    return scene


def client_for(db, *, end=True, deny=False):
    client = app.create_stash_client(True)
    if not end:
        db.execute("UPDATE scene_markers SET end_seconds=NULL")
    calls = []
    def post(url, **kwargs):
        payload = kwargs['json']
        query, variables = payload['query'], payload['variables']
        calls.append((query,variables))
        if query == OPERATION:
            if deny:
                result = dict(errors=[dict(message='SQL disabled')])
            else:
                cursor = db.execute(variables['sql'],variables['args'])
                result = dict(data={'querySQL':dict(columns=[c[0] for c in cursor.description],rows=[list(row) for row in cursor])})
        else:
            full = 'tags {' in query
            if 'StashAIScenes(' in query:
                scenes = [native_scene(db, i, app.CREATE_MARKERS, end, full) for i in (1,2)]
                page = variables['filter']
                offset = (page['page']-1)*page['per_page']
                scenes = scenes[offset:offset+page['per_page']]
                result = dict(data={'findScenes': {'scenes':scenes}})
            elif 'StashAIScene(' in query:
                result = dict(data={'findScene': native_scene(db,variables['id'],app.CREATE_MARKERS,end,full)})
            else:
                raise AssertionError('Unexpected operation')
        return SimpleNamespace(status_code=200,raise_for_status=lambda:None,json=lambda:result)
    client.api.close()
    client.api = SimpleNamespace(post=post,close=lambda:None)
    client.calls = calls
    return client


@pytest.mark.parametrize('end,markers', [(True,True),(False,True),(False,False)])
def test_sql_snapshots_match_native_fields_without_relation_resolvers(database, monkeypatch, end, markers):
    monkeypatch.setattr(app,'CREATE_MARKERS',markers)
    client = client_for(database,end=end)
    report = RunReport(write=lambda line: None)
    changed = database.total_changes
    try:
        client.configure_merge_reads(report)
        assert report.counts['merge_sql_enabled'] == 1
        scenes = list(client.iter_scenes())
        assert len(scenes) == 2
        for scene in scenes:
            assert scene_fingerprint(scene) == scene_fingerprint(native_scene(database,scene['id'],markers,end))
        before = len(client.calls)
        scene = client.find_scene_for_merge('1')
        assert scene_fingerprint(scene) == scene_fingerprint(scenes[0])
        assert len(client.calls) == before+1 and client.calls[-1][0] == OPERATION
        assert 'paths' in client.find_scene('1')
        assert report.timing_calls['stash_read'] > 0 and report.timing_calls['stash_write'] == 0
        assert database.total_changes == changed
        for query, _ in client.calls:
            if query != OPERATION:
                assert 'scene_markers' not in query and 'tags {' not in query
    finally:
        client.close()


def test_sql_read_fallback_and_dry_run_guard(database, monkeypatch):
    monkeypatch.setattr(app,'CREATE_MARKERS',True)
    client = client_for(database,deny=True)
    report = RunReport(write=lambda line:None)
    try:
        client.configure_merge_reads(report)
        assert client.merge_reader is None and report.counts['merge_graphql_fallback'] == 1
        assert scene_fingerprint(client.find_scene_for_merge('1')) == scene_fingerprint(native_scene(database,1))
        for sql in ('DELETE FROM scene_markers','SELECT 1',SQL['ranges']+'; DELETE FROM scenes'):
            assert not is_read_only_merge(OPERATION,{'sql':sql})
            with pytest.raises(RuntimeError,match='mutation attempted'):
                client.gql(OPERATION,{'sql':sql,'args':[]})
    finally:
        client.close()


def test_sql_read_uses_relation_indexes_and_reports_deleted_scene(database):
    for mode, sql in SQL.items():
        plan = ' '.join(row[-1] for row in database.execute('EXPLAIN QUERY PLAN '+sql,
            (0,10) if mode.startswith('page_') else ('[1]',)))
        assert 'st_scene' in plan
        if not mode.endswith('tags'):
            assert 'm_scene' in plan and 'mt_marker' in plan
    client = client_for(database)
    client.configure_merge_reads(RunReport(write=lambda line:None))
    with pytest.raises(RuntimeError,match='no longer exists'):
        client.find_scene_for_merge('999')
    client.close()


def test_merge_only_pages_skip_graphql_metadata_and_paths(database, monkeypatch):
    monkeypatch.setattr(app,'SCENE_PAGE_SIZE',1)
    client = client_for(database)
    try:
        client.configure_merge_reads(RunReport(write=lambda line:None),merge_only=True)
        assert [s['id'] for s in client.iter_scenes()] == ['1','2']
        assert all(query == OPERATION for query, _ in client.calls)
        assert all('OFFSET' not in params['sql'] and 'COUNT(' not in params['sql'] for _, params in client.calls)
        assert list(client.iter_scenes(['1']))[0]['id'] == '1'
        assert client.calls[-1][0] == OPERATION
    finally:
        client.close()


def test_sql_page_failure_falls_back_without_repeating_visited_scenes(database, monkeypatch):
    monkeypatch.setattr(app,'SCENE_PAGE_SIZE',1)
    client = client_for(database)
    report = RunReport(write=lambda line:None)
    client.configure_merge_reads(report,merge_only=True)
    original = client.merge_reader.page
    def page(after, size):
        if after:
            raise RuntimeError('temporary failure')
        return original(after,size)
    client.merge_reader.page = page
    try:
        assert [s['id'] for s in client.iter_scenes()] == ['1','2']
        assert report.counts['merge_graphql_fallback'] == 1
    finally:
        client.close()


@pytest.mark.parametrize('page_size,expected', [(2, [2, 2, 1]), (5000, [5])])
def test_explicit_sql_ids_are_batched_deduplicated_and_keep_order(database, page_size, expected):
    database.executemany('INSERT INTO scenes VALUES (?,?)', [(i, str(i)) for i in range(3, 6)])
    client = client_for(database)
    client.scene_page_size = page_size
    client.configure_merge_reads(RunReport(write=lambda line: None), merge_only=True)
    client.calls.clear()
    scenes = list(client.iter_scenes(['5', '2', '5', '1', '4', '3']))
    assert [s['id'] for s in scenes] == ['5', '2', '1', '4', '3']
    assert [len(json.loads(v['args'][0])) for _, v in client.calls] == expected
    assert all(q == OPERATION for q, _ in client.calls)
    assert all('paths' not in s for s in scenes)


def test_explicit_sql_batches_cap_size_and_fetch_lazily(database):
    database.executemany('INSERT INTO scenes VALUES (?,?)', [(i, str(i)) for i in range(3, 1003)])
    client = client_for(database)
    client.scene_page_size = 5000
    client.configure_merge_reads(RunReport(write=lambda line: None), merge_only=True)
    client.calls.clear()
    scenes = client.iter_scenes([str(i) for i in range(1, 1003)])
    assert next(scenes)['id'] == '1'
    assert len(client.calls) == 1
    assert len(json.loads(client.calls[0][1]['args'][0])) == 1000
    assert len(list(scenes)) == 1001
    assert len(json.loads(client.calls[1][1]['args'][0])) == 2


def test_explicit_batch_failure_falls_back_without_replay(database):
    database.executemany('INSERT INTO scenes VALUES (?,?)', [(i, str(i)) for i in range(3, 6)])
    client = client_for(database)
    client.scene_page_size = 2
    report = RunReport(write=lambda line: None)
    client.configure_merge_reads(report, merge_only=True)
    read = client.merge_reader.read
    def fail_second(ids):
        if '4' in ids:
            raise RuntimeError('snapshot unavailable')
        return read(ids)
    client.merge_reader.read = fail_second
    assert [s['id'] for s in client.iter_scenes(['5', '2', '5', '1', '4', '3'])] == ['5', '2', '1', '4', '3']
    assert report.counts['merge_graphql_fallback'] == 1
    assert [v['id'] for q, v in client.calls if q != OPERATION] == ['1', '4', '3']
    assert 'paths' in client.find_scene('1')


def test_batched_missing_scene_remains_an_error(database):
    client = client_for(database)
    client.configure_merge_reads(RunReport(write=lambda line: None), merge_only=True)
    with pytest.raises(RuntimeError, match='Scene 999 no longer exists'):
        list(client.iter_scenes(['1', '999']))


def test_snapshot_only_refresh_reads_current_sql_state(database):
    client = client_for(database)
    client.configure_merge_reads(RunReport(write=lambda line: None), merge_only=True)
    old = list(client.iter_scenes(['2']))[0]
    database.execute('INSERT INTO scenes_tags VALUES (2,3)')
    current = client.find_scene('2')
    assert old['tags'] == [] and current['tags'] == [{'id': '3', 'name': 'manual'}]
    assert all(q == OPERATION for q, _ in client.calls)


def test_exclusive_timers_do_not_double_count_and_survive_failures(monkeypatch):
    ticks = iter([0.,1.,4.,10.])
    monkeypatch.setattr('stash_ai.runtime.time.perf_counter',lambda:next(ticks))
    report = RunReport(write=lambda line:None)
    with pytest.raises(ValueError):
        with report.measure('merge_plan'):
            with report.measure('stash_read'):
                raise ValueError('failed read')
    assert report.timings == {'stash_read':3.,'merge_plan':7.}
    assert report.timing_calls == {'stash_read':1,'merge_plan':1}


def test_publication_check_and_snapshots_share_one_commit(tmp_path):
    from stash_ai.storage.store import StateStore
    from stash_ai.performance import checked
    with StateStore(tmp_path/'state.db','s') as state:
        state.prepare('1', {'scene': {'contributions': []}})
        statements = []
        state.connection.set_trace_callback(statements.append)
        state.commit_prepared('1', ['scene'], publication=('p','f'))
        assert checked(state,'1','p','f')
        assert sum(sql == 'COMMIT' for sql in statements) == 1
