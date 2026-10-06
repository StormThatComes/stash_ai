"""Real SQLite SELECT contracts and API-write semantics for large cleanup."""
from tests.helpers.evidence import empty_policies as empty_tag_policies
from tests.helpers.evidence import inference_record
from stash_ai.storage.results import KINDS
from stash_ai.stash_api.client import StashClient
from collections import Counter
from copy import deepcopy
import json
import sqlite3

import pytest

from tests.helpers.runtime import runtime as app
from stash_ai.storage import codec as state_codec
from stash_ai.publication import merged as merge_sync
from stash_ai.cleanup import sql as fast
from stash_ai.cleanup.sql_session import CleanupSqlSession, READ_OPERATION, SQL, is_read_only_cleanup
import stash_ai.evidence.labels as merge_labels
import stash_ai.storage.results as merge_results
import stash_ai.storage.store as merge_store
from stash_ai.cleanup.client import CleanupClient
from stash_ai.models import WD
from tests.helpers.cleanup import FastStash
from tests.helpers.cleanup import install
from tests.helpers.cleanup import options
from tests.helpers.evidence import hit
from tests.helpers.evidence import span


from tests.helpers.cleanup_sql import SCHEMA
from tests.helpers.cleanup_sql import SQLStash
pytestmark = pytest.mark.usefixtures("reset_progress")


@pytest.mark.parametrize('strategy', ['sql', 'scenes'])
def test_cleanup_and_exclusion_pruning_preserve_retired_namespaces(tmp_path, monkeypatch, strategy):
    from tests.helpers.legacy_namespaces import NAMESPACES
    stash = SQLStash()
    for namespace in NAMESPACES.values():
        tag = stash.tag(namespace.prefix + 'blocked')
        stash.scene['tags'].append(tag)
        stash.marker(namespace, 'blocked', str(len(stash.scene['scene_markers']) + 1))
    legacy = deepcopy((stash.scene, stash.tags))
    stash.scene['tags'].append(stash.tag('ai:blocked'))
    stash.marker(merge_labels.AI, 'blocked', '99', manual=True)
    install(monkeypatch, stash)
    assert app.main([*args(tmp_path, ['blocked']), '--cleanup-strategy', strategy]) == 0
    assert stash.scene == legacy[0]
    assert all(stash.tags[name] == tag for name, tag in legacy[1].items())
    assert 'ai:blocked' not in stash.tags


@pytest.mark.parametrize('mode,kind,operation', [
    ('cleanup-markers-only', 'markers', 'CleanupDeleteMarkers'),
    ('cleanup-scene-tags-only', 'scene', 'CleanupRemoveTags'),
])
def test_pending_partial_cleanup_requires_explicit_fingerprint_migration(tmp_path, monkeypatch, mode, kind, operation):
    from stash_ai.signatures import digest
    from stash_ai.models import SOURCES
    stash = SQLStash()
    stash.scene['tags'] = [stash.tag('ai:blocked')]
    stash.marker(merge_labels.AI, 'blocked', '1', manual=True)
    stash.fail_after = operation
    install(monkeypatch, stash)
    command = args(tmp_path, ['blocked'], mode=mode)
    assert app.main(command) == 1
    old = {'tags': {source: ['blocked'] for source in SOURCES},
           'cores': {source: [] for source in SOURCES}, 'marker_blocklist': [], 'output_kinds': [kind]}
    path = tmp_path/'state.sqlite3'
    with merge_store.StateStore(path, merge_store.scope_for(app.STASH_URL), allow_cleanup=True) as store:
        assert store.cleanup_pending == digest({key: value for key, value in old.items() if key != 'marker_blocklist'})
        with store.connection:
            store.connection.execute('UPDATE cleanup_state SET pending=?', (digest(old),))
    stash.writes.clear()
    assert app.main(args(tmp_path, ['different'], mode=mode)) == 1
    assert not stash.writes
    assert app.main(args(tmp_path, ['blocked'], mode=mode)) == 1
    assert not stash.writes
    from stash_ai.tools.cleanup_pending_migration import migrate
    with merge_store.StateStore(path, merge_store.scope_for(app.STASH_URL), allow_cleanup=True) as store:
        assert migrate(store, old['tags'], old['cores'], [kind])
    assert app.main(args(tmp_path, ['blocked'], mode=mode)) == 0
    with merge_store.StateStore(path, merge_store.scope_for(app.STASH_URL)) as store:
        assert store.cleanup_pending is None and store.cleanup_generation == 1


def args(tmp_path, labels=(), cores=(), by_model=None, mode='cleanup-only'):
    result = options(tmp_path, labels, cores, by_model)
    result[result.index('--cleanup-only')] = '--' + mode
    return [*result, '--cleanup-strategy', 'sql', '--cleanup-page-size', '3', '--cleanup-write-batch-size', '2']


@pytest.mark.parametrize('dry', [False, True])
def test_global_uses_primary_ids_only_and_preserves_manual_cores(tmp_path, monkeypatch, dry):
    stash = SQLStash(dry=dry)
    blocked = stash.tag('ai:blocked')
    stash.scene['tags'] = [blocked, stash.tag('human')]
    stash.marker(merge_labels.AI, 'blocked', '1', manual=True)
    stash.marker(merge_labels.AI, 'core', '2')
    stash.marker(merge_labels.AI, 'core', '3', manual=True)
    keep = stash.marker(merge_labels.AI, 'keep', '4')
    keep['tags'].append(blocked)
    stash.images = [{'tags': [blocked]}]
    before = deepcopy((stash.scenes, stash.tags, stash.images))
    install(monkeypatch, stash)
    command = args(tmp_path, ['blocked'], ['core'])
    assert app.main(command + (['--dry-run'] if dry else [])) == 0
    assert not stash.scene_reads
    assert not any(k in {'primary_scenes', 'extra_scenes', 'extras'} for k, _ in stash.sql_calls)
    assert not any(op.startswith('CleanupMarker') for op, _ in stash.calls)
    if dry:
        assert before == (stash.scenes, stash.tags, stash.images) and not stash.writes
    else:
        assert {m['id'] for m in stash.scene['scene_markers']} == {'3', '4'}
        assert 'ai:blocked' not in stash.tags and stash.images[0]['tags'] == []
        assert {t['name'] for t in keep['tags']} == {merge_labels.AI.owner_tag}
        assert app.main(command) == 0


def test_streaming_primary_deletion_never_offsets_or_keeps_all_ids(tmp_path, monkeypatch):
    stash = SQLStash()
    for i in range(1, 18):
        stash.marker(merge_labels.AI, 'blocked', str(i))
    install(monkeypatch, stash)
    assert app.main(args(tmp_path, ['blocked'])) == 0
    assert not stash.scene['scene_markers']
    primary = [values for key, values in stash.sql_calls if key == 'primary']
    assert [p[-3] for p in primary] == [0, 3, 6, 9, 12, 15]
    assert len([w for w in stash.writes if w[0] == 'delete']) == 17
    assert all(len(v['ids']) <= 2 for op, v in stash.calls if op == 'CleanupDeleteMarkers')


@pytest.mark.parametrize('mode', ['cleanup-markers-only', 'cleanup-scene-tags-only'])
@pytest.mark.parametrize('dry', [False, True])
def test_partial_modes_only_remove_their_associations(tmp_path, monkeypatch, mode, dry):
    stash = SQLStash(dry=dry)
    banned, human = stash.tag('ai:blocked'), stash.tag('human')
    stash.scene['tags'] = [banned, human]
    stash.marker(merge_labels.AI, 'blocked', '1', manual=True)
    keep = stash.marker(merge_labels.AI, 'keep', '2')
    keep['tags'] += [banned, human]
    keep.update(seconds=10.125, end_seconds=17.875)
    before = deepcopy((stash.scenes, stash.tags))
    install(monkeypatch, stash)
    command = args(tmp_path, ['blocked'], mode=mode)
    assert app.main(command + (['--dry-run'] if dry else [])) == 0
    assert not stash.scene_reads and 'ai:blocked' in stash.tags
    if dry:
        assert (stash.scenes, stash.tags) == before
    elif mode == 'cleanup-markers-only':
        assert [m['id'] for m in stash.scene['scene_markers']] == ['2']
        assert banned in stash.scene['tags'] and banned not in keep['tags'] and human in keep['tags']
        assert (keep['seconds'], keep['end_seconds']) == (10.125, 17.875)
    else:
        assert banned not in stash.scene['tags'] and banned in keep['tags']
        assert len(stash.scene['scene_markers']) == 2


@pytest.mark.parametrize('mode,changed', [('cleanup-markers-only', 'markers'), ('cleanup-scene-tags-only', 'scene')])
def test_partial_lazy_masks_do_not_change_the_other_publication(tmp_path, monkeypatch, mode, changed):
    stash = SQLStash()
    stash.scene['tags'] = [stash.tag('ai:blocked')]
    stash.marker(merge_labels.AI, 'blocked', '1')
    command = args(tmp_path, ['blocked'], mode=mode)
    with merge_store.StateStore(tmp_path/'state.sqlite3', merge_store.scope_for(app.STASH_URL)) as store:
        store.publish('1', {'scene': {'contributions': [hit('wd', 'blocked')]},
                            'markers': {'contributions': [span('wd', 0, 2, tag='blocked')]}})
    install(monkeypatch, stash)
    assert app.main(command) == 0
    with merge_store.StateStore(tmp_path/'state.sqlite3', merge_store.scope_for(app.STASH_URL)) as store:
        for kind in KINDS:
            assert bool(store.snapshot('1', kind)[0]['contributions']) == (kind != changed)


@pytest.mark.parametrize('failure,mode', [('CleanupDeleteMarkers','cleanup-only'), ('CleanupDeleteTags','cleanup-only'),
                                        ('CleanupRemoveTags','cleanup-scene-tags-only')])
def test_timeout_after_commit_resumes_without_replaying_deleted_ids(tmp_path, monkeypatch, failure, mode):
    stash = SQLStash()
    stash.marker(merge_labels.AI, 'blocked', '1')
    stash.scene['tags'] = [stash.tag('ai:blocked')]
    stash.fail_after = failure
    install(monkeypatch, stash)
    command = args(tmp_path, ['blocked'], mode=mode)
    assert app.main(command) == 1
    with sqlite3.connect(tmp_path/'state.sqlite3') as db:
        assert db.execute('SELECT pending FROM cleanup_state').fetchone()[0]
    assert app.main(command) == 0
    assert len([w for w in stash.writes if w[0] == 'delete']) <= 1


@pytest.mark.parametrize('new_name', ['ai:renamed', 'AI:blocked'])
def test_recheck_preserves_renamed_primary_tag(tmp_path, monkeypatch, new_name):
    from tests.helpers import cleanup_sql as sql_helpers
    monkeypatch.setattr(sql_helpers, 'SCHEMA', SCHEMA.replace('name TEXT UNIQUE', 'name TEXT COLLATE NOCASE UNIQUE'))
    stash = SQLStash()
    stash.marker(merge_labels.AI, 'blocked', '1')
    def rename(client, key):
        if key == 'primary_check':
            tag = client.tags.pop('ai:blocked')
            tag['name'] = new_name
            client.tags[tag['name']] = tag
            client.before_sql = None
    stash.before_sql = rename
    install(monkeypatch, stash)
    assert app.main(args(tmp_path, ['blocked'])) == 0
    assert stash.scene['scene_markers'] and new_name in stash.tags and not stash.writes


def test_source_specific_cleanup_reconciles_only_candidates(tmp_path, monkeypatch):
    stash = SQLStash()
    for i in range(2, 20):
        stash.scenes[str(i)] = dict(id=str(i), tags=[], scene_markers=[])
    with merge_store.StateStore(tmp_path/'state.sqlite3', merge_store.scope_for(app.STASH_URL)) as store:
        merge_sync.sync_merged(app, stash, app.services.TagResolver(stash, source=merge_labels.AI), store, stash.find_scene('1'),
                         [hit('wd'), hit('oppai')], [span('joytag',10,20),span('oppai',20,30),span('wd',30,40)], tag_policies=empty_tag_policies())
    stash.scene_reads.clear()
    install(monkeypatch, stash)
    assert app.main(args(tmp_path, by_model={'oppai':['common']})) == 0
    assert set(stash.scene_reads) == {'1'}
    assert sorted((m['seconds'],m['end_seconds']) for m in stash.scene['scene_markers']) == [(10,20),(30,40)]


def test_sql_dry_run_allowlist_cannot_execute_writes_or_injected_statements():
    assert is_read_only_cleanup(READ_OPERATION, {'sql': SQL['probe']})
    for sql in ['DELETE FROM tags', 'SELECT 1; DELETE FROM tags', SQL['probe']+'; DELETE FROM tags']:
        assert not is_read_only_cleanup(READ_OPERATION, {'sql': sql})
    stash = StashClient.__new__(StashClient)
    stash.dry_run = True
    stash.report = None
    with pytest.raises(RuntimeError, match='dry-run'):
        stash.gql(READ_OPERATION, {'sql': 'DELETE FROM tags', 'args': []})
    with pytest.raises(RuntimeError, match='dry-run'):
        stash.gql(READ_OPERATION + ' mutation { execSQL(sql:"DELETE FROM tags") { rows_affected } }', {'sql':SQL['probe']})


def test_keyset_relation_pages_work_when_prior_rows_are_deleted():
    # Unlike the small mutable API fixture, keep this real database across calls
    # so relation rowids remain stable, exactly as in Stash.
    db = sqlite3.connect(':memory:')
    db.executescript(SCHEMA)
    db.execute("INSERT INTO tags VALUES (1,'ai:blocked')")
    db.executemany('INSERT INTO scene_markers_tags VALUES (?,1)', [(i,) for i in range(20,0,-1)])
    class Reader:
        def gql(self, query, variables):
            return {'querySQL': {'rows': [list(row) for row in db.execute(variables['sql'], variables['args'])]}}
    session = CleanupSqlSession(Reader(), Counter(), page_size=3)
    seen = []
    for rows in session.pages('extras', (1,'ai:blocked'), 20):
        seen.extend(r[1] for r in rows)
        db.executemany('DELETE FROM scene_markers_tags WHERE rowid=?', [(r[0],) for r in rows])
    assert seen == list(range(20,0,-1))
    plan = [r[3] for r in db.execute('EXPLAIN QUERY PLAN '+SQL['extras'],(1,'ai:blocked',0,20,3))]
    assert any('index_scene_markers_tags_on_tag_id' in row for row in plan)
    assert not any('TEMP B-TREE' in row for row in plan)


@pytest.mark.parametrize('flag,value',[('--cleanup-page-size','0'),('--cleanup-write-batch-size','-1')])
def test_invalid_limits_fail_before_io(flag,value):
    with pytest.raises(SystemExit):
        app.main(['--cleanup-only',flag,value])


def test_sql_cleanup_uses_bulk_mutations_without_capability_queries(tmp_path, monkeypatch):
    stash = SQLStash()
    stash.marker(merge_labels.AI, 'blocked', '1')
    install(monkeypatch, stash)
    assert app.main(args(tmp_path, ['blocked'])) == 0
    assert not stash.scene['scene_markers'] and 'ai:blocked' not in stash.tags
    assert {'CleanupDeleteMarkers', 'CleanupDeleteTags'} <= {op for op, _ in stash.calls}
    assert all(op != 'CleanupCapabilities' for op, _ in stash.calls)


@pytest.mark.parametrize('strategy', ['auto', 'sql'])
def test_unavailable_sql_does_not_return_to_global_marker_filter(tmp_path, monkeypatch, strategy):
    stash = FastStash()
    stash.marker(merge_labels.AI, 'blocked', '1', manual=True)
    install(monkeypatch, stash)
    command = [*args(tmp_path, ['blocked']), '--cleanup-strategy', strategy]
    assert app.main(command) == (1 if strategy == 'sql' else 0)
    assert not any(op == 'CleanupMarkers' for op, _ in stash.calls)
    assert not stash.writes if strategy == 'sql' else not stash.scene['scene_markers']


def test_cleanup_preview_reads_only_requested_state_and_shadows_in_memory(tmp_path):
    path = tmp_path/'state.sqlite3'
    snapshot = {'contributions': [hit('wd')]}
    segment = inference_record(**dict(mode='frames', items=[dict(start=i, end=i+1, tags=[('common', .9)]) for i in range(2)]))
    with merge_store.StateStore(path,'fixture') as state:
        state.put('1','wd',{'markers':segment})
        state.publish('1',{'scene':snapshot})
    before = path.read_bytes(), path.stat().st_mtime_ns
    with merge_store.StateStore(path,'fixture',dry_run=True,cleanup_preview=True) as state:
        assert state.preview_source
        assert state.connection.execute('SELECT count(*) FROM main.results').fetchone() == (0,)
        assert state.connection.execute('SELECT count(*) FROM main.snapshots').fetchone() == (0,)
        assert state.snapshot('1','scene')[0] == snapshot
        assert state.records('1','markers')['wd'] == segment
        state.publish('1',{'scene':{'contributions':[]}})
        assert state.snapshot('1','scene')[0]['contributions'] == []
        with pytest.raises(sqlite3.OperationalError, match='readonly'):
            state.connection.execute('DELETE FROM cleanup_source.results')
    assert (path.read_bytes(),path.stat().st_mtime_ns) == before


def test_cleanup_preview_detects_pending_publications_in_readonly_source(tmp_path):
    path = tmp_path/'state.sqlite3'
    with merge_store.StateStore(path,'fixture') as state:
        state.prepare('1',{'markers':{'contributions':[]}})
    with merge_store.StateStore(path,'fixture',dry_run=True,cleanup_preview=True) as state:
        assert state.has_pending_publications()
        assert state.snapshot('1','markers')[1] is not None


def test_sql_pruning_limits_counters_to_possible_empty_tags(tmp_path, monkeypatch):
    stash = SQLStash()
    unused, image_tag = stash.tag('ai:unused'), stash.tag('ai:image_only')
    used = stash.tag('ai:used')
    stash.scene['tags'] = [used]
    stash.images = [{'tags':[image_tag]}]
    install(monkeypatch, stash)
    assert app.main([*args(tmp_path), '--prune-empty-tags']) == 0
    assert 'ai:unused' not in stash.tags
    assert 'ai:image_only' in stash.tags and 'ai:used' in stash.tags
    assert any(key == 'unused_candidates' for key, _ in stash.sql_calls)


@pytest.mark.parametrize('strategy', ['auto', 'sql'])
@pytest.mark.parametrize('dry', [False, True])
def test_legacy_interrupted_global_cleanup_can_resume_using_sql(tmp_path, monkeypatch, strategy, dry):
    from stash_ai.signatures import digest
    from stash_ai.models import SOURCES

    stash = SQLStash(dry=dry)
    stash.marker(merge_labels.AI, 'blocked', '2')  # Marker 1 was already deleted by legacy.
    tag_id = stash.tags['ai:blocked']['id']
    command = [*args(tmp_path, ['blocked']), '--cleanup-strategy', strategy]
    path = tmp_path/'state.sqlite3'
    with merge_store.StateStore(path, merge_store.scope_for(app.STASH_URL)) as store:
        store.publish('1', {'markers': {'contributions': [span('wd', 0, 2, tag='blocked')]}})
        # Persist the old full-cleanup format directly: no output_kinds in the
        # fingerprint and no marker_/scene_ prefixes in the lazy masks.
        fingerprint = digest({'tags': {s: ['blocked'] for s in SOURCES},
                              'cores': {s: [] for s in SOURCES}, 'marker_blocklist': []})
        masks = {'global': {'blocked': 1}, 'core': {}, 'ids': {tag_id: 1}}
        with store.connection:
            store.connection.execute('INSERT INTO cleanup_state VALUES (?,?,?,?)',
                                     (store.scope, 1, state_codec.encode(masks), fingerprint))
            store.connection.execute('PRAGMA user_version=18')
    from stash_ai.db.update import update_database
    update_database(path)
    from stash_ai.tools.cleanup_pending_migration import migrate
    from stash_ai.cleanup.fingerprint import fingerprint as current_fingerprint
    tags, cores = {s: ['blocked'] for s in SOURCES}, {s: [] for s in SOURCES}
    with merge_store.StateStore(path, merge_store.scope_for(app.STASH_URL), allow_cleanup=True) as store:
        assert migrate(store, tags, cores, ['scene', 'markers'])
    fingerprint = current_fingerprint(tags, cores, ['scene', 'markers'])
    before = deepcopy((stash.scenes, stash.tags)), path.read_bytes(), path.stat().st_mtime_ns
    install(monkeypatch, stash)
    assert app.main(command + (['--dry-run'] if dry else [])) == 0
    if dry:
        assert ((stash.scenes, stash.tags), path.read_bytes(), path.stat().st_mtime_ns) == before
        assert not stash.writes
    else:
        assert 'ai:blocked' not in stash.tags and not stash.scene['scene_markers']
        assert [identifier for op, identifier in stash.writes if op == 'delete'] == ['2']
    with merge_store.StateStore(path, merge_store.scope_for(app.STASH_URL), allow_cleanup=dry) as store:
        assert store.cleanup_generation == 1
        assert store.cleanup_pending == (fingerprint if dry else None)
        assert store.snapshot('1', 'markers')[0]['contributions'] == []


def test_removed_legacy_strategy_fails_before_api_or_state_access(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(app.services, 'create_stash_client', lambda *_: pytest.fail('invalid strategy accessed Stash'))
    command = [*args(tmp_path), '--cleanup-strategy', 'legacy']
    with pytest.raises(SystemExit) as exc:
        app.main(command)
    assert exc.value.code == 2
    assert "invalid choice: 'legacy'" in capsys.readouterr().err
    assert not (tmp_path/'state.sqlite3').exists()


def test_switching_output_scope_during_recovery_is_rejected(tmp_path, monkeypatch):
    stash = SQLStash()
    stash.scene['tags'] = [stash.tag('ai:blocked')]
    stash.fail_after = 'CleanupRemoveTags'
    install(monkeypatch, stash)
    command = args(tmp_path,['blocked'],mode='cleanup-scene-tags-only')
    assert app.main(command) == 1
    stash.writes.clear()
    assert app.main(args(tmp_path,['blocked'])) == 1
    assert not stash.writes
    assert app.main(command) == 0


def test_marker_preview_does_not_double_count_extras_on_deleted_cores(tmp_path, monkeypatch, capsys):
    stash = SQLStash(dry=True)
    blocked = stash.tag('ai:blocked')
    for source, label, i in [(merge_labels.AI,'blocked','1'), (merge_labels.AI,'keep','2'), (merge_labels.AI,'core','3')]:
        stash.marker(source,label,i)['tags'].append(blocked)
    install(monkeypatch, stash)
    assert app.main([*args(tmp_path,['blocked'],['core'],mode='cleanup-markers-only'),'--dry-run']) == 0
    out = capsys.readouterr().out
    assert 'markers_deleted=2' in out and 'marker_tag_removals=1' in out
    assert not stash.writes
