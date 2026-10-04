"""Cleanup candidate snapshots use the production reader against real SQLite."""
from copy import deepcopy
import json

import pytest

from tests.helpers.runtime import runtime as app
import stash_ai.evidence.labels as merge_labels
import stash_ai.storage.store as merge_store
from stash_ai.publication import merged as merge_sync
from stash_ai.stash_api.client import StashClient
from stash_ai.cleanup.sql import SQLPreview
from stash_ai.stash_api.sql import OPERATION
from stash_ai.runtime.report import RunReport
from stash_ai.models import WD
from tests.helpers.cleanup import FastStash, install, options
from tests.helpers.cleanup_sql import SQLStash
from tests.helpers.evidence import hit, span


pytestmark = pytest.mark.usefixtures('reset_progress')


class SceneSQLStash(SQLStash):
    """Real client read paths with mutable offline Stash API writes."""
    iter_scenes = StashClient.iter_scenes
    find_scene = StashClient.find_scene
    find_scene_for_merge = StashClient.find_scene_for_merge
    merge_snapshots = StashClient.merge_snapshots
    scene_fields = StashClient.scene_fields

    def __init__(self, dry=False):
        super().__init__(dry)
        self.scene_page_size = 100
        self.merge_reader, self.merge_only_reads = None, False
        self.create_markers, self.api_key = True, ''
        self.snapshot_reads = []
        self.deny_snapshots = False
        self.after_snapshot = None

    def configure_merge_reads(self, report, *, merge_only=False):
        self.create_markers = app.CREATE_MARKERS
        StashClient.configure_merge_reads(self, report, merge_only=merge_only)

    def database(self):
        db = super().database()
        db.executescript('''
            CREATE TABLE scenes(id INTEGER PRIMARY KEY, title TEXT);
            CREATE INDEX st_scene ON scenes_tags(scene_id);
            ALTER TABLE scene_markers ADD COLUMN seconds REAL;
            ALTER TABLE scene_markers ADD COLUMN end_seconds REAL;
        ''')
        db.executemany('INSERT INTO scenes VALUES (?,?)',
                       [(int(s['id']), s.get('title', '')) for s in self.scenes.values()])
        for scene in self.scenes.values():
            for marker in scene['scene_markers']:
                db.execute('UPDATE scene_markers SET seconds=?,end_seconds=? WHERE id=?',
                           (marker['seconds'], marker.get('end_seconds'), int(marker['id'])))
        return db

    def gql(self, query, variables=None):
        if query == OPERATION:
            ids = json.loads(variables['args'][0])
            self.snapshot_reads.append(ids)
            if self.deny_snapshots:
                raise RuntimeError('snapshot schema unavailable')
            with self.database() as db:
                cursor = db.execute(variables['sql'], variables['args'])
                result = {'querySQL': {'rows': [list(row) for row in cursor]}}
            if ids and self.after_snapshot:
                self.after_snapshot(self)
            return result
        if 'StashAIScene(' in query:
            return {'findScene': FastStash.find_scene(self, variables['id'])}
        if 'StashAIScenes(' in query:
            f = variables['filter']
            ids = sorted(self.scenes, key=int)
            start = (f['page'] - 1) * f['per_page']
            return {'findScenes': {'scenes': [FastStash.find_scene(self, i)
                for i in ids[start:start + f['per_page']]]}}
        return super().gql(query, variables)


@pytest.mark.parametrize('dry', [False, True])
@pytest.mark.parametrize('mode', ['--cleanup-only', '--cleanup-scene-tags-only'])
def test_noop_candidates_use_three_batches_for_205_scenes(tmp_path, monkeypatch, dry, mode):
    stash = SceneSQLStash(dry)
    tag = stash.tag('ai:common')
    stash.scenes = {str(i): dict(id=str(i), title=str(i), tags=[tag], scene_markers=[])
                    for i in range(1, 206)}
    command = options(tmp_path, by_model={'wd': ['common']})
    command[command.index('--cleanup-only')] = mode
    with merge_store.StateStore(tmp_path/'state.sqlite3', merge_store.scope_for(app.STASH_URL)) as store:
        for i in stash.scenes:
            store.publish(i, {'scene': {'contributions': [hit('oppai')], 'owned_ids': [tag['id']]}})
    before = deepcopy((stash.scenes, stash.tags)), (tmp_path/'state.sqlite3').read_bytes()
    install(monkeypatch, stash)
    assert app.main(command + (['--dry-run'] if dry else [])) == 0
    assert [len(ids) for ids in stash.snapshot_reads] == [0, 100, 100, 5]
    assert not stash.scene_reads and not stash.writes
    assert (stash.scenes, stash.tags) == before[0]
    if dry:
        assert (tmp_path/'state.sqlite3').read_bytes() == before[1]


@pytest.mark.parametrize('strategy', ['auto', 'sql', 'scenes'])
def test_snapshot_failure_falls_back_and_scene_strategy_stays_graphql(tmp_path, monkeypatch, strategy):
    stash = SceneSQLStash()
    stash.deny_snapshots = True
    stash.scene['tags'] = [stash.tag('wd:blocked'), stash.tag('human')]
    install(monkeypatch, stash)
    command = options(tmp_path, by_model={'wd': ['blocked']}, output='separate')
    assert app.main([*command, '--cleanup-strategy', strategy]) == 0
    assert [t['name'] for t in stash.scene['tags']] == ['human']
    assert stash.scene_reads
    assert stash.snapshot_reads == ([] if strategy == 'scenes' else [[]])


@pytest.mark.parametrize('output', ['merged', 'separate'])
def test_prefetched_scene_is_refreshed_before_cleanup_writes(tmp_path, monkeypatch, output):
    stash = SceneSQLStash()
    if output == 'merged':
        with merge_store.StateStore(tmp_path/'state.sqlite3', merge_store.scope_for(app.STASH_URL)) as store:
            merge_sync.sync_merged(app, stash, app.services.TagResolver(stash, source=merge_labels.AI), store,
                              stash.find_scene('1'), [hit('wd')], [])
    else:
        stash.scene['tags'] = [stash.tag('wd:common')]
        stash.marker(WD, 'keep', '1')['tags'].append(stash.tag('wd:common'))
        stash.marker(WD, 'common', '2')
    human = stash.tag('human')
    def concurrent_edit(client):
        client.after_snapshot = None
        client.scene['tags'].append(human)
        if output == 'separate':
            client.scene['scene_markers'][0]['tags'].append(human)
            client.scene['scene_markers'][1]['primary_tag'] = client.tag('wd:renamed')
    stash.after_snapshot = concurrent_edit
    stash.scene_reads.clear()
    install(monkeypatch, stash)
    assert app.main(options(tmp_path, by_model={'wd': ['common']}, output=output)) == 0
    assert [t['name'] for t in stash.scene['tags']] == ['human']
    assert not stash.scene_reads
    assert len(stash.snapshot_reads) >= 3  # Probe, batch, fresh pre-write snapshot.
    if output == 'separate':
        assert len(stash.scene['scene_markers']) == 2
        assert {t['name'] for t in stash.scene['scene_markers'][0]['tags']} == {WD.owner_tag, 'human'}
        assert stash.scene['scene_markers'][1]['primary_tag']['name'] == 'wd:renamed'


def test_preview_masks_batch_and_fresh_reads_without_mutating_snapshots():
    stash = SceneSQLStash(dry=True)
    banned = stash.tag('ai:blocked')
    stash.scene['tags'] = [banned, stash.tag('human')]
    stash.marker(merge_labels.AI, 'blocked', '1')
    stash.marker(WD, 'core', '2')
    stash.marker(WD, 'core', '3', manual=True)
    stash.marker(merge_labels.AI, 'keep', '4')['tags'].append(banned)
    before = deepcopy(stash.scene)
    stash.configure_merge_reads(RunReport(write=lambda _: None), merge_only=True)
    view = SQLPreview(stash, app, {'ai:blocked'}, {'wd:core': WD})
    reads = [list(view.iter_scenes(['1']))[0], view.find_scene('1'), view.find_scene_for_merge('1')]
    for scene in reads:
        assert [t['name'] for t in scene['tags']] == ['human']
        assert [m['id'] for m in scene['scene_markers']] == ['3', '4']
        assert all(t['name'] != 'ai:blocked' for m in scene['scene_markers'] for t in m['tags'])
    assert stash.scene == before and not stash.scene_reads and not stash.writes
    count = len(stash.snapshot_reads)
    assert list(view.iter_scenes([])) == []
    assert len(stash.snapshot_reads) == count


@pytest.mark.parametrize('dry', [False, True])
def test_global_cleanup_without_candidates_never_probes_snapshots(tmp_path, monkeypatch, dry):
    stash = SceneSQLStash(dry)
    stash.scene['tags'] = [stash.tag('ai:blocked')]
    install(monkeypatch, stash)
    assert app.main(options(tmp_path, labels=['blocked']) + (['--dry-run'] if dry else [])) == 0
    assert not stash.snapshot_reads and not stash.scene_reads


@pytest.mark.parametrize('dry', [False, True])
def test_batched_marker_cleanup_matches_graphql_with_global_preview(tmp_path, monkeypatch, dry):
    results = []
    for deny in (False, True):
        path = tmp_path / str(deny)
        path.mkdir()
        stash = SceneSQLStash()
        with merge_store.StateStore(path/'state.sqlite3', merge_store.scope_for(app.STASH_URL)) as store:
            merge_sync.sync_merged(app, stash, app.services.TagResolver(stash, source=merge_labels.AI), store,
                stash.find_scene('1'), [hit('wd'), hit('oppai'), hit('wd', 'blocked')],
                [span('wd', 10, 20), span('oppai', 20, 30), span('wd', 40, 50, tag='blocked')])
        before = deepcopy((stash.scenes, stash.tags)), (path/'state.sqlite3').read_bytes()
        stash.dry_run, stash.deny_snapshots = dry, deny
        stash.writes.clear()
        stash.scene_reads.clear()
        install(monkeypatch, stash)
        finished = []
        # Capture cleanup counters without changing argument parsing.
        monkeypatch.setattr(RunReport, 'finish',
                            lambda self, title, counts, cache=None: finished.append(dict(counts)))
        command = options(path, labels=['blocked'], by_model={'oppai': ['common']})
        assert app.main(command + (['--dry-run'] if dry else [])) == 0
        assert finished[0]['markers_deleted'] == 1
        assert finished[0]['markers_updated'] == 1
        results.append((deepcopy((stash.scenes, stash.tags)),
                        {k: v for k, v in finished[0].items() if k in
                         ('scene_tags_updated', 'markers_created', 'markers_updated', 'markers_deleted')}))
        if dry:
            assert (stash.scenes, stash.tags) == before[0] and not stash.writes
            assert (path/'state.sqlite3').read_bytes() == before[1]
        else:
            assert [(m['seconds'], m['end_seconds']) for m in stash.scene['scene_markers']] == [(10, 20)]
        assert bool(stash.scene_reads) == deny
    assert results[0] == results[1]

