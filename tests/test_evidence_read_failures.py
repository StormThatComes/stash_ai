"""Offline consumers reject source changes and classify corrupt state payloads."""
import sqlite3
from types import SimpleNamespace

import pytest

from stash_ai.storage.store import StateStore, scope_for
from stash_ai.storage.codec import decode
from tests.helpers.runtime import runtime as app
from tests.helpers.evidence import empty_policies, inference_record

pytestmark = pytest.mark.usefixtures('reset_merge')


def test_explain_rejects_commit_after_final_snapshot_read(tmp_path, monkeypatch, capsys):
    from stash_ai.commands import explain
    path = tmp_path/'state.db'
    with StateStore(path, scope_for(app.STASH_URL)) as store:
        store.connection.execute('PRAGMA journal_mode=WAL')
    original = StateStore.snapshot
    def snapshot(self, scene, kind, **kwargs):
        result = original(self, scene, kind, **kwargs)
        if kind == 'markers':
            with sqlite3.connect(path) as writer:
                writer.execute('INSERT INTO refresh_requests VALUES (?,?,?,?)',
                               (self.scope, str(scene), 'wd', 'scene'))
        return result
    monkeypatch.setattr(StateStore, 'snapshot', snapshot)
    monkeypatch.setattr(explain.merge, 'evidence', lambda *args: ([], []))
    args = SimpleNamespace(state_db=path, ram_cache_gb=0, ram_budget_gb=0,
                           scene_id=['1'], explain_tag='ai:tag')
    with pytest.raises(RuntimeError, match='source state DB changed'):
        explain.explain(app, args, empty_policies(), empty_policies())
    assert capsys.readouterr().out == ''


@pytest.mark.parametrize('payload', [b'not-zlib', b'x\x01'])
def test_corrupt_compression_has_an_offline_storage_error(payload):
    with pytest.raises(ValueError, match='Invalid compressed state payload') as caught:
        decode(payload)
    assert caught.value.__cause__ is not None


def test_review_skips_corrupt_scene_and_preserves_valid_scene(tmp_path):
    from stash_ai.reports.review.report import build_report
    with StateStore(tmp_path/'state.db', 's') as store:
        for scene in ('1', '2'):
            store.put(scene, 'wd', {'scene': inference_record(mode='scene', items=[])})
        with store.connection:
            store.connection.execute("UPDATE results SET data=? WHERE scene='1'", (b'not-zlib',))
        store.cache.clear()
        report = build_report(app, store, ['1', '2'], empty_policies(), empty_policies())
        assert report['skipped_by_reason'] == {'unusable_evidence': 1}
        assert report['totals']['analyzed_scenes'] == 1
        assert report['skipped_scene_examples'][0]['scene_id'] == '1'
        assert 'compressed state' in report['skipped_scene_examples'][0]['message']


def test_rebase_message_offers_merge_before_targeted_inference():
    from stash_ai.storage.snapshots import require_rebased
    with pytest.raises(RuntimeError) as caught:
        require_rebased([{'requires_rebase': True}])
    message = str(caught.value)
    assert '--merge-only' in message and 'compatible retained evidence' in message
    assert 'only the affected scenes/models' in message
    assert 'run fresh inference' not in message


def test_retired_publication_rebuilds_from_current_retained_cache(tmp_path):
    from stash_ai.evidence import pipeline
    from stash_ai.evidence.labels import AI
    from stash_ai.publication.merged import sync_merged
    from stash_ai.storage.snapshots import require_rebased
    from tests.helpers.evidence import paired_records, span
    from tests.helpers.stash import Stash
    stash = Stash()
    scenes, markers = paired_records({'wd': [.95] * 6})
    with StateStore(tmp_path/'state.db', 's') as store:
        store.put('1', 'wd', {'scene': scenes['wd'], 'markers': markers['wd']})
        retired = span('wd', 0, 6, tag='core')
        retired['requires_rebase'] = True
        store.publish('1', {'markers': {'contributions': [retired]}})
        hits, spans = pipeline.evidence(app, store, '1', empty_policies(), empty_policies())
        sync_merged(app, stash, app.services.TagResolver(stash, source=AI), store,
                    stash.find_scene('1'), hits, spans, tag_policies=empty_policies())
        require_rebased(store.snapshot('1', 'markers')[0]['contributions'])
