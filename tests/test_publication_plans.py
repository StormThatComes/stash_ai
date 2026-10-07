"""Concrete dry-run plans and confirmed batch deletion with durable recovery."""
from copy import deepcopy
import json

import pytest

from stash_ai.commands.arguments import parse_args
from stash_ai.evidence.labels import AI
from stash_ai.stash_api.client import StashClient
from tests.helpers.runtime import runtime as app

pytestmark = pytest.mark.usefixtures('reset_merge')


def test_plan_matches_live_mutations_and_keeps_source_database(pipeline, tmp_path):
    p = pipeline
    assert app.main(p.args) == 0
    p.stash.scene['scene_markers'].pop()
    p.stash.scene['scene_markers'][0]['title'] += ' old'
    p.stash.marker(AI, 'obsolete', 'old-id')
    p.stash.scene['tags'] = []
    before, loaded, remote = p.path.read_bytes(), len(p.loaded), deepcopy(p.stash.scene)
    destination = tmp_path / 'plan.jsonl'
    p.stash.dry_run = True
    assert app.main([*p.args, '--merge-only', '--dry-run', '--force-merge',
                     '--dry-run-plan', str(destination)]) == 0
    assert p.path.read_bytes() == before and p.stash.scene == remote and len(p.loaded) == loaded
    rows = [json.loads(line) for line in destination.read_text().splitlines()]
    assert rows[0]['type'] == 'header' and rows[-1] == {'type': 'summary', 'exit_code': 0}
    scene = next(row for row in rows if row['type'] == 'scene')
    assert scene['deletes'] == [{'id': 'old-id', 'reason': 'obsolete_owned_marker_or_excluded_primary'}]
    p.stash.dry_run = False
    app.DRY_RUN = False
    p.stash.writes.clear()
    assert app.main([*p.args, '--merge-only', '--force-merge']) == 0
    assert [payload for operation, payload in p.stash.writes if operation == 'create'] == [
        item['payload'] for item in scene['creates']]
    assert [payload for operation, payload in p.stash.writes if operation == 'update'] == [
        item['payload'] for item in scene['updates']]
    assert [payload for operation, payload in p.stash.writes if operation == 'scene'] == [scene['scene_tags']['after']]
    assert len(p.loaded) == loaded


@pytest.mark.parametrize('flags', [[], ['--dry-run'], ['--merge-only']])
def test_plan_requires_saved_evidence_preview(flags, tmp_path):
    with pytest.raises(SystemExit):
        parse_args(app, [*flags, '--dry-run-plan', str(tmp_path / 'plan.jsonl')])


def test_missing_tags_are_explicit_symbolic_mutations(pipeline, tmp_path):
    p = pipeline
    assert app.main(p.args) == 0
    p.stash.tags.clear()
    p.stash.scene['tags'] = []
    p.stash.scene['scene_markers'] = []
    p.stash.dry_run = True
    destination = tmp_path / 'missing.jsonl'
    before, loaded = p.path.read_bytes(), len(p.loaded)
    assert app.main([*p.args, '--merge-only', '--dry-run', '--dry-run-plan', str(destination)]) == 0
    rows = [json.loads(line) for line in destination.read_text().splitlines()]
    tags = [row for row in rows if row['type'] == 'tag']
    assert {'ai:common', 'ai_markers', 'm:common'} <= {row['payload']['name'] for row in tags}
    assert all(row['operation'] == 'create' and row['id'].startswith('dry-run:') for row in tags)
    assert p.path.read_bytes() == before and not p.stash.tags and len(p.loaded) == loaded


def test_native_batch_deletes_are_bounded_and_strictly_confirmed():
    client = object.__new__(StashClient)
    batches = []
    def gql(query, variables):
        batches.append(variables['ids'])
        assert 'sceneMarkersDestroy' in query
        return {'sceneMarkersDestroy': True}
    client.gql = gql
    client.delete_markers([str(i) for i in range(451)] + ['1'])
    assert list(map(len, batches)) == [200, 200, 51]
    client.gql = lambda *args: {'sceneMarkersDestroy': 'true'}
    with pytest.raises(RuntimeError, match='not confirmed'):
        client.delete_markers(['1'])


def test_lost_batch_response_resumes_without_inference(pipeline, monkeypatch):
    p = pipeline
    assert app.main(p.args) == 0
    p.stash.marker(AI, 'obsolete', 'obsolete-id')
    loaded = len(p.loaded)
    original = p.stash.delete_markers
    def lost_response(ids):
        original(ids)
        raise OSError('response lost after commit')
    monkeypatch.setattr(p.stash, 'delete_markers', lost_response)
    assert app.main([*p.args, '--merge-only']) == 1
    from stash_ai.storage.store import StateStore, scope_for
    with StateStore(p.path, scope_for(app.STASH_URL)) as store:
        assert store.has_pending_publications()
    monkeypatch.setattr(p.stash, 'delete_markers', original)
    assert app.main([*p.args, '--merge-only']) == 0
    assert len(p.loaded) == loaded and all(m['id'] != 'obsolete-id' for m in p.stash.scene['scene_markers'])
