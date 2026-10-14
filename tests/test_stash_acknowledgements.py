"""A normally returning HTTP call is not sufficient mutation confirmation."""
from types import SimpleNamespace
import pytest

from stash_ai.stash_api.client import StashClient
from stash_ai.runtime.mutations import write
from stash_ai.runtime.report import RunReport


@pytest.mark.parametrize('value', [None, {}, [], True, {'id': None}, {'id': 7}, {'id': ''},
    {'id': '0'}, {'id': '01'}, {'id': 'other'}, {'id': '999'}])
@pytest.mark.parametrize('kind', ['scene', 'marker', 'tag_parents', 'tag_organization'])
def test_wrong_acknowledgement_is_unknown_and_never_confirmed(value, kind):
    client = object.__new__(StashClient)
    report = client.report = RunReport(write=lambda _: None)
    fields = {'scene': 'sceneUpdate', 'marker': 'sceneMarkerUpdate',
              'tag_parents': 'tagUpdate', 'tag_organization': 'tagUpdate'}
    calls = []
    def gql(*args): calls.append(args); return {fields[kind]: value}
    client.gql = gql
    actions = {'scene': lambda: write(report, 'scene', lambda: client.update_scene_tags('7', [])),
               'marker': lambda: write(report, 'marker', lambda: client.update_marker({'id': '7'})),
               'tag_parents': lambda: client.update_tag_parents('7', []),
               'tag_organization': lambda: client.update_tag_organization('7', [], True)}
    with pytest.raises(RuntimeError, match='not confirmed'):
        actions[kind]()
    assert len(calls) == 1 and report.counts['mutation_unknown'] == 1
    assert report.counts['mutation_confirmed'] == 0


@pytest.mark.parametrize('value', [None, [], {}, {'id': None}, {'id': True}, {'id': ''}, {'id': '0'}])
@pytest.mark.parametrize('kind', ['tag', 'marker'])
def test_creations_reject_missing_or_wrong_scalar_ids(value, kind):
    client = object.__new__(StashClient); client.report = None
    client.gql = lambda *a: {'tagCreate' if kind == 'tag' else 'sceneMarkerCreate': value}
    with pytest.raises(RuntimeError, match='not confirmed'):
        client.create_tag('x') if kind == 'tag' else client.create_marker({})


@pytest.mark.parametrize('value', [None, False, 1, 'true', 'false', {}, []])
def test_single_delete_requires_literal_true(value):
    client = object.__new__(StashClient); client.gql = lambda *a: {'sceneMarkerDestroy': value}
    with pytest.raises(RuntimeError, match='not confirmed'): client.delete_marker('7')


def test_valid_acknowledgements_preserve_return_values():
    client = object.__new__(StashClient); client.report = None
    client.gql = lambda *a: {'sceneMarkerCreate': {'id': '7'}}
    assert client.create_marker({}) == '7'
    client.gql = lambda *a: {'sceneUpdate': {'id': '7'}}
    assert client.update_scene_tags('7', []) is None
    client.gql = lambda *a: {'tagUpdate': {'id': '7', 'name': 'x'}}
    assert client.update_tag_parents('7', []) == {'id': '7', 'name': 'x'}


@pytest.mark.parametrize('value', [None, [], {}])
def test_missing_field_is_unknown(value):
    client = object.__new__(StashClient)
    report = RunReport(write=lambda _: None)
    client.gql = lambda *a: {} if value is None else {'sceneUpdate': value}
    with pytest.raises(RuntimeError): write(report, 'scene', lambda: client.update_scene_tags('7', []))
    assert report.counts['mutation_unknown'] == 1


@pytest.mark.usefixtures('reset_merge')
def test_lost_shape_after_remote_apply_remains_pending_until_fresh_observation(pipeline, monkeypatch):
    from tests.helpers.runtime import runtime as app
    from stash_ai.storage.store import StateStore, scope_for
    assert app.main(pipeline.args) == 0
    original = pipeline.stash.update_scene_tags
    calls = []
    def invalid(scene, tags):
        calls.append(scene); original(scene, tags)
        from stash_ai.stash_api.acknowledgements import entity
        entity({'id': '999'}, scene)
    pipeline.stash.update_scene_tags = invalid
    pipeline.stash.scene['tags'] = []
    loaded = len(pipeline.loaded)
    assert app.main([*pipeline.args, '--merge-only', '--force-merge']) == 1
    with StateStore(pipeline.path, scope_for(app.STASH_URL)) as state:
        assert state.has_pending_publications()
    pipeline.stash.update_scene_tags = original
    assert app.main([*pipeline.args, '--merge-only']) == 0
    assert calls == ['1']
    assert len(pipeline.loaded) == loaded
