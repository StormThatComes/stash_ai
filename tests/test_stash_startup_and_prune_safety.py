"""Partial client acquisition and mismatched pruning reads never authorize writes."""
from copy import deepcopy

import pytest
import requests

from stash_ai.cleanup.prune import prune_tags
from stash_ai.recognition.stash import RecognitionStash
from stash_ai.stash_api.client import StashClient
from stash_ai.stash_api.transport import StashTransport
from tests.helpers.prune import PruneStash


def sessions(monkeypatch, *, stage=None, error=RuntimeError, after_mount=False, close_failure=False):
    import stash_ai.stash_api.read_connections as connections
    created, adapters, attempts = [], [], {'session': 0, 'adapter': 0, 'mount': 0}
    def fail(boundary):
        if stage == boundary:
            raise error('primary startup failure')
    class Headers(dict):
        def __init__(self, number): self.number = number
        def __setitem__(self, name, value):
            fail('headers' + str(self.number))
            return super().__setitem__(name, value)
    class Adapter:
        def __init__(self, **kwargs):
            attempts['adapter'] += 1
            fail('adapter' + str(attempts['adapter']))
            self.closes = 0
            adapters.append(self)
        def close(self): self.closes += 1
    class Session:
        def __init__(self):
            attempts['session'] += 1
            fail('session' + str(attempts['session']))
            self.headers, self.adapters = Headers(attempts['session']), {}
            self.closes = 0
            created.append(self)
        def mount(self, prefix, adapter):
            attempts['mount'] += 1
            boundary = 'mount' + str(attempts['mount'])
            if not after_mount: fail(boundary)
            self.adapters[prefix] = adapter
            if after_mount: fail(boundary)
        def close(self):
            self.closes += 1
            for adapter in self.adapters.values(): adapter.close()
            if close_failure and self is created[0]:
                raise OSError('secondary close failure')
    monkeypatch.setattr(requests, 'Session', Session)
    monkeypatch.setattr(connections, 'ReadAdapter', Adapter)
    return created, adapters


@pytest.mark.parametrize('error', [RuntimeError, KeyboardInterrupt])
@pytest.mark.parametrize('stage', ['session1', 'session2', 'headers1', 'headers2', 'adapter1', 'adapter2'])
def test_partial_client_acquisition_releases_every_acquired_resource(monkeypatch, stage, error):
    created, adapters = sessions(monkeypatch, stage=stage, error=error)
    with pytest.raises(error, match='primary startup failure'):
        StashTransport(False, url='http://stash/', api_key='synthetic-key', request_timeout=1, http_retries=0)
    assert all(session.closes == 1 for session in created)
    assert all(adapter.closes > 0 for adapter in adapters)


@pytest.mark.parametrize('stage', ['mount1', 'mount2', 'mount3', 'mount4'])
@pytest.mark.parametrize('after_mount', [False, True])
@pytest.mark.parametrize('error', [RuntimeError, KeyboardInterrupt])
def test_mount_failure_releases_the_incomplete_adapter_before_closing_sessions(monkeypatch, stage, after_mount, error):
    created, adapters = sessions(monkeypatch, stage=stage, error=error, after_mount=after_mount)
    with pytest.raises(error, match='primary startup failure'):
        StashTransport(False, url='http://stash/', api_key='', request_timeout=1, http_retries=0)
    assert all(session.closes == 1 for session in created)
    assert all(adapter.closes > 0 for adapter in adapters)
    assert adapters[-1].closes == 1  # failure never leaves two owners of this adapter


@pytest.mark.parametrize('kind', ['tagger', 'recognition'])
@pytest.mark.parametrize('error', [RuntimeError, KeyboardInterrupt])
@pytest.mark.parametrize('close_failure', [False, True])
def test_version_failure_keeps_the_primary_error_and_attempts_both_session_closes(monkeypatch, kind, error, close_failure):
    created, _ = sessions(monkeypatch, close_failure=close_failure)
    def version(client): raise error('primary version failure')
    monkeypatch.setattr(StashClient, '_require_supported_version', version)
    with pytest.raises(error, match='primary version failure') as raised:
        if kind == 'recognition':
            RecognitionStash(url='http://stash/', api_key='')
        else:
            StashClient(False, url='http://stash/', api_key='', request_timeout=1,
                        http_retries=0, scene_page_size=100, create_markers=True)
    assert len(created) == 2 and all(session.closes == 1 for session in created)
    if close_failure:
        assert any('secondary close failure' in note for note in raised.value.__notes__)


def test_partial_mount_and_secondary_close_fault_preserve_the_interruption(monkeypatch):
    created, adapters = sessions(monkeypatch, stage='mount2', error=KeyboardInterrupt,
                                 after_mount=True, close_failure=True)
    with pytest.raises(KeyboardInterrupt, match='primary startup failure'):
        StashTransport(False, url='http://stash/', api_key='', request_timeout=1, http_retries=0)
    assert all(session.closes == 1 for session in created) and adapters[0].closes == 1


def test_adapter_close_failure_does_not_skip_session_rollback(monkeypatch):
    import stash_ai.stash_api.read_connections as connections
    created, adapters = sessions(monkeypatch, stage='mount2', error=KeyboardInterrupt, after_mount=True)
    def close(adapter):
        adapter.closes += 1
        raise OSError('adapter close failure')
    monkeypatch.setattr(connections.ReadAdapter, 'close', close)
    with pytest.raises(KeyboardInterrupt, match='primary startup failure') as raised:
        StashTransport(False, url='http://stash/', api_key='', request_timeout=1, http_retries=0)
    assert all(session.closes == 1 for session in created) and adapters[0].closes == 1
    assert any('adapter close failure' in note for note in raised.value.__notes__)


def test_domain_client_setup_failure_also_closes_owned_sessions(monkeypatch):
    import stash_ai.stash_api.client as clients
    created, _ = sessions(monkeypatch)
    def initialize(client): raise KeyboardInterrupt('scene reader setup')
    monkeypatch.setattr(clients, 'SceneReads', initialize)
    with pytest.raises(KeyboardInterrupt, match='scene reader setup'):
        StashClient(False, url='http://stash/', api_key='', request_timeout=1,
                    http_retries=0, scene_page_size=100, create_markers=True)
    assert len(created) == 2 and all(session.closes == 1 for session in created)


def test_supported_version_client_keeps_its_sessions_until_normal_close(monkeypatch):
    created, _ = sessions(monkeypatch)
    monkeypatch.setattr(StashClient, '_require_supported_version', lambda client: None)
    client = StashClient(False, url='http://stash/', api_key='synthetic-key', request_timeout=1,
                         http_retries=0, scene_page_size=100, create_markers=True)
    assert all(session.closes == 0 and session.headers['ApiKey'] == 'synthetic-key' for session in created)
    client.close()
    assert all(session.closes == 1 for session in created)


@pytest.mark.parametrize('returned', ['other', 'missing', None, False, 1, '0', '01', '\u0661', '1 '])
def test_mismatched_or_malformed_refresh_never_deletes_the_used_requested_tag(returned):
    stash = PruneStash()
    target = stash.add('op:catalog-unused')
    other = stash.add('op:unrelated-unused')
    gql = stash.gql
    def response(query, variables=None):
        if 'query StashAIPruneTag(' in query:
            stash.tags[target]['scene_count'] = 1  # a new use after catalog collection
            tag = stash.tag(other)
            if returned == 'missing':
                del tag['id']
            elif returned != 'other':
                tag['id'] = returned
            return {'findTag': tag}
        return gql(query, variables)
    stash.gql = response
    with pytest.raises(ValueError, match='prune tag ID'):
        prune_tags(stash, ['op:'], report=lambda _: None)
    assert target in stash.tags and stash.tags[target]['scene_count'] == 1
    assert other in stash.tags and stash.deleted == []


def test_disappeared_tag_and_dry_run_remain_safe():
    stash = PruneStash()
    target = stash.add('op:empty')
    def removed(stash, identifier):
        del stash.tags[identifier]
        stash.before_refresh = None
    stash.before_refresh = removed
    assert prune_tags(stash, ['op:'], report=lambda _: None)['tags_pruned'] == 0
    assert not stash.deleted and target not in stash.tags
    preview = PruneStash(True)
    preview.add('op:empty')
    before = deepcopy(preview.tags)
    assert prune_tags(preview, ['op:'], report=lambda _: None)['tags_pruned'] == 1
    assert preview.tags == before and not preview.deleted and not preview.calls['refresh']
