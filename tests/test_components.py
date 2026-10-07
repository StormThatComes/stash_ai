"""Regression coverage for the extracted client, resolver and state codec."""
from types import SimpleNamespace
import time

import pytest
import requests

from stash_ai.stash_api.client import StashClient
from stash_ai.stash_api.transport import safe_error
from stash_ai.storage.codec import encode, decode
from stash_ai.stash_api.tags import TagResolver
from stash_ai.models import JOYTAG

pytestmark = pytest.mark.usefixtures("supported_stash_version")


def client(url, key, *, markers=True, page_size=100, timeout=30):
    return StashClient(True, url=url, api_key=key, request_timeout=timeout,
                       http_retries=2, scene_page_size=page_size, create_markers=markers)


def test_clients_keep_connection_and_output_settings_isolated(monkeypatch):
    first = client('http://first.test/stash', 'first-secret', markers=False, page_size=3, timeout=7)
    second = client('http://second.test', 'second-secret', page_size=5, timeout=11)
    calls = []

    def post(url, **kwargs):
        calls.append((url, kwargs))
        return SimpleNamespace(status_code=200, content=b'', raise_for_status=lambda: None,
                               json=lambda: {'data': {'findScenes': {'scenes': [{'id': '1'}]}}})

    try:
        for current in (first, second):
            monkeypatch.setattr(current.api, 'post', post)
            assert list(current.iter_scenes()) == [{'id': '1'}]
        assert [url for url, _ in calls] == ['http://first.test/stash/graphql', 'http://second.test/graphql']
        assert [kw['timeout'] for _, kw in calls] == pytest.approx([7, 11], abs=.1)
        assert [kw['json']['variables']['filter']['per_page'] for _, kw in calls] == [3, 5]
        assert 'scene_markers' not in calls[0][1]['json']['query']
        assert 'scene_markers' in calls[1][1]['json']['query']
        assert first.api.headers['ApiKey'] == first.media.headers['ApiKey'] == 'first-secret'
        assert second.api.headers['ApiKey'] == second.media.headers['ApiKey'] == 'second-secret'
        assert all(kw['allow_redirects'] is False for _, kw in calls)
    finally:
        first.close()
        second.close()


def http_response(status, data=None):
    def raise_for_status():
        if status >= 400:
            raise requests.HTTPError(f'HTTP {status}')
    return SimpleNamespace(status_code=status, content=b'', headers={}, close=lambda: None,
                           raise_for_status=raise_for_status,
                           json=lambda: {'data': data or {'ok': True}})


@pytest.mark.parametrize('failure', ['status', 'connection'])
def test_read_only_graphql_retries_transient_failures(monkeypatch, failure):
    current = client('http://stash.test', 'private-key')
    calls = 0
    sleeps = []

    def post(url, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 1:
            if failure == 'connection':
                raise requests.ConnectionError('temporary connection failure')
            return http_response(503)
        return http_response(200)

    try:
        monkeypatch.setattr(current.api, 'post', post)
        monkeypatch.setattr(time, 'sleep', sleeps.append)
        assert current.gql('query StashAIRead { ping }') == {'ok': True}
        assert calls == 2
        assert sleeps == [.5]
    finally:
        current.close()


def test_allowlisted_sql_read_mutation_is_retried(monkeypatch):
    from stash_ai.stash_api.sql import OPERATION, SQL
    current = client('http://stash.test', 'private-key')
    calls = 0

    def post(url, **kwargs):
        nonlocal calls
        calls += 1
        return http_response(503 if calls == 1 else 200)

    try:
        monkeypatch.setattr(current.api, 'post', post)
        monkeypatch.setattr(time, 'sleep', lambda *_: None)
        variables = {'sql': SQL['tags'], 'args': ['[]']}
        assert current.gql(OPERATION, variables) == {'ok': True}
        assert calls == 2
    finally:
        current.close()


def test_graphql_mutation_is_never_retried(monkeypatch):
    current = StashClient(False, url='http://stash.test', api_key='private-key',
                          request_timeout=30, http_retries=2, scene_page_size=100,
                          create_markers=True)
    calls = 0

    def post(url, **kwargs):
        nonlocal calls
        calls += 1
        return http_response(503)

    try:
        monkeypatch.setattr(current.api, 'post', post)
        monkeypatch.setattr(time, 'sleep', lambda *_: pytest.fail('mutation entered retry backoff'))
        with pytest.raises(requests.HTTPError, match='503'):
            current.create_marker({'scene_id': '1', 'primary_tag_id': '2', 'seconds': 1})
        assert calls == 1
    finally:
        current.close()


def test_nontransient_read_failure_is_not_retried(monkeypatch):
    current = client('http://stash.test', 'private-key')
    calls = 0

    def post(url, **kwargs):
        nonlocal calls
        calls += 1
        return http_response(400)

    try:
        monkeypatch.setattr(current.api, 'post', post)
        monkeypatch.setattr(time, 'sleep', lambda *_: pytest.fail('400 entered retry backoff'))
        with pytest.raises(requests.HTTPError, match='400'):
            current.gql('query StashAIRead { ping }')
        assert calls == 1
    finally:
        current.close()


def test_media_redirect_keeps_credentials_on_configured_origin(monkeypatch):
    current = client('http://stash.test', 'private-key')
    calls = []

    def get(url, **kwargs):
        calls.append((url, kwargs))
        return SimpleNamespace(status_code=302, headers={'Location': 'http://other.test/sprite'})

    try:
        monkeypatch.setattr(current.media, 'get', get)
        with pytest.raises(ValueError, match='different Stash origin'):
            current.get_media('http://stash.test/sprite')
        assert len(calls) == 1
        assert calls[0][1]['allow_redirects'] is False
    finally:
        current.close()


def test_error_redaction_uses_explicit_key_and_masks_media_tokens():
    message = safe_error(RuntimeError('private-key http://stash.test/media?apikey=url-secret&token=other-secret'),
                         api_key='private-key')
    assert message == '[redacted] http://stash.test/media?apikey=[redacted]&token=[redacted]'


def test_tag_resolver_requires_explicit_source():
    import inspect

    parameter = inspect.signature(TagResolver).parameters["source"]
    assert parameter.default is inspect.Parameter.empty
    assert parameter.kind is inspect.Parameter.KEYWORD_ONLY


def test_resolvers_choose_missing_tag_behavior_independently():
    from stash_ai.evidence.labels import AI
    stash = SimpleNamespace(dry_run=True, find_tag=lambda name: None)
    skip = TagResolver(stash, source=AI, create_missing=False)
    create = TagResolver(stash, source=AI, create_missing=True)
    assert skip.resolve('ai:label') is None
    assert create.resolve('ai:label') == 'dry-run:ai:label'
    assert skip.cache['ai:label'] is None
    assert create.cache['ai:label'] == 'dry-run:ai:label'


def test_state_codec_preserves_pre_refactor_blob_format():
    # Produced by main e2bbcaa before moving the codec out of stash_merge.py.
    saved = bytes.fromhex(
        '789cab562a48cd4bc9cc4b57b2aa56ca4d2cca4e2d2a56b28a8eadd5512a4ece2f4a55b232d0b3d051'
        '2a49042a504accb43abae7f0e6a34d47da956a0152b81592')
    value = {'tag': 'ai:żółć', 'score': 0.8, 'pending': {'markers': []}}
    assert decode(saved) == value
    assert encode(value) == saved
