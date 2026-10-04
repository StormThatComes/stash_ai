"""Verify the supported server floor before any domain read or write."""
from types import SimpleNamespace

import pytest
import requests

from stash_ai.stash_api.client import StashClient


def connect(dry_run=False):
    return StashClient(dry_run, url='http://stash.test', api_key='private-key',
                       request_timeout=7, http_retries=0,
                       scene_page_size=100, create_markers=True)


def serve(monkeypatch, data):
    calls, closed = [], []
    original_close = requests.Session.close

    def post(self, url, **kwargs):
        calls.append((url, kwargs))
        return SimpleNamespace(status_code=200, content=b'', raise_for_status=lambda: None,
                               json=lambda: {'data': data})

    def close(self):
        closed.append(self)
        original_close(self)

    monkeypatch.setattr(requests.Session, 'post', post)
    monkeypatch.setattr(requests.Session, 'close', close)
    return calls, closed


@pytest.mark.parametrize('version', ['0.31.1', 'v0.31.1', 'v0.31.2', 'v0.32.0', 'v1.0.0',
    'v0.31.1-175-gabcdef', 'v0.31.1-175-gabcdef-dirty', 'v0.31.1-dirty',
    '0.31.1+build.1', 'v0.32.0-rc.1'])
def test_supported_release_and_development_builds(monkeypatch, version):
    calls, closed = serve(monkeypatch, {'version': {'version': version}})
    current = connect()
    try:
        assert len(calls) == 1
        url, kwargs = calls[0]
        assert url == 'http://stash.test/graphql'
        assert kwargs['json'] == {'query': 'query StashAIVersion { version { version } }', 'variables': {}}
        assert kwargs['timeout'] == 7 and kwargs['allow_redirects'] is False
        assert not closed
        assert 'end_seconds' in current.scene_fields()
        assert 'parents' in current.tag_fields()
    finally:
        current.close()
    assert len(closed) == 2 and closed[0] is not closed[1]


@pytest.mark.parametrize('dry_run', [False, True])
@pytest.mark.parametrize('version', ['v0.30.0', 'v0.31.0', 'v0.31.0-200-gabcdef',
    'v0.31.1-rc.1', 'v0.31.1-alpha', None, '', 'develop', 'v0.31', 31])
def test_unsupported_or_unidentified_server_stops_before_domain_operations(monkeypatch, version, dry_run):
    calls, closed = serve(monkeypatch, {'version': {'version': version}})
    with pytest.raises(RuntimeError, match=r'Stash >= 0\.31\.1 is required'):
        connect(dry_run)
    assert len(calls) == 1 and 'StashAIVersion' in calls[0][1]['json']['query']
    assert len(closed) == 2 and closed[0] is not closed[1]


@pytest.mark.parametrize('data', [{}, {'version': None}, {'version': {}}])
def test_missing_version_fails_with_upgrade_guidance(monkeypatch, data):
    calls, closed = serve(monkeypatch, data)
    with pytest.raises(RuntimeError, match='Upgrade Stash'):
        connect()
    assert len(calls) == 1 and len(closed) == 2


def test_failed_version_request_closes_both_sessions(monkeypatch):
    _, closed = serve(monkeypatch, {})
    def post(*args, **kwargs):
        raise requests.ConnectionError('offline')
    monkeypatch.setattr(requests.Session, 'post', post)
    with pytest.raises(requests.ConnectionError, match='offline'):
        connect()
    assert len(closed) == 2
