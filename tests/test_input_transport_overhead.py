"""Successful reads use one worker; spilled hashes still detect early input changes."""
from types import SimpleNamespace
from threading import Event
import hashlib
import time

import pytest
import requests

from stash_ai.inference.inputs import RunMedia
from stash_ai.runtime.cache import RamCache
from stash_ai.runtime.profile import ProfileMetrics
from stash_ai.runtime.report import RunReport
from stash_ai.stash_api.transport import StashTransport
from stash_ai.stash_api import deadlines


def test_twenty_successful_graphql_reads_use_twenty_workers_and_keep_accounting(monkeypatch):
    thread, workers = deadlines.Thread, []
    def create(*args, **kwargs):
        if kwargs.get('name') == 'stash-deadline-read':
            workers.append(True)
        return thread(*args, **kwargs)
    monkeypatch.setattr(deadlines, 'Thread', create)
    client = StashTransport(False, url='http://stash.test', api_key='', request_timeout=1., http_retries=0)
    report = client.report = RunReport(write=lambda _: None)
    report.profile = ProfileMetrics()
    data = b'{"data":{"ping":true}}'
    closed = []
    def response(*args, **kwargs):
        result = requests.Response()
        result.status_code, result._content = 200, data
        result.close = lambda: closed.append(True)
        return result
    client.api.post = response
    try:
        with report.model('wd'):
            report.work('fresh')
            for _ in range(20):
                assert client.gql('query StashAIPing { ping }') == {'ping': True}
        assert len(workers) == len(closed) == 20
        api = [r for r in report.profile.export() if r['kind'] == 'api']
        assert sum(r['units'] for r in api) == 20 * len(data)
        assert {r['model'] for r in api} == {'wd'}
        assert {r['work'] for r in api} == {'fresh'}
    finally:
        client.close()


def test_json_stall_remains_bounded_and_response_is_eventually_closed():
    release, closed = Event(), Event()
    client = StashTransport(False, url='http://stash.test', api_key='', request_timeout=.03, http_retries=0)
    client.api.post = lambda *a, **k: SimpleNamespace(status_code=200, content=b'',
        raise_for_status=lambda: None, close=closed.set,
        json=lambda: (release.wait(2) or {'data': {}}))
    start = time.monotonic()
    try:
        with pytest.raises(deadlines.ReadDeadlineExceeded):
            client.gql('query StashAIPing { ping }')
        assert time.monotonic()-start < .3
    finally:
        release.set()
        assert closed.wait(1)
        client.close()


def test_spilled_hashes_detect_changes_after_thousands_of_zero_cache_reads():
    changed = False
    def fetch(url):
        return b'changed' if changed and url == 'url-0' else url.encode()
    media = RunMedia(SimpleNamespace(get_media=fetch), RamCache(0))
    try:
        for i in range(3000):
            media.get_media(f'url-{i}')
        assert len(media.hashes) == 3000
        assert media.hashes._db.execute('PRAGMA cache_size').fetchone() == (-64,)
        assert not any(isinstance(value, dict) for value in vars(media.hashes).values())
        assert media.get_verified_media('url-0')[1] == hashlib.sha256(b'url-0').hexdigest()
        changed = True
        with pytest.raises(RuntimeError, match='changed during model passes'):
            media.get_media('url-0')
    finally:
        media.close()
    assert not media.hashes
