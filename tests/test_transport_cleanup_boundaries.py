from collections import Counter
from threading import Event
from time import monotonic
from types import SimpleNamespace
import pytest

from stash_ai.cleanup.sql_session import CleanupSqlSession
from stash_ai.stash_api.deadlines import call_with_deadline, ReadDeadlineExceeded
from stash_ai.stash_api.transport import StashTransport


def test_failed_cancel_preserves_deadline_and_attempts_all_owned_abort(monkeypatch):
    from stash_ai.stash_api.read_connections import ReadControl
    entered, release, done = Event(), Event(), Event()
    attempted = []
    def broken(self): attempted.append('control'); raise OSError('socket cleanup failed')
    monkeypatch.setattr(ReadControl, 'cancel', broken)
    def operation():
        try: entered.set(); release.wait(2)
        finally: done.set()
    def cancel(): attempted.append('caller'); raise ValueError('caller cleanup failed')
    try:
        with pytest.raises(ReadDeadlineExceeded) as caught:
            call_with_deadline(operation, monotonic()+.05, cancel=cancel)
        assert entered.is_set() and attempted == ['control', 'caller']
        assert any('cleanup failed' in note for note in caught.value.__notes__)
    finally:
        release.set(); assert done.wait(1)


def test_read_retry_survives_discarded_response_close_failure():
    client = StashTransport(False, url='http://stash/', api_key='', request_timeout=2, http_retries=1)
    calls = []
    good = SimpleNamespace(status_code=200, close=lambda: None)
    def request(timeout):
        calls.append(timeout)
        return SimpleNamespace(status_code=503, close=lambda: (_ for _ in ()).throw(OSError('close failed'))) if len(calls)==1 else good
    try:
        assert client._request_response(request, read=True, deadline=monotonic()+2) is good
        assert len(calls) == 2
    finally: client.close()


@pytest.mark.parametrize('value', [None, [], 1, {}, {'rows': None}, {'rows': [1]}])
def test_sql_envelopes_fail_with_controlled_error(value):
    session = CleanupSqlSession(SimpleNamespace(gql=lambda *a: {'querySQL': value}), Counter(), page_size=100)
    with pytest.raises(ValueError, match='Invalid SQL cleanup response'): session.read('probe')


@pytest.mark.parametrize('bounds', [[], [[]], [[1,2]], [[1,2,3,4]], [[-1,2,3]],
    [[True,2,3]], [[1.0,2,3]], [['1',2,3]], [[1,2,3],[4,5,6]]])
def test_sql_probe_validates_exact_integral_nonnegative_bounds(bounds):
    session = CleanupSqlSession(None, Counter(), page_size=100)
    session.read = lambda key, *a: [] if key=='probe' else [[1]] if key=='json' else bounds
    with pytest.raises(ValueError, match='cursor bounds'): session.probe()
    assert session.bounds is None


def test_sql_probe_accepts_empty_library_bounds():
    session = CleanupSqlSession(None, Counter(), page_size=100)
    session.read = lambda key, *a: [] if key=='probe' else [[1]] if key=='json' else [[0,0,0]]
    session.probe(); assert session.bounds == (0,0,0)
