"""Visible candidate reads, honest batch progress and heartbeat lifecycle."""
from io import StringIO
from threading import Event, Thread, current_thread
from types import SimpleNamespace

import pytest
from tqdm import tqdm

from stash_ai import prune
from tests.helpers.stash import Stash


@pytest.fixture
def display(monkeypatch):
    observed = SimpleNamespace(bars=[], threads=[], ticks=[], refreshed=Event())
    caller = current_thread()

    class Progress(tqdm):
        def __init__(self, *args, **kwargs):
            kwargs['file'] = StringIO()
            super().__init__(*args, **kwargs)
            observed.bars.append(self)

        def set_postfix_str(self, value, refresh=True):
            super().set_postfix_str(value, refresh=refresh)
            if current_thread() is not caller:
                observed.ticks.append((self.n, value))
                observed.refreshed.set()

    def thread(*args, **kwargs):
        result = Thread(*args, **kwargs)
        observed.threads.append(result)
        return result

    monkeypatch.setattr(prune, 'tqdm', Progress)
    monkeypatch.setattr(prune, 'Thread', thread)
    return observed


@pytest.mark.parametrize('dry', [False, True])
def test_progress_counts_completed_lookups_including_missing_tags(display, dry):
    stash = Stash(dry)
    ids = [stash.tag(f'ai:tag_{i}')['id'] for i in range(205)]
    del stash.tags['ai:tag_204']  # A tag can disappear between discovery and lookup.
    original = stash.gql
    requests = []
    caller = current_thread()

    def gql(query, variables=None):
        if 'StashAIPruneCandidates' in query:
            bar = display.bars[0]
            assert not bar.disable and bar.total == 205
            assert current_thread() is caller
            requests.append((bar.n, len(variables['ids'])))
        return original(query, variables)

    stash.gql = gql
    result = prune.PruneClient(stash).catalog(iter(ids))
    assert len(result) == 204
    assert requests == [(0, 100), (100, 100), (200, 5)]
    assert len(display.bars) == 1
    assert display.bars[0].n == display.bars[0].total == 205
    assert display.bars[0].disable
    assert all(not thread.is_alive() for thread in display.threads)
    assert not stash.writes


@pytest.mark.parametrize('failure', [None, RuntimeError, KeyboardInterrupt])
def test_wait_time_refreshes_without_advancing_or_extra_requests_and_stops(display, failure):
    stash = Stash()
    identifier = stash.tag('ai:empty')['id']
    original = stash.gql
    requests = []
    caller = current_thread()

    def gql(query, variables=None):
        requests.append(query)
        assert current_thread() is caller
        if 'StashAIPruneCandidates' in query:
            bar = display.bars[0]
            assert bar.n == 0 and 'waiting for Stash: 0s' in bar.postfix
            assert display.refreshed.wait(5), 'Wait time did not refresh during the blocked API call'
            assert bar.n == 0
            assert 'waiting for Stash: ' in display.ticks[0][1]
            assert not display.ticks[0][1].endswith(': 0s')
            if failure is not None:
                raise failure('interrupted read')
        return original(query, variables)

    stash.gql = gql
    client = prune.PruneClient(stash)
    if failure is None:
        assert set(client.catalog([identifier])) == {identifier}
    else:
        with pytest.raises(failure, match='interrupted read'):
            client.catalog([identifier])
    assert len(requests) == 2  # Capabilities plus one read; heartbeat makes no calls.
    assert display.bars[0].n == (1 if failure is None else 0)
    assert display.bars[0].disable
    assert display.threads and all(not thread.is_alive() for thread in display.threads)
    assert not stash.writes


@pytest.mark.parametrize('invalid', ['unexpected_id', 'unknown_count'])
def test_invalid_batch_leaves_partial_progress_and_never_starts_deleting(display, invalid):
    stash = Stash()
    ids = [stash.tag(f'ai:tag_{i}')['id'] for i in range(101)]
    original = stash.gql

    def gql(query, variables=None):
        response = original(query, variables)
        if 'StashAIPruneCandidates' in query and variables['ids'] == [ids[-1]]:
            tag = response['findTags']['tags'][0]
            if invalid == 'unexpected_id':
                tag['id'] = '9999'
            else:
                tag['scene_marker_count'] = None
        return response

    stash.gql = gql
    with pytest.raises(ValueError):
        prune.prune_tags(stash, candidate_ids=ids, report=lambda _: None)
    assert len(display.bars) == 1 and display.bars[0].desc == 'Check tag usage'
    assert display.bars[0].n == 100 and display.bars[0].total == 101
    assert display.bars[0].disable
    assert all(not thread.is_alive() for thread in display.threads)
    assert not stash.writes
