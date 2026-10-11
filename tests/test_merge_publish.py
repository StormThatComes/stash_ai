"""Publication scheduling preserves one writer, accounting and refresh guards."""
from tests.helpers.temporal import threshold_policy

from collections import Counter
from types import SimpleNamespace

import pytest

from stash_ai.publication import fingerprints as performance
import stash_ai.storage.merge_checks as performance_merge_checks
from stash_ai.evidence import pipeline as merge
from stash_ai.storage import progress
from stash_ai.publication import parallel as merge_parallel
from stash_ai.publication import merged as merge_sync
from stash_ai.runtime.diagnostics import MergeDiagnostics
from stash_ai.publication.publisher import MergePublisher
from stash_ai.runtime.report import RunReport
from stash_ai.runtime.worker_result import WorkerResult
from stash_ai.storage.store import StateStore
from stash_ai.evidence.temporal.policy import TemporalPolicy
from tests.helpers.cli import runtime_args
from stash_ai.runtime.context import RuntimeContext


def worker_result(scene, *, replayed=0, error=None):
    payload = merge_parallel.MergeEvidence([scene], [], [],
        Counter(replayed=replayed) if replayed else Counter())
    return WorkerResult(scene, payload if error is None else None, error)


@pytest.fixture
def publication(tmp_path, monkeypatch):
    messages, writes, computed, worker_jobs = [], [], [], []
    app = RuntimeContext(
        DRY_RUN=False, APPLY_SCENE_TAGS=True, CREATE_MARKERS=True,
        TEMPORAL_POLICY=threshold_policy(), STASH_API_KEY='',
    )
    app.services.safe_error = lambda exc, **kwargs: str(exc)
    report = RunReport(False, messages.append)
    args = runtime_args(
        report=report,
        state_db=tmp_path / 'state.db', merge_workers=1, profile_merge=None,
        force_merge=False, ram_cache_gb=1, ram_budget_gb=3,
    )
    reasons = {}
    monkeypatch.setattr(
        performance_merge_checks, 'check_reason',
        lambda store, scene_id, key, fingerprint: reasons.get(scene_id, 'stale'))

    def evidence(app, store, scene_id, core_policies, scene_policies, diagnostics):
        computed.append(scene_id)
        store.cache.stats['hits'] += 2
        diagnostics.update(replayed=3)
        return [scene_id], []

    def sync(app, stash, resolver, store, current, hits, spans, **kwargs):
        assert kwargs['publication_key'] == 'policy'
        store.publish(current['id'], {'scene': {'hits': hits, 'spans': spans, 'contributions': []}})
        store.cache.stats['misses'] += 1
        writes.append(current['id'])
        return Counter(markers_updated=2)

    def workers(app, state_db, scope, scene_ids, core_policies, scene_policies, **kwargs):
        worker_jobs.append((scene_ids, kwargs))
        for scene_id in scene_ids:
            yield worker_result(scene_id, replayed=3)

    monkeypatch.setattr(merge, 'evidence', evidence)
    monkeypatch.setattr(merge_sync, 'sync_merged', sync)
    monkeypatch.setattr(merge_parallel, 'iter_parallel_evidence', workers)
    with StateStore(args.state_db, 'scope') as store, MergeDiagnostics(report) as diagnostics:
        publisher = MergePublisher(
            app, args, store, SimpleNamespace(find_scene_for_merge=lambda identifier:
                dict(id=identifier, tags=[], scene_markers=[]), merge_snapshots=lambda ids: None), object(), {}, {},
            publication_key='policy', stats=Counter(), report=report)
        yield SimpleNamespace(
            publisher=publisher, app=app, args=args, store=store,
            report=report, diagnostics=diagnostics, messages=messages,
            writes=writes, computed=computed, worker_jobs=worker_jobs, reasons=reasons)


def scenes(*ids):
    return {scene_id: {'id': scene_id, 'fingerprint': scene_id} for scene_id in ids}


@pytest.mark.parametrize('workers', [1, 4])
def test_local_and_worker_evidence_share_publication_accounting(publication, workers):
    p = publication
    p.args.merge_workers = workers
    p.publisher.run(scenes('1', '2'))

    assert p.writes == ['1', '2']
    assert p.publisher.stats['scenes_synced'] == 2
    assert p.publisher.stats['markers_updated'] == 4
    assert p.publisher.stats['replayed'] == 6
    assert p.publisher.stats['errors'] == 0
    assert p.diagnostics.counts['cache_misses'] == 2
    for scene_id in p.writes:
        assert p.store.snapshot(scene_id, 'scene') == ({'hits': [scene_id], 'spans': [], 'contributions': []}, None)
    if workers == 1:
        assert p.computed == ['1', '2']
        assert p.diagnostics.counts['cache_hits'] == 4
        assert p.diagnostics.scenes == 2
        assert p.publisher.stats['merge_parallel_scenes'] == 0
        assert not p.worker_jobs
    else:
        assert not p.computed
        assert p.publisher.stats['merge_parallel_scenes'] == 2
        assert p.worker_jobs == [(['1', '2'], dict(
            workers=2, cache_bytes=2**30, memory_budget_bytes=3 * 2**30))]


@pytest.mark.parametrize('dry_run,profile,ids', [
    (True, None, ('1', '2')),
    (False, 'profile.jsonl', ('1', '2')),
    (False, None, ('1',)),
])
def test_dry_run_profile_and_single_scene_use_local_evidence(publication, dry_run, profile, ids):
    p = publication
    p.app.DRY_RUN, p.args.profile_merge = dry_run, profile
    p.args.merge_workers = 4
    p.publisher.run(scenes(*ids))
    assert p.computed == list(ids)
    assert p.writes == list(ids)
    assert not p.worker_jobs


def test_one_changed_scene_avoids_workers_after_reuse_checks(publication):
    p = publication
    p.args.merge_workers = 4
    p.reasons.update({'1': 'reused', '3': 'reused'})
    p.publisher.run(scenes('1', '2', '3'))
    assert p.writes == p.computed == ['2']
    assert not p.worker_jobs
    assert p.publisher.stats['merge_reused'] == 2
    assert p.publisher.stats['scenes_synced'] == 1
    assert p.diagnostics.counts['cache_hits'] == 2


@pytest.mark.parametrize('workers', [1, 4])
@pytest.mark.parametrize('kind,occurrences', [('markers', False), ('scene', True)])
def test_pending_refresh_blocks_even_forced_publication(publication, workers, kind, occurrences):
    p = publication
    p.args.merge_workers, p.args.force_merge = workers, True
    p.app.APPLY_SCENE_TAGS = False
    p.app.TEMPORAL_POLICY = TemporalPolicy(occurrence_markers=occurrences)
    progress.request_refresh(p.store, ['1'], ['wd'], [kind])
    p.publisher.run(scenes('1', '2', '3'))
    assert p.writes == ['2', '3']
    assert p.publisher.stats['errors'] == 1
    assert p.publisher.stats['merge_refresh_blocked'] == 1
    assert p.publisher.stats['merge_forced'] == 2
    assert p.store.snapshot('1', 'scene') == (None, None)
    assert any('Fresh inference is still pending for wd/' + kind in message for message in p.messages)
    assert all('1' not in ids for ids, _ in p.worker_jobs)


@pytest.mark.parametrize('failure', ['scene', 'evidence', 'publication'])
def test_failed_worker_scene_is_not_published_and_later_scenes_continue(publication, monkeypatch, failure):
    p = publication
    p.args.merge_workers = 4

    def workers(*args, **kwargs):
        yield worker_result('wrong' if failure == 'scene' else '1',
                            error='worker failed' if failure == 'evidence' else None)
        yield worker_result('2', replayed=3)

    monkeypatch.setattr(merge_parallel, 'iter_parallel_evidence', workers)
    if failure == 'publication':
        sync = merge_sync.sync_merged

        def fail_first(*args, **kwargs):
            if args[4]['id'] == '1':
                raise RuntimeError('publication failed')
            return sync(*args, **kwargs)

        monkeypatch.setattr(merge_sync, 'sync_merged', fail_first)
    p.publisher.run(scenes('1', '2'))
    # Identity corruption is a protocol failure; ordinary scene errors continue.
    completed = 0 if failure == 'scene' else 1
    assert p.writes == ([] if failure == 'scene' else ['2'])
    assert not p.computed
    assert p.publisher.stats['errors'] == 2 - completed
    assert p.publisher.stats['scenes_synced'] == completed
    assert p.publisher.live['published'] == completed
    assert p.publisher.live['unresolved'] == (2 if failure == 'scene' else 0)
    assert p.publisher.stats['merge_parallel_scenes'] == completed


@pytest.mark.parametrize('workers', [1, 4])
@pytest.mark.parametrize('failure', ['evidence', 'publication'])
def test_local_failures_share_error_accounting_without_retry(publication, monkeypatch, workers, failure):
    p = publication
    p.args.merge_workers = workers
    p.reasons['2'] = 'reused'

    def fail(*args, **kwargs):
        raise RuntimeError('local failure')

    monkeypatch.setattr(merge if failure == 'evidence' else merge_sync,
                        'evidence' if failure == 'evidence' else 'sync_merged', fail)
    p.publisher.run(scenes('1', '2'))
    assert not p.writes
    assert not p.worker_jobs
    assert p.publisher.stats['errors'] == 1
    assert p.publisher.stats['scenes_synced'] == 0
    assert p.publisher.stats['merge_reused'] == 1
    if workers == 1:
        assert p.diagnostics.statuses == Counter(error=1, reused=1)


@pytest.mark.parametrize('completed', [0, 1])
def test_pool_failure_never_retries_completed_scenes(publication, monkeypatch, completed):
    p = publication
    p.args.merge_workers = 4

    def workers(*args, **kwargs):
        if completed:
            yield worker_result('1', replayed=3)
        raise RuntimeError('pool failed')

    monkeypatch.setattr(merge_parallel, 'iter_parallel_evidence', workers)
    p.publisher.run(scenes('1', '2', '3'))
    assert p.writes == (['1'] if completed else [])
    assert not p.computed
    assert p.publisher.stats['errors'] == 3 - completed
    assert p.publisher.stats['scenes_synced'] == completed
    assert p.publisher.live['published'] == completed
    assert p.publisher.live['unresolved'] == 3 - completed
    assert any('merge workers' in message and 'pool failed' in message for message in p.messages)


@pytest.mark.parametrize('failure', [None, RuntimeError, KeyboardInterrupt])
def test_parent_budget_is_shared_until_workers_close_and_restored(publication, monkeypatch, failure):
    from stash_ai.runtime.cache import RamCache
    p = publication
    p.args.merge_workers = 2
    p.args.ram_cache_gb, p.args.ram_budget_gb = 3 / 1024, 9 / 1024
    p.store.cache = RamCache(3 * 2**20, rss_budget=9 * 2**20)
    monkeypatch.setattr('stash_ai.runtime.memory.process_rss', lambda: 0)
    closed = []
    def workers(*args, **kwargs):
        try:
            assert p.store.cache.limit == 2**20
            assert p.store.cache.rss_budget == 3 * 2**20
            p.store.cache.put('oversized-parent-snapshot', bytes(2 * 2**20))
            assert p.store.cache.get('oversized-parent-snapshot') is None
            if failure:
                raise failure('pool failed')
            yield worker_result('1')
            yield worker_result('2')
        finally:
            # Pool shutdown completes before the parent can reclaim its budget.
            assert p.store.cache.limit == 2**20
            closed.append(True)
    monkeypatch.setattr(merge_parallel, 'iter_parallel_evidence', workers)
    if failure is KeyboardInterrupt:
        with pytest.raises(KeyboardInterrupt):
            p.publisher.run(scenes('1', '2'))
    else:
        p.publisher.run(scenes('1', '2'))
    assert closed == [True]
    assert p.store.cache.limit == 3 * 2**20
    assert p.store.cache.rss_budget == 9 * 2**20
    p.store.cache.put('after-workers', bytes(2 * 2**20))
    assert p.store.cache.get('after-workers') is not None


@pytest.mark.parametrize('workers', [1, 4])
@pytest.mark.parametrize('failure', ['missing', 'read_error', 'wrong_id'])
def test_reuse_refresh_failure_is_reported_per_scene_without_publishing(publication, monkeypatch, workers, failure):
    p = publication
    p.args.merge_workers = workers
    p.reasons.update({'1': 'reused', '2': 'reused'})
    def find(identifier):
        if identifier == '1':
            if failure == 'read_error':
                raise OSError('fresh read failed')
            if failure == 'missing':
                return None
            return dict(id='wrong', tags=[], scene_markers=[])
        return dict(id=identifier, tags=[], scene_markers=[])
    monkeypatch.setattr(p.publisher.stash, 'find_scene_for_merge', find)
    p.publisher.run(scenes('1', '2', '3'))
    assert p.writes == ['3']
    assert p.publisher.stats['errors'] == 1
    assert p.publisher.stats['merge_reused'] == 1


@pytest.mark.parametrize('workers', [1, 4])
def test_reuse_refresh_uses_bounded_bulk_queries_and_no_per_scene_reads(publication, monkeypatch, workers):
    p = publication
    p.args.merge_workers = workers
    p.app.SCENE_PAGE_SIZE = 2
    p.reasons.update({str(i): 'reused' for i in range(5)})
    batches = []
    def bulk(ids):
        batches.append(ids)
        return {identifier: dict(id=identifier, tags=[], scene_markers=[]) for identifier in ids}
    monkeypatch.setattr(p.publisher.stash, 'merge_snapshots', bulk, raising=False)
    monkeypatch.setattr(p.publisher.stash, 'find_scene_for_merge', lambda *_: pytest.fail('redundant per-scene read'))
    p.publisher.run(scenes('0', '1', '2', '3', '4'))
    assert batches == [['0', '1'], ['2', '3'], ['4']]
    assert p.publisher.stats['merge_reused'] == 5
    assert not p.writes and not p.computed


@pytest.mark.parametrize('returned,completed', [(('1',), 1), (('1', '2', '3'), 2), (('1', '1'), 1)])
def test_worker_result_coverage_errors_are_reported_without_republishing(publication, monkeypatch, returned, completed):
    p = publication
    p.args.merge_workers = 2
    closed = []
    def workers(*a, **kw):
        try:
            yield from (worker_result(scene) for scene in returned)
        finally:
            closed.append(True)
    monkeypatch.setattr(merge_parallel, 'iter_parallel_evidence', workers)
    p.publisher.run(scenes('1', '2'))
    assert p.writes == list(('1', '2')[:completed])
    assert p.publisher.stats['errors'] == max(1, 2-completed)
    assert closed == [True] and not p.computed
