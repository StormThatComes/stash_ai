"""Parallel merged evidence preparation stays read-only and deterministic."""
from tests.helpers.evidence import inference_record
from tests.helpers.temporal import threshold_policy

import json
from collections import Counter
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
from threading import Event

import pytest

from stash_ai.rethreshold import markers as marker_updates
from tests.helpers.runtime import runtime as app
from stash_ai.evidence import pipeline as merge
import stash_ai.storage.store as merge_store
from stash_ai.publication import parallel as merge_parallel
import stash_ai.evidence.identity as thresholds_identity
from stash_ai.publication.parallel import iter_parallel_evidence
from stash_ai.evidence.temporal.policy import TemporalPolicy
from stash_ai.models import ModelSettings, SOURCES
from tests.helpers.evidence import empty_policies, segment


pytestmark = pytest.mark.usefixtures("reset_temporal")


def _outputs(source, label):
    marker = segment(
        source,
        [[(label, .92)], [(label, .90)], [], [(label, .91)]],
        identity="scene-input",
    )
    marker["meta"]["scene_threshold"] = .4
    scene = inference_record(**{
        "mode": "scene",
        "meta": dict(marker["meta"]),
        "items": [{"tag": label, "score": .92}],
    })
    return {"scene": scene, "markers": marker}


def _normalized(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


@pytest.mark.parametrize('rethresholded', [False, True])
def test_parallel_worker_tolerates_parent_publication_commits(tmp_path, rethresholded):
    app.TEMPORAL_POLICY = threshold_policy()
    path = tmp_path / "state.sqlite3"
    scope = merge_store.scope_for(app.STASH_URL)
    policies = empty_policies()

    with merge_store.StateStore(path, scope) as store:
        store.put("1", "wd", _outputs("wd", "one"))
        store.put("2", "dtq", _outputs("dtq", "two"))
        if rethresholded:
            for scene, model in (('1', 'wd'), ('2', 'dtq')):
                config = ModelSettings(SOURCES[model], Path('.'), scene_threshold=.4, marker_threshold=.95, batch_size=4)
                assert marker_updates.apply(app, store, scene, config)

    merge_parallel._init_worker(
        str(path),
        scope,
        merge_parallel.runtime_settings(app),
        policies,
        policies,
        64 * 2**20,
        512 * 2**20,
    )
    worker = merge_parallel._WORKER_CONTEXT.store
    try:
        if rethresholded:
            assert worker.preview_source
            assert worker.model_record('1', 'markers', 'wd')[thresholds_identity.PUBLICATION_THRESHOLD] == .95
        first = merge_parallel._worker_evidence("1")
        assert first[-1] is None

        # The parent publishes one prepared scene while the same worker remains
        # alive for later scenes. This changes DB data_version but not results.
        with merge_store.StateStore(path, scope) as parent:
            parent.publish("1", {"scene": {"contributions": []}})

        second = merge_parallel._worker_evidence("2")
        assert second[-1] is None
        assert second[0] == "2"
    finally:
        merge_parallel._WORKER_CONTEXT.close()
        merge_parallel._WORKER_CONTEXT = None


def test_parallel_evidence_matches_serial_read_only(tmp_path):
    app.TEMPORAL_POLICY = threshold_policy()
    path = tmp_path / "state.sqlite3"
    scope = merge_store.scope_for(app.STASH_URL)
    policies = empty_policies()

    with merge_store.StateStore(path, scope) as store:
        store.put("1", "wd", _outputs("wd", "one"))
        store.put("2", "dtq", _outputs("dtq", "two"))

    expected = {}
    with merge_store.StateStore(path, scope, True) as store:
        for scene_id in ("1", "2"):
            diagnostics = Counter()
            hits, spans = merge.evidence(
                app, store, scene_id, policies, policies, diagnostics)
            expected[scene_id] = hits, spans, diagnostics

    results = list(iter_parallel_evidence(
        app,
        path,
        scope,
        ["1", "2"],
        policies,
        policies,
        workers=2,
        cache_bytes=64 * 2**20,
        memory_budget_bytes=512 * 2**20,
    ))

    assert [row[0] for row in results] == ["1", "2"]
    for scene_id, hits, spans, diagnostics, error in results:
        assert error is None
        wanted_hits, wanted_spans, wanted_diagnostics = expected[scene_id]
        assert _normalized(hits) == _normalized(wanted_hits)
        assert _normalized(spans) == _normalized(wanted_spans)
        assert dict(diagnostics) == dict(wanted_diagnostics)

    # Worker StateStores are dry-run/read-only overlays.
    before = path.read_bytes()
    list(iter_parallel_evidence(
        app,
        path,
        scope,
        ["1", "2"],
        policies,
        policies,
        workers=2,
        cache_bytes=0,
        memory_budget_bytes=256 * 2**20,
    ))
    assert path.read_bytes() == before


def test_evidence_queue_preserves_order_and_bounds_fast_results(monkeypatch):
    second_finished = Event()
    consumed = []

    def scenes():
        for scene in range(100):
            consumed.append(scene)
            yield scene

    def evidence(scene):
        if scene == 0:
            assert second_finished.wait(5), 'second scene did not finish'
        elif scene == 1:
            second_finished.set()
        return scene

    monkeypatch.setattr(merge_parallel, '_worker_evidence', evidence)
    with ThreadPoolExecutor(max_workers=2) as executor:
        results = merge_parallel._bounded_evidence(executor, scenes(), 2)
        try:
            # Scene 1 completes first; scene 0 must still be published first.
            assert next(results) == 0
            # Only two queued results plus the one held by the publisher are
            # prepared, even when the workers finish faster than publication.
            assert consumed == [0, 1, 2]
            assert list(results) == list(range(1, 100))
        finally:
            results.close()


@pytest.mark.parametrize('failure', [False, True])
def test_evidence_queue_cancels_pending_work_on_stop(failure):
    futures = []

    class Executor:
        def submit(self, function, scene):
            future = Future()
            if scene == 'first':
                if failure:
                    future.set_exception(RuntimeError('worker failed'))
                else:
                    future.set_result(scene)
            futures.append(future)
            return future

    results = merge_parallel._bounded_evidence(
        Executor(), ['first', 'second', 'third'], 2)
    if failure:
        with pytest.raises(RuntimeError, match='worker failed'):
            next(results)
    else:
        assert next(results) == 'first'
        results.close()
    assert all(future.cancelled() for future in futures[1:])
