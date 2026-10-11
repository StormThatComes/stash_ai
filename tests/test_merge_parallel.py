"""Parallel merged evidence preparation stays read-only and deterministic."""
from stash_ai.rethreshold.plans import ThresholdRequest
from stash_ai.rethreshold import scenes as scene_updates
from tests.helpers.evidence import inference_record
from tests.helpers.temporal import threshold_policy

import json
from collections import Counter
from pathlib import Path

import pytest

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
                prepared = scene_updates.plan(app, store, scene, config, request=ThresholdRequest(kinds=('markers',)))
                assert prepared
                scene_updates.apply(store, scene, config.source.key, prepared)

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
        assert first.error is None

        # The parent publishes one prepared scene while the same worker remains
        # alive for later scenes. This changes DB data_version but not results.
        with merge_store.StateStore(path, scope) as parent:
            parent.publish("1", {"scene": {"contributions": []}})

        second = merge_parallel._worker_evidence("2")
        assert second.error is None
        assert second.scene_id == "2"
    finally:
        merge_parallel._WORKER_CONTEXT.close()
        merge_parallel._WORKER_CONTEXT = None


@pytest.mark.parametrize('telemetry_enabled', [False, True])
def test_parallel_evidence_matches_serial_read_only(tmp_path, telemetry_enabled):
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

    from multiprocessing import get_context
    from stash_ai.runtime.resources import WorkerTelemetry
    telemetry = WorkerTelemetry(get_context('spawn'), 2, 4) if telemetry_enabled else None
    results = list(iter_parallel_evidence(
        app,
        path,
        scope,
        ["1", "2"],
        policies,
        policies,
        workers=2,
        **({'telemetry': telemetry} if telemetry is not None else {}),
        cache_bytes=64 * 2**20,
        memory_budget_bytes=512 * 2**20,
    ))
    if telemetry is not None:
        assert telemetry.snapshot()['computed_scenes'] == 2
        assert telemetry.snapshot()['busy'] == telemetry.snapshot()['ready_chunks'] == 0

    assert [row.scene_id for row in results] == ["1", "2"]
    for result in results:
        assert result.error is None
        payload = result.payload
        wanted_hits, wanted_spans, wanted_diagnostics = expected[result.scene_id]
        assert _normalized(payload.scene_hits) == _normalized(wanted_hits)
        assert _normalized(payload.spans) == _normalized(wanted_spans)
        assert dict(payload.diagnostics) == dict(wanted_diagnostics)
        from stash_ai.evidence.intervals import union_intervals
        assert _normalized(payload.markers) == _normalized(union_intervals(wanted_spans, borrow=True))

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
