"""Parallel merged evidence preparation stays read-only and deterministic."""
import json
from collections import Counter

import pytest

from stash_ai import cli as app
from stash_ai import merge, merge_parallel, performance
from stash_ai.merge_parallel import iter_parallel_evidence
from stash_ai.temporal import TemporalPolicy
from tests.helpers.evidence import empty_policies, segment


pytestmark = pytest.mark.usefixtures("reset_temporal")


def _outputs(source, label):
    marker = segment(
        source,
        [[(label, .92)], [(label, .90)], [], [(label, .91)]],
        identity="scene-input",
    )
    marker["meta"]["scene_threshold"] = .4
    scene = {
        "mode": "scene",
        "meta": dict(marker["meta"]),
        "items": [{"tag": label, "score": .92}],
    }
    return {"scene": scene, "markers": marker}


def _normalized(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def test_parallel_evidence_matches_serial_read_only(tmp_path):
    app.TEMPORAL_POLICY = TemporalPolicy(enabled=False)
    path = tmp_path / "state.sqlite3"
    scope = merge.scope_for(app.STASH_URL)
    policies = empty_policies()

    with merge.StateStore(path, scope) as store:
        store.put("1", "wd", _outputs("wd", "one"))
        store.put("2", "dtq", _outputs("dtq", "two"))

    expected = {}
    with merge.StateStore(path, scope, True) as store:
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


def test_parallel_worker_tolerates_parent_publication_commits(tmp_path):
    app.TEMPORAL_POLICY = TemporalPolicy(enabled=False)
    path = tmp_path / "state.sqlite3"
    scope = merge.scope_for(app.STASH_URL)
    policies = empty_policies()

    with merge.StateStore(path, scope) as store:
        store.put("1", "wd", _outputs("wd", "one"))
        store.put("2", "dtq", _outputs("dtq", "two"))

    merge_parallel._init_worker(
        str(path),
        scope,
        merge_parallel.runtime_settings(app),
        policies,
        policies,
        64 * 2**20,
        512 * 2**20,
    )
    worker = merge_parallel._WORKER_STORE
    try:
        first = merge_parallel._worker_evidence("1")
        assert first[-1] is None

        with merge.StateStore(path, scope) as parent:
            with parent.connection:
                performance.record_check(parent, "1", "parent-policy", "parent-fingerprint")

        second = merge_parallel._worker_evidence("2")
        assert second[-1] is None
        assert second[0] == "2"
    finally:
        worker.close()
        merge_parallel._WORKER_STORE = None
        merge_parallel._WORKER_APP = None
        merge_parallel._WORKER_CORE_POLICIES = None
        merge_parallel._WORKER_SCENE_POLICIES = None


def test_normal_preview_still_rejects_parent_publication_commits(tmp_path):
    path = tmp_path / "state.sqlite3"
    scope = merge.scope_for(app.STASH_URL)

    with merge.StateStore(path, scope) as store:
        store.put("1", "wd", _outputs("wd", "one"))
        store.put("2", "dtq", _outputs("dtq", "two"))

    with merge.StateStore(path, scope, True) as preview:
        assert preview.records("1", "scene")
        with merge.StateStore(path, scope) as parent:
            with parent.connection:
                performance.record_check(parent, "1", "parent-policy", "parent-fingerprint")
        with pytest.raises(RuntimeError, match="source state DB changed during dry-run"):
            preview.records("2", "scene")
