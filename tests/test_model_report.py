"""Offline model utility and leave-one-model-out ablation reporting."""
from tests.helpers.evidence import inference_record
from tests.helpers.temporal import threshold_policy


from stash_ai.reports.model.comparison import _marker_key, _marker_delta
from stash_ai.evidence.marker_identity import is_occurrence

from dataclasses import replace
import json

import pytest

from tests.helpers.runtime import runtime as app
from stash_ai.evidence import pipeline as merge
import stash_ai.evidence.intervals as merge_intervals
import stash_ai.storage.store as merge_store
from stash_ai.reports.model import report as model_report
from stash_ai.reports.model import workers as model_workers
from stash_ai.evidence.temporal.policy import TemporalPolicy


pytestmark = pytest.mark.usefixtures("reset_temporal")


def empty_policies():
    from stash_ai.models import SOURCES
    return {source: frozenset() for source in SOURCES}


def inference_pair(label, *, score=.9, identity="same", marker_threshold=.8):
    meta = {
        "origin": "inference",
        "inference_signature": "sig-" + label,
        "frame_input_signature": identity,
        "scene_threshold": .4,
        "marker_threshold": marker_threshold,
        "marker_score_floor": marker_threshold * .85,
    }
    return {kind: inference_record(**record) for kind, record in ({
        "scene": {
            "mode": "scene",
            "meta": dict(meta),
            "items": [{"tag": label, "score": score}],
        },
        "markers": {
            "mode": "frames",
            "meta": dict(meta),
            "items": [
                {"start": 0.0, "end": 1.0, "tags": [[label, score]]},
                {"start": 1.0, "end": 2.0, "tags": [[label, score]]},
            ],
        },
    }).items()}


def _report_marker(tag, start, end, *, occurrence=False, extras=()):
    return {
        "tag": tag,
        "start": start,
        "end": end,
        "contributors": [{"derived": "occurrence" if occurrence else "range"}],
        "extra_tags": list(extras),
    }


def _naive_overlap(left, right):
    if left["tag"] != right["tag"] or bool(is_occurrence(left)) != bool(is_occurrence(right)):
        return False
    a0, b0 = float(left["start"]), float(right["start"])
    a1 = float(left["end"]) if left["end"] is not None else a0
    b1 = float(right["end"]) if right["end"] is not None else b0
    return max(a0, b0) <= min(a1, b1) + .0005


def _naive_marker_delta(full, reduced):
    reduced_keys = {_marker_key(item) for item in reduced}
    full_keys = {_marker_key(item) for item in full}
    full_overlap = [any(_naive_overlap(marker, other) for other in reduced) for marker in full]
    reduced_overlap = [any(_naive_overlap(marker, other) for other in full) for marker in reduced]
    return {
        "lost": sum(not overlap for overlap in full_overlap),
        "changed": sum(
            overlap and _marker_key(marker) not in reduced_keys
            for marker, overlap in zip(full, full_overlap)
        ),
        "added": sum(not overlap for overlap in reduced_overlap),
        "exact_removed": len(full_keys - reduced_keys),
        "exact_added": len(reduced_keys - full_keys),
        "occurrences_lost": sum(
            is_occurrence(marker) and not overlap
            for marker, overlap in zip(full, full_overlap)
        ),
    }


def test_marker_delta_linear_overlap_matches_quadratic_reference():
    full = [
        _report_marker("a", 0.0, 1.0, extras=("x",)),
        _report_marker("b", 1.0, 2.0),
        _report_marker("a", 3.0, None, occurrence=True),
        _report_marker("c", 5.0, 5.2),
    ]
    reduced = [
        _report_marker("a", 0.2, 0.4, extras=("y",)),
        # A second component still overlaps the same full component through the
        # report's 0.5 ms matching tolerance.
        _report_marker("a", 1.0004, 1.2),
        _report_marker("b", 1.5, 2.5),
        _report_marker("a", 3.0004, None, occurrence=True),
        _report_marker("d", 6.0, 7.0),
    ]
    full.sort(key=lambda item: (item["start"], item["tag"]))
    reduced.sort(key=lambda item: (item["start"], item["tag"]))

    assert _marker_delta(full, reduced) == _naive_marker_delta(full, reduced)


def test_selected_sources_replays_consensus_instead_of_filtering_finished_markers(tmp_path):
    path = tmp_path / "state.sqlite3"
    policies = empty_policies()
    app.TEMPORAL_POLICY = replace(
        TemporalPolicy(),
        occurrence_markers=False,
        shared_scene_tags=False,
        shared_marker_extras=False,
    )

    weak = .72
    with merge_store.StateStore(path, "scope") as state:
        for source in ("wd", "dtq"):
            outputs = inference_pair("consensus_only", score=weak, identity="identical-input")
            outputs["scene"]["items"] = []
            outputs["scene"]["frames"] = []
            outputs["scene"]["meta"]["inference_signature"] = source + "-scene"
            outputs["markers"]["meta"]["inference_signature"] = source + "-markers"
            state.put("1", source, outputs)

    with merge_store.StateStore(path, "scope", True) as state:
        _, full = merge.evidence(app, state, "1", policies, policies)
        _, without_dtq = merge.evidence(
            app, state, "1", policies, policies, selected_sources={"wd"})

    assert merge_intervals.union_intervals(full)
    assert merge_intervals.union_intervals(without_dtq) == []


def test_ablation_report_distinguishes_redundant_and_unique_models(tmp_path):
    path = tmp_path / "state.sqlite3"
    policies = empty_policies()
    app.TEMPORAL_POLICY = threshold_policy()

    with merge_store.StateStore(path, "scope") as state:
        state.put("1", "wd", inference_pair("shared"))
        dtq = inference_pair("shared")
        dtq["scene"]["meta"]["inference_signature"] = "dtq-scene"
        dtq["markers"]["meta"]["inference_signature"] = "dtq-markers"
        state.put("1", "dtq", dtq)

        unique = inference_pair("unique")
        unique["scene"]["meta"]["inference_signature"] = "joy-scene"
        unique["markers"]["meta"]["inference_signature"] = "joy-markers"
        state.put("1", "joytag", unique)

    with merge_store.StateStore(path, "scope", True) as state:
        report = model_report.build_report(app, state, ["1"], policies, policies)

    rows = {row["model"]: row for row in report["models"]}
    assert report["mode"] == "leave-one-model-out"
    assert report["totals"]["scenes"] == 1
    assert report["totals"]["scene_tags"] == 2
    assert report["totals"]["markers"] == 2

    assert rows["wd"]["scene_tags_supported"] == 1
    assert rows["wd"]["scene_tags_sole_source"] == 0
    assert rows["wd"]["scene_tags_multi_model"] == 1
    assert rows["wd"]["marker_components_supported"] == 1
    assert rows["wd"]["marker_components_sole_source"] == 0
    assert rows["wd"]["marker_components_multi_model"] == 1
    assert rows["wd"]["scene_tags_lost"] == 0
    assert rows["wd"]["marker_components_lost"] == 0
    assert rows["wd"]["scenes_changed"] == 0

    assert rows["dtq"]["scene_tags_lost"] == 0
    assert rows["dtq"]["marker_components_lost"] == 0
    assert rows["dtq"]["scenes_changed"] == 0

    assert rows["joytag"]["scene_tags_sole_source"] == 1
    assert rows["joytag"]["marker_components_sole_source"] == 1
    assert rows["joytag"]["scene_tags_lost"] == 1
    assert rows["joytag"]["marker_components_lost"] == 1
    assert rows["joytag"]["scenes_changed"] == 1
    assert rows["joytag"]["changed_scene_pct_of_evidence"] == 100.0

    assert rows["oppai"]["scenes_with_evidence"] == 0
    assert rows["oppai"]["scenes_unchanged"] == 1


def test_ablation_replays_only_models_present_in_scene(tmp_path, monkeypatch):
    path = tmp_path / "state.sqlite3"
    policies = empty_policies()
    app.TEMPORAL_POLICY = threshold_policy()

    with merge_store.StateStore(path, "scope") as state:
        state.put("1", "wd", inference_pair("only"))

    calls = []
    original = merge.evidence

    def evidence(*args, **kwargs):
        calls.append(kwargs.get("selected_sources"))
        return original(*args, **kwargs)

    monkeypatch.setattr(merge, "evidence", evidence)
    with merge_store.StateStore(path, "scope", True) as state:
        report = model_report.build_report(
            app, state, ["1"], policies, policies)

    assert calls[0] is None
    assert calls[1:] == [{"joytag", "dtq", "oppai"}]
    rows = {row["model"]: row for row in report["models"]}
    assert rows["wd"]["scenes_with_evidence"] == 1
    assert all(
        rows[model]["scenes_with_evidence"] == 0
        for model in ("joytag", "dtq", "oppai")
    )


def test_report_file_is_new_json(tmp_path):
    destination = tmp_path / "report.json"
    report = {
        "format": 1,
        "mode": "leave-one-model-out",
        "semantics": {},
        "totals": {"scenes": 0},
        "models": [],
    }

    assert model_report.write_report(report, destination) == destination.resolve()
    assert json.loads(destination.read_text(encoding="utf-8")) == report
    with pytest.raises(FileExistsError):
        model_report.write_report(report, destination)
    with pytest.raises(ValueError, match=r"\.json"):
        model_report.write_report(report, tmp_path / "report.txt")


def test_cli_model_report_uses_saved_evidence_without_models_or_stash(tmp_path, monkeypatch, capsys):
    path = tmp_path / "state.sqlite3"
    scope = merge_store.scope_for(app.STASH_URL)
    with merge_store.StateStore(path, scope) as state:
        state.put("1", "wd", inference_pair("saved"))

    destination = tmp_path / "model-report.json"
    monkeypatch.setattr(app.services, "load_model_settings", lambda *a, **k: pytest.fail("loaded model settings"))
    monkeypatch.setattr(app.services, "create_stash_client", lambda *a, **k: pytest.fail("connected to Stash"))
    preloaded = []
    original_preload = merge_store.StateStore.preload_sources

    def preload_sources(store, scenes=None, **kwargs):
        preloaded.append(True)
        return original_preload(store, scenes, **kwargs)

    monkeypatch.setattr(merge_store.StateStore, "preload_sources", preload_sources)
    monkeypatch.setattr(app, "APPLY_SCENE_TAGS", False)
    monkeypatch.setattr(app, "CREATE_MARKERS", False)
    policy_before = app.TEMPORAL_POLICY

    assert app.main([
        "--state-db", str(path),
        "--model-report", str(destination),
    ]) == 0
    assert app.APPLY_SCENE_TAGS is app.CREATE_MARKERS is False
    assert app.TEMPORAL_POLICY is policy_before

    captured = capsys.readouterr()
    assert preloaded == [True]
    assert "No Stash requests or inference were performed" in captured.out
    assert "Model report" in captured.err
    data = json.loads(destination.read_text(encoding="utf-8"))
    assert data["totals"]["scenes"] == 1
    assert {row["model"] for row in data["models"]} == {"joytag", "wd", "dtq", "oppai"}


def test_cli_model_report_accepts_scene_filter_and_rejects_existing_output(tmp_path):
    path = tmp_path / "state.sqlite3"
    scope = merge_store.scope_for(app.STASH_URL)
    with merge_store.StateStore(path, scope) as state:
        state.put("1", "wd", inference_pair("one"))
        state.put("2", "wd", inference_pair("two"))

    destination = tmp_path / "filtered.json"
    assert app.main([
        "--state-db", str(path),
        "--model-report", str(destination),
        "--scene-id", "2",
    ]) == 0
    assert json.loads(destination.read_text(encoding="utf-8"))["totals"]["scenes"] == 1

    with pytest.raises(SystemExit) as exc:
        app.main(["--state-db", str(path), "--model-report", str(destination)])
    assert exc.value.code == 2


def test_parallel_model_report_matches_serial_output(tmp_path):
    path = tmp_path / "state.sqlite3"
    scope = merge_store.scope_for(app.STASH_URL)
    with merge_store.StateStore(path, scope) as state:
        for scene in range(1, 7):
            shared = inference_pair("shared-" + str(scene))
            shared["scene"]["meta"]["inference_signature"] = "wd-scene-" + str(scene)
            shared["markers"]["meta"]["inference_signature"] = "wd-markers-" + str(scene)
            state.put(str(scene), "wd", shared)

            if scene % 2:
                dtq = inference_pair("shared-" + str(scene))
                dtq["scene"]["meta"]["inference_signature"] = "dtq-scene-" + str(scene)
                dtq["markers"]["meta"]["inference_signature"] = "dtq-markers-" + str(scene)
                state.put(str(scene), "dtq", dtq)
            else:
                unique = inference_pair("unique-" + str(scene))
                unique["scene"]["meta"]["inference_signature"] = "joy-scene-" + str(scene)
                unique["markers"]["meta"]["inference_signature"] = "joy-markers-" + str(scene)
                state.put(str(scene), "joytag", unique)

    serial = tmp_path / "serial.json"
    parallel = tmp_path / "parallel.json"
    common = ["--state-db", str(path)]

    assert app.main([
        *common,
        "--model-report", str(serial),
        "--model-report-workers", "1",
    ]) == 0
    assert app.main([
        *common,
        "--model-report", str(parallel),
        "--model-report-workers", "2",
    ]) == 0

    assert json.loads(parallel.read_text(encoding="utf-8")) == json.loads(
        serial.read_text(encoding="utf-8"))


def test_empty_parallel_report_does_not_open_database(tmp_path):
    path = tmp_path / "missing.sqlite3"
    policies = empty_policies()
    report = model_workers.build_report_parallel(
        app, path, "scope", [], policies, policies,
        workers=2, cache_bytes=0, memory_budget_bytes=1)
    assert report == model_report.build_report(app, None, [], policies, policies)
    assert not path.exists()


@pytest.mark.parametrize("preload", [False, True])
def test_parallel_report_single_scene_fallback_is_read_only(tmp_path, preload):
    path = tmp_path / "state.sqlite3"
    scope = merge_store.scope_for(app.STASH_URL)
    policies = empty_policies()
    with merge_store.StateStore(path, scope) as state:
        state.put("1", "wd", inference_pair("common"))
    before = path.read_bytes()
    with merge_store.StateStore(path, scope, True, allow_cleanup=True) as state:
        serial = model_report.build_report(app, state, ["1"], policies, policies)
    parallel = model_workers.build_report_parallel(
        app, path, scope, ["1"], policies, policies,
        workers=2, cache_bytes=0, memory_budget_bytes=1024**2,
        preload_sources=preload)
    assert parallel == serial
    assert path.read_bytes() == before
