"""Offline model utility and leave-one-model-out ablation reporting."""

from dataclasses import replace
import json

import pytest

from stash_ai import cli as app
from stash_ai import merge
from stash_ai import model_report
from stash_ai.temporal import TemporalPolicy


pytestmark = pytest.mark.usefixtures("reset_temporal")


def empty_policies():
    from stash_ai.taggers import SOURCES
    return {source: frozenset() for source in SOURCES}


def inference_pair(label, *, score=.9, identity="same", marker_threshold=.8):
    meta = {
        "origin": "inference",
        "inference_signature": "sig-" + label,
        "frame_input_signature": identity,
        "scene_threshold": .4,
        "marker_threshold": marker_threshold,
        "marker_score_floor": marker_threshold * .85,
        "analysis_format": 3,
    }
    return {
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
    }


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
    with merge.StateStore(path, "scope") as state:
        for source in ("wd", "dtq"):
            outputs = inference_pair("consensus_only", score=weak, identity="identical-input")
            outputs["scene"]["items"] = []
            outputs["scene"]["meta"]["inference_signature"] = source + "-scene"
            outputs["markers"]["meta"]["inference_signature"] = source + "-markers"
            state.put("1", source, outputs)

    with merge.StateStore(path, "scope", True) as state:
        _, full = merge.evidence(app, state, "1", policies, policies)
        _, without_dtq = merge.evidence(
            app, state, "1", policies, policies, selected_sources={"wd"})

    assert merge.union_intervals(full)
    assert merge.union_intervals(without_dtq) == []


def test_ablation_report_distinguishes_redundant_and_unique_models(tmp_path):
    path = tmp_path / "state.sqlite3"
    policies = empty_policies()
    app.TEMPORAL_POLICY = TemporalPolicy(enabled=False)

    with merge.StateStore(path, "scope") as state:
        state.put("1", "wd", inference_pair("shared"))
        dtq = inference_pair("shared")
        dtq["scene"]["meta"]["inference_signature"] = "dtq-scene"
        dtq["markers"]["meta"]["inference_signature"] = "dtq-markers"
        state.put("1", "dtq", dtq)

        unique = inference_pair("unique")
        unique["scene"]["meta"]["inference_signature"] = "joy-scene"
        unique["markers"]["meta"]["inference_signature"] = "joy-markers"
        state.put("1", "joytag", unique)

    with merge.StateStore(path, "scope", True) as state:
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
    app.TEMPORAL_POLICY = TemporalPolicy(enabled=False)

    with merge.StateStore(path, "scope") as state:
        state.put("1", "wd", inference_pair("only"))

    calls = []
    original = merge.evidence

    def evidence(*args, **kwargs):
        calls.append(kwargs.get("selected_sources"))
        return original(*args, **kwargs)

    monkeypatch.setattr(merge, "evidence", evidence)
    with merge.StateStore(path, "scope", True) as state:
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
    scope = merge.scope_for(app.STASH_URL)
    with merge.StateStore(path, scope) as state:
        state.put("1", "wd", inference_pair("saved"))

    destination = tmp_path / "model-report.json"
    monkeypatch.setattr(app, "load_model_settings", lambda *a, **k: pytest.fail("loaded model settings"))
    monkeypatch.setattr(app, "create_stash_client", lambda *a, **k: pytest.fail("connected to Stash"))

    assert app.main([
        "--state-db", str(path),
        "--model-report", str(destination),
    ]) == 0

    output = capsys.readouterr().out
    assert "No Stash requests or inference were performed" in output
    data = json.loads(destination.read_text(encoding="utf-8"))
    assert data["totals"]["scenes"] == 1
    assert {row["model"] for row in data["models"]} == {"joytag", "wd", "dtq", "oppai"}


def test_cli_model_report_accepts_scene_filter_and_rejects_existing_output(tmp_path):
    path = tmp_path / "state.sqlite3"
    scope = merge.scope_for(app.STASH_URL)
    with merge.StateStore(path, scope) as state:
        state.put("1", "wd", inference_pair("one"))
        state.put("2", "wd", inference_pair("two"))

    destination = tmp_path / "filtered.json"
    assert app.main([
        "--state-db", str(path),
        "--model-report", str(destination),
        "--scene-id", "2",
        "--no-temporal",
    ]) == 0
    assert json.loads(destination.read_text(encoding="utf-8"))["totals"]["scenes"] == 1

    with pytest.raises(SystemExit) as exc:
        app.main(["--state-db", str(path), "--model-report", str(destination)])
    assert exc.value.code == 2
