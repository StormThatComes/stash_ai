"""Review priorities must preserve model provenance and cache limitations."""

from stash_ai.reports.review.options import ReviewOptions, REASONS
from copy import deepcopy
import json

import pytest

from stash_ai.inference import sprites
from stash_ai import cli as app, merge
from stash_ai.reports.review import report as tag_review
from stash_ai.reports.review.accumulator import ReviewAccumulator
from stash_ai.models import SOURCES
from stash_ai.temporal import TemporalPolicy
from tests.test_model_report import inference_pair

pytestmark = pytest.mark.usefixtures("reset_temporal")


def policies():
    return {model: frozenset() for model in SOURCES}


def pair(label="common", score=.42, threshold=.4, identity="same"):
    result = inference_pair(label, score=score, identity=identity, marker_threshold=threshold)
    result["scene"]["meta"]["scene_threshold"] = threshold
    result["scene"]["meta"].update(scene_frames_format=1, scene_score_floor=.1,
                                   scene_support_floor=threshold * .85)
    result["scene"]["frames"] = deepcopy(result["markers"]["items"])
    return result


def report(path, scenes=("1",), options=ReviewOptions(), core=None, tags=None):
    with merge.StateStore(path, merge.scope_for(app.STASH_URL), True, allow_cleanup=True) as state:
        return tag_review.build_report(app, state, list(scenes), core or policies(), tags or policies(), options)


def seed(path, scene, model, outputs, replace=None):
    with merge.StateStore(path, merge.scope_for(app.STASH_URL)) as state:
        state.put(scene, model, outputs, replace)


def model_rows(data):
    return {row["model"]: row for row in data["models"]}


def test_accumulator_preserves_global_counts_and_caps_examples():
    accumulator = ReviewAccumulator(27, 1, REASONS)
    rows = {
        ("wd", "shared"): dict(reasons={"long_marker", "weak_or_borderline"}, evidence=[]),
        ("dtq", "shared"): dict(reasons={"long_marker"}, evidence=[]),
        ("wd", "accepted"): dict(reasons=set(), evidence=[]),
    }
    accumulator.record("first", rows)
    accumulator.record("second", rows)
    for scene in range(25):
        accumulator.skip(str(scene), "unusable_evidence", "Incomplete cache")
    data = accumulator.payload()
    assert data["totals"] == dict(
        selected_scenes=27, analyzed_scenes=2, scenes_flagged=2,
        scene_tag_pairs_reviewed=4, scene_tag_pairs_flagged=2,
        unique_tags_reviewed=2, unique_tags_flagged=1, skipped_scenes=25)
    models = model_rows(data)
    assert models["wd"]["scene_tag_pairs_reviewed"] == 4
    assert models["dtq"]["scene_tag_pairs_flagged"] == 2
    assert models["wd"]["reasons"]["long_marker"] == 2
    assert models["wd"]["reasons"]["weak_or_borderline"] == 2
    assert all(row["scene_count"] == 2 and len(row["examples"]) == 1 for row in data["tags"])
    assert all(row["examples"][0]["scene_id"] == "first" for row in data["tags"])
    assert data["skipped_by_reason"] == {"unusable_evidence": 25}
    assert len(data["skipped_scene_examples"]) == 20


def test_counts_deduplicate_models_scenes_history_and_marker_ranges(tmp_path):
    app.TEMPORAL_POLICY = TemporalPolicy(enabled=False)
    path = tmp_path / "state.sqlite3"
    first = pair()
    first["markers"]["items"].extend([
        {"start": 10., "end": 11., "tags": [["common", .42]]},
        {"start": 11., "end": 12., "tags": [["common", .42]]},
    ])
    seed(path, "1", "wd", first)
    seed(path, "1", "wd", first, {"scene": False, "markers": False})
    seed(path, "2", "wd", pair(score=.43))
    seed(path, "1", "dtq", pair(score=.28, threshold=.25))
    seed(path, "1", "joytag", pair(score=.95))

    data = report(path, ("1", "2"), ReviewOptions(examples=1))
    rows = model_rows(data)
    assert data["totals"]["unique_tags_flagged"] == 1
    assert data["totals"]["scene_tag_pairs_flagged"] == 2
    assert rows["wd"]["unique_tags_flagged"] == rows["dtq"]["unique_tags_flagged"] == 1
    assert rows["wd"]["scene_tag_pairs_flagged"] == 2
    assert rows["dtq"]["scene_tag_pairs_flagged"] == 1
    assert rows["joytag"]["unique_tags_reviewed"] == 1
    assert rows["joytag"]["unique_tags_flagged"] == rows["oppai"]["unique_tags_flagged"] == 0
    wd = next(row for row in data["tags"] if row["model"] == "wd")
    assert wd["scene_count"] == 2 and len(wd["examples"]) == 1
    assert wd["reasons"] == {"weak_or_borderline": 2}
    assert {item["kind"] for item in wd["examples"][0]["evidence"]} == {"scene", "marker"}


@pytest.mark.parametrize("peer_case,expected", [
    ("rejected", True), ("other_input", False), ("missing_label", False),
    ("excluded", False), ("accepting_history", False),
])
def test_disagreement_requires_retained_comparable_negative_evidence(tmp_path, peer_case, expected):
    app.TEMPORAL_POLICY = TemporalPolicy(enabled=False)
    path = tmp_path / "state.sqlite3"
    seed(path, "1", "wd", pair(score=.95))
    peer = pair(score=.2, identity="other" if peer_case == "other_input" else "same")
    if peer_case == "missing_label":
        peer["scene"]["items"] = []
        peer["scene"]["frames"] = []
        peer["markers"]["items"] = []
    seed(path, "1", "dtq", peer)
    tags = policies()
    if peer_case == "excluded":
        tags["dtq"] = frozenset({"common"})
    if peer_case == "accepting_history":
        seed(path, "1", "dtq", pair(score=.95), {"scene": False, "markers": False})
    data = report(path, tags=tags)
    assert bool(model_rows(data)["wd"]["reasons"]["model_disagreement"]) == expected
    assert model_rows(data)["dtq"]["unique_tags_flagged"] == 0
    if expected:
        example = data["tags"][0]["examples"][0]["evidence"][0]
        assert example["peer_model"] == "dtq" and example["peer_score"] == .2


@pytest.mark.parametrize("high_hits,expected", [(1, 1), (2, 0)])
def test_isolated_peak_uses_scene_statistics_not_aggregate_count(tmp_path, high_hits, expected):
    path = tmp_path / "state.sqlite3"
    outputs = pair(score=.99)
    outputs["scene"]["items"][0]["statistics"] = dict(
        max_score=.99, high_hits=high_hits, sampled_frames=10, peak_time=1.,
        hits=high_hits, longest_hits=high_hits, sample_fraction=high_hits / 10)
    seed(path, "1", "wd", outputs)
    data = report(path)
    assert model_rows(data)["wd"]["reasons"]["isolated_peak"] == expected


def test_review_uses_effective_sql_thresholds_and_leaves_evidence_untouched(tmp_path):
    app.TEMPORAL_POLICY = TemporalPolicy(enabled=False)
    path = tmp_path / "state.sqlite3"
    seed(path, "1", "wd", pair(score=.82, threshold=.4))
    assert report(path)["totals"]["unique_tags_flagged"] == 0
    with merge.StateStore(path, merge.scope_for(app.STASH_URL)) as state:
        with state.connection:
            for kind in ("scene", "markers"):
                state.set_result_threshold("1", "wd", kind, .8, state.model_records("1", kind, "wd"))
    before = path.read_bytes()
    data = report(path)
    assert path.read_bytes() == before
    assert model_rows(data)["wd"]["unique_tags_flagged"] == 1
    evidence = data["tags"][0]["examples"][0]["evidence"]
    assert {item["kind"] for item in evidence} == {"scene", "marker"}
    assert all(item["threshold"] == .8 for item in evidence)


def test_long_marker_flags_all_contributors_using_the_merged_range(tmp_path):
    app.TEMPORAL_POLICY = TemporalPolicy(enabled=False)
    path = tmp_path / "state.sqlite3"
    for model, start in (("wd", 0.), ("dtq", 2.)):
        outputs = pair(score=.95)
        outputs["markers"]["items"] = [
            {"start": start, "end": start + 2., "tags": [["common", .95]]},
            {"start": start + 2., "end": start + 4., "tags": [["common", .95]]},
        ]
        seed(path, "1", model, outputs)
    data = report(path, options=ReviewOptions(long_marker_seconds=5.))
    assert data["totals"]["scene_tag_pairs_flagged"] == 1
    for model in ("wd", "dtq"):
        assert model_rows(data)[model]["reasons"]["long_marker"] == 1
        row = next(row for row in data["tags"] if row["model"] == model)
        assert row["examples"][0]["evidence"][0]["duration_seconds"] == 6.


def test_exclusions_remove_candidates_and_core_exclusions_preserve_scene_review(tmp_path):
    app.TEMPORAL_POLICY = TemporalPolicy(enabled=False)
    path = tmp_path / "state.sqlite3"
    seed(path, "1", "wd", pair())
    excluded = policies()
    excluded["wd"] = frozenset({"common"})
    assert report(path, tags=excluded)["totals"]["unique_tags_flagged"] == 0
    data = report(path, core=excluded)
    assert {item["kind"] for item in data["tags"][0]["examples"][0]["evidence"]} == {"scene"}


@pytest.mark.parametrize("pending", ["refresh", "publication", "incomplete"])
def test_unusable_scenes_are_skipped_without_negative_votes(tmp_path, pending):
    path = tmp_path / "state.sqlite3"
    outputs = pair()
    if pending == "incomplete":
        outputs["markers"]["meta"]["marker_score_floor"] = .99
    seed(path, "1", "wd", outputs)
    with merge.StateStore(path, merge.scope_for(app.STASH_URL)) as state:
        if pending == "refresh":
            from stash_ai.storage.progress import request_refresh
            request_refresh(state, ["1"], ["wd"], ["scene", "markers"])
        elif pending == "publication":
            state.prepare("1", {"scene": {"owned_ids": [], "contributions": []}})
    before = path.read_bytes()
    data = report(path)
    assert data["totals"]["skipped_scenes"] == 1
    assert data["totals"]["analyzed_scenes"] == data["totals"]["unique_tags_flagged"] == 0
    assert path.read_bytes() == before
    reason = {"refresh": "pending_refresh", "publication": "pending_publication",
              "incomplete": "unusable_evidence"}[pending]
    assert data["skipped_by_reason"] == {reason: 1}


def test_shared_scene_candidates_are_attributed_to_each_actual_supporting_model(tmp_path):
    from tests.helpers.evidence import paired_records
    path = tmp_path / "state.sqlite3"
    # Alternating high samples do not independently qualify either source.
    scenes, markers = paired_records({"wd": [.6, .1, .6, .1],
                                     "dtq": [.1, .6, .1, .6]}, threshold=.6, marker_threshold=.9)
    for model in scenes:
        values = [.6, .1, .6, .1] if model == "wd" else [.1, .6, .1, .6]
        scenes[model][0]["meta"].update(scene_frames_format=1, scene_score_floor=.1,
                                       scene_support_floor=.6 * .85)
        scenes[model][0]["frames"] = [
            dict(start=float(i), end=float(i + 1), tags=[["core", score]])
            for i, score in enumerate(values)]
        for item in scenes[model][0]["items"]:
            item["score"] = .1
            item["statistics"] = dict(max_score=.6, high_hits=2, hits=2,
                                      longest_hits=1, sampled_frames=4, sample_fraction=.5)
        seed(path, "1", model, {"scene": scenes[model][0], "markers": markers[model][0]})
    data = report(path)
    rows = model_rows(data)
    assert rows["wd"]["unique_tags_flagged"] == rows["dtq"]["unique_tags_flagged"] == 1
    assert rows["wd"]["reasons"]["model_disagreement"] == rows["dtq"]["reasons"]["model_disagreement"] == 0


def test_cli_offline_json_and_summary_preserve_database_and_skip_model_loading(tmp_path, monkeypatch, capsys):
    path = tmp_path / "state.sqlite3"
    seed(path, "1", "wd", pair())
    before = path.read_bytes()
    for name in ('load_model_settings', 'load_vocabulary', 'load_tagger', 'create_stash_client'):
        monkeypatch.setattr(app, name, lambda *a, **k: pytest.fail("accessed inference or Stash"))
    monkeypatch.setattr(sprites, 'fetch_cues', lambda *a, **k: pytest.fail("accessed inference or Stash"))
    destination = tmp_path / "review.json"
    assert app.main(["--tag-review-report", str(destination), "--state-db", str(path),
                     "--no-temporal"]) == 0
    assert path.read_bytes() == before
    captured = capsys.readouterr()
    assert "Tag review" in captured.err
    assert "wd" in captured.out and "tags flagged" in captured.out
    assert "No Stash requests or inference" in captured.out
    data = json.loads(destination.read_text())
    assert data["totals"]["unique_tags_flagged"] == 1
    assert set(model_rows(data)) == set(SOURCES)
    with pytest.raises(SystemExit):
        app.main(["--tag-review-report", str(destination), "--state-db", str(path)])


def test_cli_scene_selection_options_missing_scene_and_empty_report(tmp_path):
    path = tmp_path / "state.sqlite3"
    seed(path, "1", "wd", pair())
    seed(path, "2", "wd", pair("other"))
    destination = tmp_path / "filtered.json"
    assert app.main(["--tag-review-report", str(destination), "--state-db", str(path),
                     "--scene-id", "2", "--no-temporal", "--tag-review-margin", "0",
                     "--tag-review-examples", "1"]) == 0
    data = json.loads(destination.read_text())
    assert data["totals"]["selected_scenes"] == 1
    assert data["totals"]["unique_tags_flagged"] == 0
    missing = tmp_path / "missing.json"
    assert app.main(["--tag-review-report", str(missing), "--state-db", str(path),
                     "--scene-id", "3"]) == 1
    assert not missing.exists()
    absent = tmp_path / "absent.sqlite3"
    assert app.main(["--tag-review-report", str(missing), "--state-db", str(absent)]) == 1
    assert not absent.exists()
    empty = tmp_path / "empty.sqlite3"
    with merge.StateStore(empty, merge.scope_for(app.STASH_URL)):
        pass
    assert app.main(["--tag-review-report", str(missing), "--state-db", str(empty)]) == 0
    assert json.loads(missing.read_text())["totals"]["selected_scenes"] == 0


def test_report_on_real_inference_keeps_isolated_peak_without_modifying_cache(pipeline, tmp_path):
    p = pipeline
    p.scores["wd"][:] = .1
    p.scores["wd"][3, 0] = .99
    assert app.main(p.args) == 0
    before = p.path.read_bytes()
    destination = tmp_path / "real-review.json"
    assert app.main(["--tag-review-report", str(destination), "--state-db", str(p.path)]) == 0
    data = json.loads(destination.read_text())
    assert data["totals"]["unique_tags_flagged"] == 1
    assert model_rows(data)["wd"]["reasons"]["isolated_peak"] == 1
    assert p.path.read_bytes() == before


def test_marker_consensus_vote_is_counted_even_without_an_independent_range(tmp_path):
    from tests.helpers.evidence import paired_records
    path = tmp_path / "state.sqlite3"
    scenes, markers = paired_records({"wd": [.72, .72], "dtq": [.72, 0]},
                                     threshold=.9, marker_threshold=.8)
    for model in scenes:
        seed(path, "1", model, {"scene": scenes[model][0], "markers": markers[model][0]})
    data = report(path)
    rows = model_rows(data)
    assert rows["wd"]["unique_tags_flagged"] == rows["dtq"]["unique_tags_flagged"] == 1
    assert data["totals"]["unique_tags_flagged"] == 1
    dtq = next(row for row in data["tags"] if row["model"] == "dtq")
    detail = dtq["examples"][0]["evidence"][0]
    assert detail["support_only"] is True
    assert detail["score"] == .72 and detail["threshold"] == .8


@pytest.mark.parametrize("flags", [
    ["--models", "wd"], ["--restart"], ["--dry-run"], ["--output-mode", "separate"],
    ["--markers-only"], ["--rethreshold-scenes"], ["--tag-review-margin", "nan"],
    ["--tag-review-margin", "-.1"], ["--tag-review-long-marker-seconds", "0"],
    ["--tag-review-long-marker-seconds", "inf"], ["--tag-review-examples", "0"],
])
def test_cli_rejects_unsafe_or_invalid_review_options(tmp_path, flags):
    with pytest.raises(SystemExit) as exc:
        app.main(["--tag-review-report", str(tmp_path / "review.json"), *flags])
    assert exc.value.code == 2


def test_review_modifiers_require_the_review_mode():
    with pytest.raises(SystemExit):
        app.main(["--tag-review-margin", ".1"])
