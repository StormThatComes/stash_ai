"""Privacy and usefulness guarantees for anonymous developer diagnostics."""
from tests.helpers.evidence import inference_record

from stash_ai.reports.developer.distribution import Distribution

import hashlib
import json
import sqlite3

import pytest

from tests.helpers.runtime import runtime as app
from stash_ai.reports.developer import report as diagnostics
import stash_ai.storage.store as merge_store
from stash_ai.storage.codec import encode


SECRETS = (
    "scope-secret-alpha",
    "scope-secret-beta",
    "scene-secret-123",
    "scene-secret-999",
    "private_tag_name",
    "another_private_tag",
    "secret-signature-value",
    "private_customer_name",
    "custom-secret-model",
)


def segment(tag="private_tag_name"):
    return inference_record(**{
        "mode": "scene",
        "meta": {
            "origin": "inference",
            "inference_signature": "secret-signature-value",
            "frame_input_signature": "also-secret",
            "captured_at": "2026-01-02T03:04:05Z",
        },
        "items": [{"tag": tag, "score": .91}],
        "frames": [
            {"start": 12.5, "end": 13.5, "tags": [[tag, .91], ["different_private_tag", .72]]},
            {"start": 13.5, "end": 14.5, "tags": [[tag, .83]]},
        ],
    })


def seed(path):
    with merge_store.StateStore(path, "scope-secret-alpha") as state:
        state.put("scene-secret-123", "wd", {
            "scene": segment(),
            "markers": inference_record(**{
                "mode": "frames",
                "meta": {"origin": "inference", "inference_signature": "secret-signature-value"},
                "items": [
                    {"start": 12.5, "end": 13.5,
                     "tags": [["private_tag_name", .91], ["another_private_tag", .72]]},
                    {"start": 13.5, "end": 14.5, "tags": [["private_tag_name", .83]]},
                ],
            }),
        })
    with merge_store.StateStore(path, "scope-secret-beta") as state:
        state.put("scene-secret-999", "joytag", {"scene": segment("another_private_tag")})

    # Simulate unexpected/custom state. Names must never be copied into the report.
    with sqlite3.connect(path) as db:
        db.execute("CREATE TABLE private_customer_name(secret TEXT)")
        db.execute("INSERT INTO private_customer_name VALUES ('do-not-export')")
        custom = segment()
        custom["meta"]["analysis_format"] = 77777
        custom["meta"]["scene_frames_format"] = 88888
        db.execute(
            "INSERT INTO results(scope,scene,model,kind,data) VALUES (?,?,?,?,?)",
            ("scope-secret-alpha", "secret-extra-scene", "custom-secret-model", "custom-kind",
             encode(custom)),
        )


def serialized(report):
    return json.dumps(report, ensure_ascii=False, sort_keys=True)


def test_distribution_memory_is_bounded_for_many_distinct_values():
    distribution = Distribution()
    for value in range(100_000):
        distribution.add(value)

    summary = distribution.summary(total=True)
    assert len(distribution.buckets) <= distribution.MAX_BUCKETS
    assert summary["count"] == 100_000
    assert summary["min"] == 0
    assert summary["max"] == 99_999
    assert summary["total"] == 4_999_950_000
    assert summary["mean"] == 49_999.5
    assert summary["min"] <= summary["p50"] <= summary["p95"] <= summary["p99"] <= summary["max"]


def test_distribution_keeps_small_percentiles_exact_and_large_singletons_exact():
    distribution = Distribution()
    for value in (0, 1, 2, 2, 3, 10_000, 10_000):
        distribution.add(value)

    assert distribution.summary()["p50"] == 2
    single = Distribution()
    single.add(10_000)
    assert single.summary()["p50"] == single.summary()["p95"] == single.summary()["p99"] == 10_000


def test_report_is_read_only_content_free_and_cross_scope(tmp_path):
    path = tmp_path / "descriptive-private-library-name.sqlite3"
    seed(path)
    before = hashlib.sha256(path.read_bytes()).digest()

    report = diagnostics.build_report(path)
    text = serialized(report)

    assert hashlib.sha256(path.read_bytes()).digest() == before
    assert report["format"] == 1
    assert report["database"]["quick_check_ok"]
    assert report["database"]["scope_count"] == 2
    assert report["database"]["scene_count"] == 3
    assert report["database"]["unknown_table_count"] == 1
    assert "descriptive-private-library-name" not in text
    for secret in SECRETS:
        assert secret not in text
    assert "do-not-export" not in text

    privacy = report["privacy"]
    assert not privacy["automatic_upload"]
    assert not privacy["contains_scene_ids"]
    assert not privacy["contains_scope_values"]
    assert not privacy["contains_tag_names"]
    assert not privacy["contains_scores"]
    assert privacy["aggregate_counts_and_sizes_describe_the_installation"]

    groups = {(row["model"], row["kind"]): row for row in report["evidence"]}
    assert ("wd", "scene") in groups
    assert ("wd", "markers") in groups
    assert ("other", "other") in groups
    assert groups["other", "other"]["analysis_formats"] == {"other": 1}
    assert groups["other", "other"]["scene_frame_formats"] == {"other": 1}

    markers = groups["wd", "markers"]
    assert markers["rows"] == 1
    assert markers["frame_detections_per_frame"]["count"] == 2
    assert markers["frame_detections_per_frame"]["max"] == 2
    assert markers["compressed_bytes"]["total"] > 0
    assert markers["json_bytes"]["total"] > markers["compressed_bytes"]["total"]

    scenes = groups["wd", "scene"]
    assert scenes["scene_observation_frames_per_record"]["p50"] == 2
    assert scenes["scene_observations_per_frame"]["count"] == 2
    assert scenes["scene_observations_per_frame"]["max"] == 2


def test_report_file_is_new_json_and_never_overwrites(tmp_path):
    path = tmp_path / "state.sqlite3"
    seed(path)
    destination = tmp_path / "share" / "diagnostics.json"

    created, report = diagnostics.write_report(path, destination)

    assert created == destination.resolve()
    loaded = json.loads(destination.read_text(encoding="utf-8"))
    assert loaded == report
    for secret in SECRETS:
        assert secret not in destination.read_text(encoding="utf-8")

    original = destination.read_bytes()
    with pytest.raises(FileExistsError):
        diagnostics.write_report(path, destination)
    assert destination.read_bytes() == original

    with pytest.raises(ValueError, match=r"\.json"):
        diagnostics.write_report(path, tmp_path / "diagnostics.txt")
    with pytest.raises(ValueError, match="must not replace"):
        diagnostics.write_report(path, path)


def test_cli_does_not_load_models_connect_to_stash_or_use_stash_url(tmp_path, monkeypatch, capsys):
    path = tmp_path / "state.sqlite3"
    seed(path)
    destination = tmp_path / "anonymous.json"
    monkeypatch.setattr(app.services, "load_model_settings", lambda *a, **k: pytest.fail("loaded model config"))
    monkeypatch.setattr(app.services, "create_stash_client", lambda *a, **k: pytest.fail("connected to Stash"))
    monkeypatch.setattr(app, "STASH_URL", "https://private-host.example/sensitive-path")

    assert app.main([
        "--state-db", str(path),
        "--anonymous-diagnostics", str(destination),
    ]) == 0

    text = destination.read_text(encoding="utf-8")
    assert "private-host" not in text
    assert "sensitive-path" not in text
    assert "no upload was performed" in capsys.readouterr().out


def test_cli_rejects_dry_run_because_report_itself_is_the_only_write(tmp_path):
    path = tmp_path / "state.sqlite3"
    seed(path)

    with pytest.raises(SystemExit) as exc:
        app.main([
            "--state-db", str(path),
            "--anonymous-diagnostics", str(tmp_path / "report.json"),
            "--dry-run",
        ])
    assert exc.value.code == 2

