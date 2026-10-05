"""State database reporting and safe compaction."""
from tests.helpers.evidence import inference_record
from stash_ai.storage.snapshots import encode as encode_snapshot

import hashlib
import sqlite3
import zlib

import pytest

from tests.helpers.runtime import runtime as app
from stash_ai.db.checks import check_database, format_check_report
from stash_ai.db.compaction import compact_database
from stash_ai.db.inspection import format_report, inspect_database
import stash_ai.storage.store as merge_store
from stash_ai.storage.codec import encode


def seed(path, scope):
    with merge_store.StateStore(path, scope) as state:
        state.put("1", "wd", {kind: inference_record(**record) for kind, record in {
            "scene": {
                "mode": "scene",
                "meta": {"origin": "inference", "inference_signature": "scene-sig"},
                "items": [{"tag": "blue_hair", "score": .9}],
            },
            "markers": {
                "mode": "frames",
                "meta": {"origin": "inference", "inference_signature": "marker-sig"},
                "items": [{"start": 0.0, "end": 1.0, "tags": [["blue_hair", .9]]}],
            },
        }.items()})


def test_report_is_read_only_and_privacy_safe(tmp_path):
    path = tmp_path / "state.sqlite3"
    scope = "scope-a"
    seed(path, scope)
    before = hashlib.sha256(path.read_bytes()).digest()

    report = inspect_database(path, scope)

    assert hashlib.sha256(path.read_bytes()).digest() == before
    assert report["quick_check"] == "ok"
    assert report["scenes"] == 1
    assert {(row["model"], row["kind"], row["scenes"]) for row in report["results"]} == {
        ("wd", "markers", 1),
        ("wd", "scene", 1),
    }
    assert report["pending_publications"] == 0
    text = format_report(report)
    assert "blue_hair" not in text
    assert "scene-sig" not in text
    assert "marker-sig" not in text


def test_compact_reclaims_free_pages_and_preserves_state(tmp_path):
    path = tmp_path / "state.sqlite3"
    scope = "scope-a"
    seed(path, scope)
    with sqlite3.connect(path) as db:
        db.execute("CREATE TABLE scratch(data BLOB)")
        db.execute("INSERT INTO scratch VALUES (randomblob(4000000))")
        db.execute("DELETE FROM scratch")

    before = inspect_database(path, scope)
    assert before["free_page_bytes"] > 0
    with merge_store.StateStore(path, scope, True) as state:
        expected = state.records("1", "scene"), state.records("1", "markers")

    old, after = compact_database(path, scope)

    assert old["file_bytes"] == before["file_bytes"]
    assert after["quick_check"] == "ok"
    assert after["file_bytes"] < before["file_bytes"]
    assert after["free_page_bytes"] <= before["free_page_bytes"]
    with merge_store.StateStore(path, scope, True) as state:
        assert (state.records("1", "scene"), state.records("1", "markers")) == expected


def test_compact_refuses_pending_publication_or_cleanup(tmp_path):
    path = tmp_path / "state.sqlite3"
    scope = "scope-a"
    seed(path, scope)
    with merge_store.StateStore(path, scope) as state:
        state.prepare("1", {"scene": {"owned_ids": [], "contributions": []}})

    with pytest.raises(RuntimeError, match="Pending publication"):
        compact_database(path, scope)

    with merge_store.StateStore(path, scope, allow_cleanup=True) as state:
        with state.connection:
            state.connection.execute(
                "UPDATE snapshots SET pending=NULL WHERE scope=?", (scope,))
            state.connection.execute(
                "INSERT INTO cleanup_state VALUES (?,?,?,?) "
                "ON CONFLICT(scope) DO UPDATE SET pending=excluded.pending",
                (scope, 1, state.pack({}), "pending-cleanup"),
            )

    with pytest.raises(RuntimeError, match="cleanup plan"):
        compact_database(path, scope)


def test_compact_refuses_pending_plan_in_another_scope(tmp_path):
    path = tmp_path / "state.sqlite3"
    seed(path, "scope-a")
    with merge_store.StateStore(path, "scope-b") as state:
        state.prepare("9", {"scene": {"owned_ids": [], "contributions": []}})

    report = inspect_database(path, "scope-a")
    assert report["scope_pending_publications"] == 0
    assert report["pending_publications"] == 1
    with pytest.raises(RuntimeError, match="Pending publication"):
        compact_database(path, "scope-a")


def test_compact_dry_run_does_not_change_database(tmp_path):
    path = tmp_path / "state.sqlite3"
    scope = "scope-a"
    seed(path, scope)
    with sqlite3.connect(path) as db:
        db.execute("CREATE TABLE scratch(data BLOB)")
        db.execute("INSERT INTO scratch VALUES (randomblob(1000000))")
        db.execute("DELETE FROM scratch")
    before = path.read_bytes()

    first, second = compact_database(path, scope, dry_run=True)

    assert path.read_bytes() == before
    assert first == second
    assert first["free_page_bytes"] > 0


def test_db_cli_does_not_load_models_or_stash(tmp_path, monkeypatch, capsys):
    path = tmp_path / "state.sqlite3"
    scope = merge_store.scope_for(app.STASH_URL)
    seed(path, scope)
    monkeypatch.setattr(app.services, "load_model_settings", lambda *a, **k: pytest.fail("loaded model config"))
    monkeypatch.setattr(app.services, "create_stash_client", lambda *a, **k: pytest.fail("connected to Stash"))

    assert app.main(["--state-db", str(path), "--db-report"]) == 0
    output = capsys.readouterr().out
    assert "Health: ok" in output
    assert "wd" in output

    before = path.read_bytes()
    assert app.main(["--state-db", str(path), "--db-compact", "--dry-run"]) == 0
    assert path.read_bytes() == before
    assert "VACUUM was not executed" in capsys.readouterr().out



def issues(report, field="errors"):
    return {item["code"]: item["count"] for item in report[field]}


def test_deep_check_is_read_only_and_accepts_consistent_state(tmp_path):
    path = tmp_path / "state.sqlite3"
    scope = "scope-a"
    seed(path, scope)
    before = hashlib.sha256(path.read_bytes()).digest()

    report = check_database(path, scope)

    assert hashlib.sha256(path.read_bytes()).digest() == before
    assert report["ok"]
    assert report["errors"] == []
    assert report["results_checked"] == 2
    assert report["integrity_check_ok"]
    assert "No logical consistency problems found." in format_check_report(report)


def test_deep_check_detects_result_index_relationship_and_content_errors(tmp_path):
    path = tmp_path / "state.sqlite3"
    scope = "scope-a"
    seed(path, scope)

    with sqlite3.connect(path) as db:
        db.execute(
            "UPDATE result_index SET populated=0 "
            "WHERE scope=? AND scene='1' AND model='wd' AND kind='scene'",
            (scope,),
        )
        db.execute(
            "DELETE FROM result_index "
            "WHERE scope=? AND scene='1' AND model='wd' AND kind='markers'",
            (scope,),
        )
        db.execute(
            "INSERT INTO result_index VALUES (?,?,?,?,?,?)",
            (scope, "orphan", "dtq", "scene", 0, None),
        )

    report = check_database(path, scope)
    found = issues(report)

    assert not report["ok"]
    assert found["result_index_mismatch"] == 1
    assert found["result_missing_index"] == 1
    assert found["result_index_orphan"] == 1


def test_deep_check_detects_corrupt_state_blobs_and_unknown_domains(tmp_path):
    path = tmp_path / "state.sqlite3"
    scope = "scope-a"
    seed(path, scope)

    with sqlite3.connect(path) as db:
        db.execute(
            "UPDATE results SET data=? "
            "WHERE scope=? AND scene='1' AND model='wd' AND kind='scene'",
            (b"not-zlib", scope),
        )
        db.execute(
            "INSERT INTO snapshots VALUES (?,?,?,?,?)",
            (scope, "1", "scene", b"not-zlib", None),
        )
        db.execute(
            "INSERT INTO refresh_requests VALUES (?,?,?,?)",
            (scope, "2", "unknown-model", "scene"),
        )
        db.execute(
            "INSERT INTO refresh_requests VALUES (?,?,?,?)",
            (scope, "2", "wd", "unknown-kind"),
        )

    report = check_database(path, scope)
    found = issues(report)

    assert not report["ok"]
    assert found["result_payload_invalid"] == 1
    assert found["state_payload_invalid"] == 1
    assert found["invalid_model"] == 1
    assert found["invalid_kind"] == 1


def test_deep_check_detects_checkpoint_parent_progress_and_score_corruption(tmp_path):
    path = tmp_path / "state.sqlite3"
    scope = "scope-a"
    seed(path, scope)

    with sqlite3.connect(path) as db:
        db.execute(
            "INSERT INTO checkpoints VALUES (?,?,?,?,?,?,?,?)",
            (scope, "cp", "wd", "sig", 5, 4, 6, zlib.compress(b"bad")),
        )
        db.execute(
            "INSERT INTO checkpoint_frames VALUES (?,?,?,?,?)",
            (scope, "cp", "wd", 9, encode([])),
        )
        db.execute(
            "INSERT INTO checkpoint_frames VALUES (?,?,?,?,?)",
            (scope, "orphan", "dtq", 1, encode([])),
        )
        db.execute(
            "INSERT INTO checkpoint_analysis VALUES (?,?,?,?)",
            (scope, "orphan", "dtq", encode({})),
        )

    report = check_database(path, scope)
    found = issues(report)

    assert not report["ok"]
    assert found["checkpoint_dimensions"] == 1
    assert found["checkpoint_scores"] == 1
    assert found["checkpoint_frame_progress"] == 1
    assert found["checkpoint_orphan_frames"] == 1
    assert found["checkpoint_orphan_analysis"] == 1


def test_deep_check_validates_cleanup_and_recovery_generations(tmp_path):
    path = tmp_path / "state.sqlite3"
    scope = "scope-a"
    seed(path, scope)

    with sqlite3.connect(path) as db:
        db.execute(
            "INSERT INTO cleanup_state VALUES (?,?,?,?)",
            (scope, 1, encode({"global": {"some-tag": 2}}), None),
        )
        db.execute(
            "INSERT INTO recovery_cleanups VALUES (?,?,?)",
            (scope, 2, '["scene","invalid-kind"]'),
        )

    report = check_database(path, scope)
    found = issues(report)

    assert not report["ok"]
    assert found["cleanup_state_invalid"] == 1
    assert found["recovery_cleanup_invalid"] == 1


def test_deep_check_relation_warnings_do_not_mark_database_failed(tmp_path):
    path = tmp_path / "state.sqlite3"
    scope = "scope-a"
    seed(path, scope)

    with sqlite3.connect(path) as db:
        db.execute("CREATE TABLE legacy_imports(scope TEXT,scene TEXT,kind TEXT,data BLOB)")
        db.execute(
            "INSERT INTO snapshots VALUES (?,?,?,?,?)",
            (scope, "unused", "scene", None, None),
        )
        db.execute(
            "INSERT INTO legacy_imports VALUES (?,?,?,?)",
            (scope, "unused", "markers", encode([])),
        )
        db.execute(
            "INSERT INTO recovery_publications VALUES (?,?,?,?,?)",
            (scope, "recovery-only", "scene", "op", encode_snapshot({"contributions": []})),
        )

    report = check_database(path, scope)
    found = issues(report, "warnings")

    assert report["ok"]
    assert found["empty_snapshot_without_recovery"] == 1
    assert found["legacy_import_without_result"] == 1
    assert found["recovery_publication_without_snapshot"] == 1


def test_db_check_cli_is_offline_and_returns_nonzero_for_inconsistency(tmp_path, monkeypatch, capsys):
    path = tmp_path / "state.sqlite3"
    scope = merge_store.scope_for(app.STASH_URL)
    seed(path, scope)
    monkeypatch.setattr(app.services, "load_model_settings", lambda *a, **k: pytest.fail("loaded model config"))
    monkeypatch.setattr(app.services, "create_stash_client", lambda *a, **k: pytest.fail("connected to Stash"))

    assert app.main(["--state-db", str(path), "--db-check"]) == 0
    assert "State DB consistency: ok" in capsys.readouterr().out

    with sqlite3.connect(path) as db:
        db.execute(
            "DELETE FROM result_index WHERE scope=? AND scene='1' AND model='wd' AND kind='scene'",
            (scope,),
        )

    assert app.main(["--state-db", str(path), "--db-check"]) == 1
    output = capsys.readouterr().out
    assert "State DB consistency: FAILED" in output
    assert "result_missing_index" in output

