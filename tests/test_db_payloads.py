"""Reject decodable evidence that cannot be consumed by merge/resume."""
from tests.helpers.evidence import current_record

from tests.helpers.temporal import policy_args

import hashlib
import json
import sqlite3
import struct
import zlib

import pytest

from tests.helpers.runtime import runtime as app
from stash_ai import db as stash_db
import stash_ai.storage.store as merge_store
from tests.test_db import issues, seed
from tests.test_result_thresholds import outputs


FRAME = {"start": 0.0, "end": 1.0, "tags": [["blue_hair", .9]]}


def save_result(path, mode, items, **extra):
    kind = "scene" if mode == "scene" else "markers"
    segment = {"mode": mode, "meta": {"origin": "inference"}, "items": items, **extra}
    with merge_store.StateStore(path, "scope") as state:
        # Build a consistent index, then inject a decodable payload. JSON NaN/Inf
        # cannot be produced by the normal encoder, but must be detected on disk.
        state.put("1", "wd", {kind: outputs()[kind]})
        state.connection.execute("UPDATE results SET data=?",
                                 (zlib.compress(json.dumps([segment]).encode()),))
        state.connection.commit()


@pytest.mark.parametrize("row", [
    {},
    {**FRAME, "start": -1},
    {**FRAME, "end": 0},
    {**FRAME, "end": float("inf")},
    {**FRAME, "start": "0"},
    {**FRAME, "tags": {"blue_hair": .9}},
    {**FRAME, "tags": [["blue_hair"]]},
    {**FRAME, "tags": [[None, .9]]},
    {**FRAME, "tags": [["blue_hair", float("nan")]]},
    {**FRAME, "tags": [["blue_hair", True]]},
    {**FRAME, "tags": [["blue_hair", 1.1]]},
])
def test_check_rejects_unusable_frame_rows(tmp_path, row):
    path = tmp_path / "state.sqlite3"
    save_result(path, "frames", [row])
    before = hashlib.sha256(path.read_bytes()).digest()

    report = stash_db.check_database(path, "scope")

    assert not report["ok"]
    assert issues(report) == {"result_payload_invalid": 1}
    assert hashlib.sha256(path.read_bytes()).digest() == before
    assert "blue_hair" not in stash_db.format_check_report(report)


@pytest.mark.parametrize("mode,items,extra", [
    ("scene", [{}], {}),
    ("scene", [{"tag": "blue_hair", "score": "0.9"}], {}),
    ("scene", [{"tag": "blue_hair", "score": .9, "statistics": {}}], {}),
    ("scene", [{"tag": "blue_hair", "score": .9}], {"frames": [{}]}),
    ("scene", [{"tag": "another_tag", "score": .9}], {"frames": [FRAME]}),
    ("spans", [{}], {}),
    ("spans", [{"tag": "blue_hair", "score": .9, "start": 2, "end": 1}], {}),
    ("spans", [{"tag": "blue_hair", "score": .9, "start": 0, "end": 1,
                "extra_tags": "not-a-list"}], {}),
])
def test_check_rejects_unusable_scene_and_span_rows(tmp_path, mode, items, extra):
    path = tmp_path / "state.sqlite3"
    save_result(path, mode, items, **extra)
    report = stash_db.check_database(path, "scope")
    assert not report["ok"]
    assert issues(report) == {"result_payload_invalid": 1}


@pytest.mark.parametrize("score", [float("nan"), float("inf"), -float("inf"), -.01, 1.01])
def test_check_rejects_invalid_float32_probabilities(tmp_path, score):
    path = tmp_path / "state.sqlite3"
    seed(path, "scope")
    with sqlite3.connect(path) as db:
        db.execute("INSERT INTO checkpoints VALUES (?,?,?,?,?,?,?,?)",
                   ("scope", "cp", "wd", "sig", 1, 2, 1,
                    zlib.compress(struct.pack("<ff", .5, score))))
    report = stash_db.check_database(path, "scope")
    assert not report["ok"]
    assert issues(report) == {"checkpoint_scores": 1}


def test_check_reports_fractional_checkpoint_dimensions_without_crashing(tmp_path):
    path = tmp_path / "state.sqlite3"
    seed(path, "scope")
    with sqlite3.connect(path) as db:
        # SQLite's INTEGER affinity still permits REAL values. This byte length
        # passes the old rows * 4 divisibility test, but is not a float32 array.
        db.execute("INSERT INTO checkpoints VALUES (?,?,?,?,?,?,?,?)",
                   ("scope", "cp", "wd", "sig", 2, 3, 1.25, zlib.compress(b"12345")))
    report = stash_db.check_database(path, "scope")
    assert not report["ok"]
    assert issues(report) == {"checkpoint_dimensions": 1, "checkpoint_scores": 1}


@pytest.mark.parametrize("scores", [[], [0.0, .5, 1.0]])
def test_check_accepts_empty_and_boundary_checkpoint_scores(tmp_path, scores):
    path = tmp_path / "state.sqlite3"
    seed(path, "scope")
    with sqlite3.connect(path) as db:
        db.execute("INSERT INTO checkpoints VALUES (?,?,?,?,?,?,?,?)",
                   ("scope", "cp", "wd", "sig", 1, 2, int(bool(scores)),
                    zlib.compress(struct.pack(f"<{len(scores)}f", *scores))))
    report = stash_db.check_database(path, "scope")
    assert report["ok"], report["errors"]


@pytest.mark.parametrize("row", [{}, {**FRAME, "scene_tags": [["blue_hair", "bad"]]}])
def test_check_rejects_invalid_checkpoint_frames(tmp_path, row):
    path = tmp_path / "state.sqlite3"
    seed(path, "scope")
    with sqlite3.connect(path) as db:
        db.execute("INSERT INTO checkpoints VALUES (?,?,?,?,?,?,?,?)",
                   ("scope", "cp", "wd", "sig", 1, 2, 1,
                    zlib.compress(struct.pack("<f", .9))))
        db.execute("INSERT INTO checkpoint_frames VALUES (?,?,?,?,?)",
                   ("scope", "cp", "wd", 1, zlib.compress(json.dumps([row]).encode())))
    report = stash_db.check_database(path, "scope")
    assert not report["ok"]
    assert issues(report) == {"state_payload_invalid": 1}


def test_check_preserves_legacy_and_sparse_formats(tmp_path):
    path = tmp_path / "state.sqlite3"
    seed(path, "scope")  # Old inference without retention metadata/statistics.
    with merge_store.StateStore(path, "scope") as state:
        state.put("legacy", "wd", {
            "scene": {"mode": "scene", "meta": {"origin": "legacy"},
                      "items": [{"tag": "blue_hair", "score": None}]},
            "markers": {"mode": "spans", "meta": {"origin": "legacy"},
                        "items": [{"tag": "blue_hair", "score": None, "hits": None,
                                   "start": 0, "end": end, "extra_tags": [], "extra_ids": ["42"]}
                                  for end in (None, 0, 1)]},
        })
        state.put("empty", "wd", {
            "scene": current_record({"mode": "scene", "meta": {"origin": "inference"}, "items": []}),
            "markers": current_record({"mode": "frames", "meta": {"origin": "inference"},
                        "items": [{**FRAME, "tags": []}]}),
        })
    report = stash_db.check_database(path, "scope")
    assert report["ok"], report["errors"]


@pytest.mark.parametrize("scene_only", [False, True])
@pytest.mark.parametrize("temporal", [False, True])
def test_check_accepts_real_checkpoint_and_completed_inference(pipeline, scene_only, temporal):
    p = pipeline
    args = [*p.args, *(["--scene-tags-only"] if scene_only else []),
            *policy_args(p.path.parent, temporal)]
    scope = merge_store.scope_for(app.STASH_URL)
    p.fail["wd"] = 4
    with pytest.raises(KeyboardInterrupt):
        app.main(args)
    report = stash_db.check_database(p.path, scope)
    assert report["ok"], report["errors"]
    p.fail.clear()
    assert app.main(args) == 0
    report = stash_db.check_database(p.path, scope)
    assert report["ok"], report["errors"]


def test_cli_rejects_unusable_payload_offline(tmp_path, monkeypatch):
    path = tmp_path / "state.sqlite3"
    save_result(path, "frames", [{}])
    for name in ("create_stash_client", "load_model_settings", "load_tagger"):
        monkeypatch.setattr(app.services, name, lambda *a, **kw: pytest.fail("db-check must stay offline"))
    assert app.main(["--state-db", str(path), "--db-check"]) == 1
