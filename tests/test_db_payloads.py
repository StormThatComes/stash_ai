"""Reject decodable evidence that cannot be consumed by merge/resume."""
from tests.helpers.evidence import inference_record

from tests.helpers.temporal import policy_args

import hashlib
import json
import sqlite3
import struct
import zlib

import pytest

from tests.helpers.runtime import runtime as app
from stash_ai.db.checks import check_database, format_check_report
import stash_ai.storage.store as merge_store
from tests.test_db import issues, seed
from tests.test_result_thresholds import outputs


FRAME = {"start": 0.0, "end": 1.0, "tags": [["blue_hair", .9]]}


def save_result(path, mode, items, **extra):
    kind = "scene" if mode == "scene" else "markers"
    segment = {**outputs()[kind], "mode": mode, "items": items, **extra}
    with merge_store.StateStore(path, "scope") as state:
        # Build a consistent index, then inject a decodable payload. JSON NaN/Inf
        # cannot be produced by the normal encoder, but must be detected on disk.
        state.put("1", "wd", {kind: outputs()[kind]})
        state.connection.execute("UPDATE results SET data=?",
                                 (zlib.compress(json.dumps(segment).encode()),))
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

    report = check_database(path, "scope")

    assert not report["ok"]
    assert issues(report) == {"result_payload_invalid": 1}
    assert hashlib.sha256(path.read_bytes()).digest() == before
    assert "blue_hair" not in format_check_report(report)


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
    report = check_database(path, "scope")
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
    report = check_database(path, "scope")
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
    report = check_database(path, "scope")
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
    report = check_database(path, "scope")
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
    report = check_database(path, "scope")
    assert not report["ok"]
    assert issues(report) == {"state_payload_invalid": 1}


def test_check_accepts_current_and_sparse_formats(tmp_path):
    path = tmp_path / "state.sqlite3"
    seed(path, "scope")
    with merge_store.StateStore(path, "scope") as state:
        state.put("raw", "wd", {
            "scene": inference_record(mode="scene", items=[dict(tag="blue_hair", score=.9)]),
            "markers": inference_record(mode="frames", items=[
                dict(start=0, end=1, tags=[("blue_hair", .9)]),
                dict(start=1, end=2, tags=[("blue_hair", .8)])]),
        })
        state.put("empty", "wd", {
            "scene": inference_record(**{"mode": "scene", "meta": {"origin": "inference"}, "items": []}),
            "markers": inference_record(**{"mode": "frames", "meta": {"origin": "inference"},
                        "items": [{**FRAME, "tags": []}]}),
        })
    report = check_database(path, "scope")
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
    report = check_database(p.path, scope)
    assert report["ok"], report["errors"]
    p.fail.clear()
    assert app.main(args) == 0
    report = check_database(p.path, scope)
    assert report["ok"], report["errors"]


def test_cli_rejects_unusable_payload_offline(tmp_path, monkeypatch):
    path = tmp_path / "state.sqlite3"
    save_result(path, "frames", [{}])
    for name in ("create_stash_client", "load_model_settings", "load_tagger"):
        monkeypatch.setattr(app.services, name, lambda *a, **kw: pytest.fail("db-check must stay offline"))
    assert app.main(["--state-db", str(path), "--db-check"]) == 1


@pytest.mark.parametrize('mode,items', [
    ('scene', [{'tag': 'blue_hair', 'score': .9}]),
    ('scene', [{'tag': 'blue_hair', 'score': .9, 'statistics': {
        'hits': 1, 'high_hits': 1, 'longest_hits': 1, 'max_score': .9, 'sample_fraction': 1}}]),
    ('frames', [FRAME]),
    ('spans', [{'tag': 'blue_hair', 'score': .9, 'start': 0, 'end': 1}]),
    ('spans', [{'tag': 'blue_hair', 'score': .9, 'start': 0, 'end': None}]),
])
def test_retired_inference_shapes_are_rejected_in_current_archives(tmp_path, mode, items):
    from stash_ai.evidence.identity import marker_segment_signature
    from stash_ai.storage.codec import encode
    path = tmp_path / 'state.sqlite3'
    seed(path, 'scope')
    kind = 'scene' if mode == 'scene' else 'markers'
    record = {'mode': mode, 'meta': {'origin': 'inference', 'analysis_format': 1}, 'items': items}
    with sqlite3.connect(path) as connection:
        connection.execute('INSERT INTO inference_archive VALUES (?,?,?,?,?,?)',
            ('scope', 'old', 'wd', kind, marker_segment_signature(record), encode(record)))
    before = path.read_bytes()
    report = check_database(path, 'scope')
    assert issues(report) == {'state_payload_invalid': 1}
    assert path.read_bytes() == before
    with sqlite3.connect(path) as connection:
        connection.execute('UPDATE results SET data=? WHERE kind=?', (encode(record), kind))
    assert issues(check_database(path, 'scope')) == {'result_payload_invalid': 1, 'state_payload_invalid': 1}


@pytest.mark.parametrize('record', [
    [{'mode': 'scene', 'meta': {'origin': 'inference'}, 'items': []}],
    {'mode': 'scene', 'meta': {'origin': 'legacy'}, 'items': []},
    {'mode': 'scene', 'meta': {'origin': 'inference'}, 'items': [{'tag': 'blue_hair', 'score': None}]},
    {'mode': 'spans', 'meta': {'origin': 'inference'}, 'items': [{'tag': 'blue_hair', 'score': None, 'start': 0, 'end': 1}]},
    {'mode': 'spans', 'meta': {'origin': 'inference'}, 'items': [{'tag': 'blue_hair', 'score': .9, 'start': 2, 'end': 1}]},
])
def test_archive_rejects_imports_histories_and_invalid_proofs(tmp_path, record):
    from stash_ai.storage.codec import encode
    path = tmp_path / 'state.sqlite3'
    seed(path, 'scope')
    kind = 'markers' if isinstance(record, dict) and record['mode'] == 'spans' else 'scene'
    with sqlite3.connect(path) as connection:
        connection.execute('INSERT INTO inference_archive VALUES (?,?,?,?,?,?)',
            ('scope', 'old', 'wd', kind, 'invalid', encode(record)))
    before = path.read_bytes()
    assert issues(check_database(path, 'scope')) == {'state_payload_invalid': 1}
    assert path.read_bytes() == before
