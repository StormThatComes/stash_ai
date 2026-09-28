"""Compatibility boundaries for extracted media and inference helpers."""
from stash_ai import cli
from stash_ai import inference


def test_cli_media_wrappers_delegate(monkeypatch):
    monkeypatch.setattr(inference, "evenly_sample", lambda items, count: ("sample", items, count))
    monkeypatch.setattr(inference, "fetch_cues", lambda app, stash, paths: ("cues", app, stash, paths))
    monkeypatch.setattr(inference, "iter_sprite_frames", lambda stash, paths, cues: iter([("frame", stash, paths, cues)]))
    monkeypatch.setattr(inference, "prepare_image", lambda app, image, size: ("image", app, image, size))
    monkeypatch.setattr(inference, "select_scores", lambda scores, labels, threshold, limit:
                        ("scores", scores, labels, threshold, limit))

    assert cli.evenly_sample([1, 2], 1) == ("sample", [1, 2], 1)
    fetched = cli.fetch_cues("stash", {"vtt": "x"})
    assert fetched == ("cues", cli, "stash", {"vtt": "x"})
    assert list(cli.iter_sprite_frames("stash", {"sprite": "x"}, ["cue"])) == [
        ("frame", "stash", {"sprite": "x"}, ["cue"])
    ]
    assert cli.prepare_image("image", 448) == ("image", cli, "image", 448)
    assert cli.select_scores([.9], ["tag"], .5, 0) == (
        "scores", [.9], ["tag"], .5, 0
    )


def test_cli_predict_scene_wrapper_preserves_arguments(monkeypatch):
    captured = {}

    def fake(*args, **kwargs):
        captured["args"] = args
        captured["kwargs"] = kwargs
        return ["prediction"], ["frame"]

    monkeypatch.setattr(inference, "predict_scene", fake)
    result = cli.predict_scene(
        "tagger",
        "stash",
        {"vtt": "x"},
        ["cue"],
        frozenset({"blocked"}),
        raw_scene=True,
        checkpoint="checkpoint",
        prefetch=3,
        analysis={"a": 1},
        report="report",
        prepare_workers=2,
    )

    assert result == (["prediction"], ["frame"])
    assert captured["args"] == (
        cli,
        "tagger",
        "stash",
        {"vtt": "x"},
        ["cue"],
        frozenset({"blocked"}),
    )
    assert captured["kwargs"] == {
        "raw_scene": True,
        "checkpoint": "checkpoint",
        "prefetch": 3,
        "analysis": {"a": 1},
        "report": "report",
        "prepare_workers": 2,
    }
