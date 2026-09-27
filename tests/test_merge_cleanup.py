"""Cleanup extraction keeps merge.py compatibility entry points."""
from stash_ai import merge
from stash_ai import merge_cleanup


def test_run_cleanup_wrapper_delegates_to_orchestrator(monkeypatch):
    captured = {}

    class FakeRun:
        def __init__(self, *args):
            captured["args"] = args

        def run(self):
            return 7

    monkeypatch.setattr(merge_cleanup, "CleanupRun", FakeRun)
    values = ("app", ["settings"], "args", {"core": 1}, {"scene": 1})

    assert merge.run_cleanup(*values) == 7
    assert captured["args"] == values


def test_cleanup_support_wrapper_preserves_public_api(monkeypatch):
    captured = {}

    def fake(*args):
        captured["args"] = args
        return ["cleaned"]

    monkeypatch.setattr(merge_cleanup, "cleanup_support", fake)
    values = ("app", "store", "1", ["span"], {"wd"}, {"wd": frozenset()}, {"wd": frozenset()})

    assert merge.cleanup_support(*values) == ["cleaned"]
    assert captured["args"] == values
