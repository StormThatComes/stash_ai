"""Merged run orchestration keeps merge.py compatibility entry points."""
from types import SimpleNamespace

from stash_ai import merge
from stash_ai import merge_run


def test_run_merged_wrapper_delegates_to_orchestrator(monkeypatch):
    captured = {}

    class FakeRun:
        def __init__(self, *args):
            captured["args"] = args

        def run(self):
            return 7

    monkeypatch.setattr(merge_run, "MergedRun", FakeRun)
    args = SimpleNamespace()

    values = ("app", ["settings"], args, {"catalog": 1}, "hierarchy", {"core": 1}, {"scene": 1})
    assert merge._run_merged(*values) == 7
    assert captured["args"] == values
