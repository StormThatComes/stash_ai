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


def test_dry_run_and_profile_keep_merge_workers_serial(monkeypatch):
    calls = []

    for dry_run, profile in ((True, None), (False, "profile.jsonl")):
        app = SimpleNamespace(DRY_RUN=dry_run)
        args = SimpleNamespace(
            merge_workers=4,
            profile_merge=profile,
            verbose=False,
            report=None,
        )
        run = merge_run.MergedRun(app, [], args, {}, None, {}, {})
        run.scenes = {"1": {}, "2": {}}
        monkeypatch.setattr(
            run, "_merge_scenes_serial",
            lambda: calls.append(("serial", dry_run, profile)))
        monkeypatch.setattr(
            run, "_merge_scenes_parallel",
            lambda workers: calls.append(("parallel", workers)))

        run._merge_scenes()

    assert calls == [
        ("serial", True, None),
        ("serial", False, "profile.jsonl"),
    ]
