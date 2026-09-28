"""Compatibility boundary for extracted separate-output reconciliation."""
from stash_ai import cli
from stash_ai import separate_sync
from stash_ai.taggers import WD


def test_cli_separate_sync_wrappers_delegate(monkeypatch):
    calls = []

    monkeypatch.setattr(
        separate_sync,
        "owned_marker",
        lambda marker, source: calls.append(("owned", marker, source)) or True,
    )
    monkeypatch.setattr(
        separate_sync,
        "marker_has_excluded_core",
        lambda marker, exclusions, source:
            calls.append(("core", marker, exclusions, source)) or False,
    )
    monkeypatch.setattr(
        separate_sync,
        "marker_extra_cleanup_payload",
        lambda marker, scene_id, blocked, has_end:
            calls.append(("extras", marker, scene_id, blocked, has_end)) or {"id": "x"},
    )
    monkeypatch.setattr(
        separate_sync,
        "marker_differs",
        lambda existing, wanted:
            calls.append(("differs", existing, wanted)) or True,
    )

    assert cli.owned_marker({"id": "1"}, WD)
    assert not cli.marker_has_excluded_core({"id": "1"}, frozenset({"x"}), WD)
    assert cli.marker_extra_cleanup_payload(
        {"id": "1"}, "scene", {"wd:x"}, True
    ) == {"id": "x"}
    assert cli.marker_differs({"id": "1"}, {"id": "2"})

    assert [item[0] for item in calls] == ["owned", "core", "extras", "differs"]


def test_cli_sync_scene_wrapper_passes_live_app(monkeypatch):
    captured = {}

    def fake(*args):
        captured["args"] = args
        return {"processed": 1}

    monkeypatch.setattr(separate_sync, "sync_scene", fake)
    values = (
        "stash",
        "resolver",
        {"id": "1"},
        [("tag", .9)],
        ["spec"],
        frozenset({"core"}),
        WD,
        frozenset({"scene"}),
    )

    assert cli.sync_scene(*values) == {"processed": 1}
    assert captured["args"][0] is cli
    assert captured["args"][1:] == values


def test_cli_marker_plan_wrappers_pass_live_app(monkeypatch):
    captured = {}

    def payload(*args):
        captured["payload"] = args
        return {"scene_id": "1"}

    def plan(*args):
        captured["plan"] = args
        return [], [], [], 0

    monkeypatch.setattr(separate_sync, "marker_payload", payload)
    monkeypatch.setattr(separate_sync, "plan_marker_changes", plan)

    spec = object()
    ids = {"wd:x": "1"}
    assert cli.marker_payload(spec, "1", ids, True, WD) == {"scene_id": "1"}
    assert cli.plan_marker_changes([], [], frozenset(), WD) == ([], [], [], 0)

    assert captured["payload"] == (cli, spec, "1", ids, True, WD)
    assert captured["plan"] == (cli, [], [], frozenset(), WD)


def test_generic_sync_helpers_require_explicit_source():
    import inspect

    assert inspect.signature(separate_sync.owned_marker).parameters["source"].default is inspect.Parameter.empty
    assert inspect.signature(separate_sync.marker_has_excluded_core).parameters["source"].default is inspect.Parameter.empty
    assert inspect.signature(separate_sync.marker_payload).parameters["source"].default is inspect.Parameter.empty
    assert inspect.signature(separate_sync.plan_marker_changes).parameters["source"].default is inspect.Parameter.empty
    assert inspect.signature(separate_sync.sync_scene).parameters["source"].default is inspect.Parameter.empty
    assert inspect.signature(cli.owned_marker).parameters["source"].default is inspect.Parameter.empty
    assert inspect.signature(cli.sync_scene).parameters["source"].default is inspect.Parameter.empty
