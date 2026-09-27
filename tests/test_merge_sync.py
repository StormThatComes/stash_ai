"""One-scene sync extraction keeps merge.py compatibility entry points."""
from stash_ai import merge
from stash_ai import merge_sync


def test_sync_merged_wrapper_delegates_to_sync_module(monkeypatch):
    captured = {}

    def fake(*args, **kwargs):
        captured["args"] = args
        captured["kwargs"] = kwargs
        return {"ok": 1}

    monkeypatch.setattr(merge_sync, "sync_merged", fake)
    values = ("app", "stash", "resolver", "store", {"id": "1"}, ["hit"], ["span"])

    assert merge.sync_merged(
        *values, migrate=False, cleanup=True, report="report",
        tag_policies="policies", publication_key="publication") == {"ok": 1}
    assert captured["args"] == values
    assert captured["kwargs"] == {
        "migrate": False,
        "cleanup": True,
        "report": "report",
        "tag_policies": "policies",
        "publication_key": "publication",
    }


def test_merge_sync_plan_reports_scene_change():
    unchanged = merge_sync.MergeSyncPlan({}, {"1"}, set(), {"1"}, set())
    changed = merge_sync.MergeSyncPlan({}, {"1"}, {"2"}, {"1", "2"}, set())

    assert not unchanged.scene_changed
    assert changed.scene_changed
