"""Compatibility boundary for separate-output orchestration."""
from stash_ai import cli
from stash_ai import separate_run


def test_cli_run_model_wrapper_delegates_to_separate_run(monkeypatch):
    captured = {}

    def fake(*args):
        captured["args"] = args
        return {"processed": 1}

    monkeypatch.setattr(separate_run, "run_model", fake)
    values = ("stash", "config", "vocabulary", "hierarchy", "args", frozenset({"core"}), frozenset({"scene"}))

    assert cli.run_model(*values) == {"processed": 1}
    assert captured["args"][0] is cli
    assert captured["args"][1:] == values


def test_run_model_accumulates_scene_phases_and_closes_tagger(monkeypatch):
    from types import SimpleNamespace
    from stash_ai.taggers import WD

    events = []
    issues = []

    class Resolver:
        def __init__(self, *args, **kwargs):
            self.source = WD
            self.missing = set()
            self.hierarchy = None

    class Tagger:
        def close(self):
            events.append("closed")

    class Report:
        verbose = False
        def detail(self, message):
            pass
        def issue(self, *args):
            issues.append(args)
        def progress(self, progress, stats):
            events.append(("progress", dict(stats)))

    class Stash:
        def iter_scenes(self, scene_id):
            return [{"id": "1"}, {"id": "2"}]

    monkeypatch.setattr(cli, "TagResolver", Resolver)
    monkeypatch.setattr(cli, "load_tagger", lambda *args: Tagger())
    monkeypatch.setattr(cli, "print_hierarchy_summary", lambda resolver: events.append("summary"))
    monkeypatch.setattr(cli, "safe_error", lambda exc, **kwargs: str(exc))

    def process(*args):
        scene_id = args[5]
        if scene_id == "2":
            raise ValueError("bad scene")
        return {"processed": 1, "markers_created": 2}

    monkeypatch.setattr(separate_run, "_process_scene", process)
    args = SimpleNamespace(
        report=Report(),
        sync_hierarchy_only=False,
        scene_id=None,
    )
    config = SimpleNamespace(source=WD)

    stats = separate_run.run_model(
        cli, Stash(), config, SimpleNamespace(labels=[]), None, args,
        frozenset(), frozenset(),
    )

    assert stats["processed"] == 1
    assert stats["markers_created"] == 2
    assert stats["errors"] == 1
    assert len(issues) == 1 and issues[0][0:3] == ("inference", "wd", "2")
    assert events[-2:] == ["summary", "closed"]
    assert sum(isinstance(event, tuple) and event[0] == "progress" for event in events) == 2
