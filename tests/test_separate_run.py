"""Separate-output model lifecycle and reporting."""
from stash_ai import cli
from stash_ai.inference import preparation
from stash_ai import separate_run
from tests.helpers.cli import runtime_args


def test_run_model_accumulates_scene_phases_and_closes_tagger(monkeypatch):
    from types import SimpleNamespace
    from stash_ai.models import WD

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
    def prepare_image(app, image, size):
        events.append(("prepare", app, image, size))
        return "prepared"

    def load_tagger(config, vocabulary, prepare):
        assert prepare("frame", 448) == "prepared"
        return Tagger()

    monkeypatch.setattr(preparation, "prepare_image", prepare_image)
    monkeypatch.setattr(cli, "load_tagger", load_tagger)
    monkeypatch.setattr(cli, "print_hierarchy_summary", lambda resolver: events.append("summary"))
    monkeypatch.setattr(cli, "safe_error", lambda exc, **kwargs: str(exc))

    def process(*args):
        scene_id = args[5]
        if scene_id == "2":
            raise ValueError("bad scene")
        return {"processed": 1, "markers_created": 2}

    monkeypatch.setattr(separate_run, "_process_scene", process)
    args = runtime_args(
        report=Report(),
        output_mode="separate",
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
    assert events[0] == ("prepare", cli, "frame", 448)
    assert events[-2:] == ["summary", "closed"]
    assert sum(isinstance(event, tuple) and event[0] == "progress" for event in events) == 2
