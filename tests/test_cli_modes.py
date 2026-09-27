"""Standalone CLI mode dispatch remains isolated from the normal pipeline."""
from stash_ai import cli as app
from stash_ai import cli_args, cli_modes


def test_database_dispatch_ignores_normal_pipeline():
    args = cli_args.parse_args(app, [])

    assert cli_modes.run_database_command(app, args) is None


def test_database_dispatch_delegates_before_runtime(monkeypatch):
    from stash_ai import db

    called = {}
    monkeypatch.setattr(db, "run_mode",
                        lambda args, url: called.update(url=url, has_report=hasattr(args, "report")) or 7)
    args = cli_args.parse_args(app, ["--db-report"])

    assert cli_modes.run_database_command(app, args) == 7
    assert called == {"url": app.STASH_URL, "has_report": False}


def test_model_report_dispatch_skips_model_loading(monkeypatch, tmp_path):
    from stash_ai import model_report

    destination = tmp_path / "report.json"
    monkeypatch.setattr(model_report, "run", lambda selected_app, args:
                        9 if selected_app is app and args.model_report == destination else 1)
    args = cli_args.parse_args(app, ["--model-report", str(destination)])

    assert cli_modes.run_standalone_command(app, args, cleanup_only=False) == 9


def test_prune_dispatch_runs_config_validation(monkeypatch):
    from stash_ai import prune

    validated = []
    monkeypatch.setattr(app, "validate_config", lambda *, scene_work: validated.append(scene_work))
    monkeypatch.setattr(prune, "run_prune", lambda selected_app, args: 5)
    args = cli_args.parse_args(app, ["--prune-empty-tags-only"])

    assert cli_modes.run_standalone_command(app, args, cleanup_only=False) == 5
    assert validated == [False]
