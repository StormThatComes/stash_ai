"""CLI phase dispatch keeps saved-evidence modes ahead of model assets."""
import pytest

from stash_ai import cli as app
from stash_ai.commands import arguments, dispatch


def test_database_dispatch_ignores_normal_pipeline():
    args = arguments.parse_args(app, [])

    assert dispatch.run_database_command(app, args) is None


def test_database_dispatch_delegates_before_runtime(monkeypatch):
    from stash_ai import db

    called = {}
    monkeypatch.setattr(db, "run_mode",
                        lambda args, url: called.update(url=url, has_report=hasattr(args, "report")) or 7)
    args = arguments.parse_args(app, ["--db-report"])

    assert dispatch.run_database_command(app, args) == 7
    assert called == {"url": app.STASH_URL, "has_report": False}


def test_model_report_dispatch_skips_model_loading(monkeypatch, tmp_path):
    from stash_ai.reports import model as model_report

    destination = tmp_path / "report.json"
    monkeypatch.setattr(model_report, "run", lambda selected_app, args:
                        9 if selected_app is app and args.model_report == destination else 1)
    args = arguments.parse_args(app, ["--model-report", str(destination)])

    assert dispatch.run_standalone_command(app, args, cleanup_only=False) == 9


def test_prune_dispatch_runs_config_validation(monkeypatch):
    from stash_ai.cleanup import prune

    validated = []
    monkeypatch.setattr(app, "validate_config", lambda *, scene_work: validated.append(scene_work))
    monkeypatch.setattr(prune, "run_prune", lambda selected_app, args: 5)
    args = arguments.parse_args(app, ["--prune-empty-tags-only"])

    assert dispatch.run_standalone_command(app, args, cleanup_only=False) == 5
    assert validated == [False]


@pytest.mark.parametrize('flags', [[], ['--output-mode', 'separate']])
def test_saved_merge_dispatch_leaves_normal_tagging_to_pipeline(flags):
    args = arguments.parse_args(app, flags)
    assert dispatch.run_saved_merge_command(app, [], args, {}, {}) is None


def test_merge_only_dispatch_passes_empty_catalogs_and_selected_policies(monkeypatch):
    from stash_ai.commands import hierarchy

    args = arguments.parse_args(app, ['--merge-only'])
    settings, cores, tags, combined = [], {'wd': frozenset()}, {'wd': frozenset()}, object()
    monkeypatch.setattr(hierarchy, 'combined_hierarchy',
                        lambda selected_app, selected_args, policies:
                        combined if (selected_app is app and selected_args is args and policies is tags) else None)
    called = []
    monkeypatch.setattr(dispatch, 'run_merged', lambda *values: called.append(values) or 7)

    assert dispatch.run_saved_merge_command(app, settings, args, cores, tags) == 7
    assert called == [(app, settings, args, {}, combined, cores, tags)]


def test_explain_dispatch_uses_diagnostics_without_entering_merge(monkeypatch):
    from stash_ai.commands import explain

    args = arguments.parse_args(app, ['--explain-tag', 'common', '--scene-id', '1'])
    cores, tags = {}, {}
    called = []
    monkeypatch.setattr(explain, 'explain', lambda *values: called.append(values) or 9)
    monkeypatch.setattr(dispatch, 'run_merged', lambda *args: pytest.fail('explain entered merged publication'))

    assert dispatch.run_saved_merge_command(app, [], args, cores, tags) == 9
    assert called == [(app, args, cores, tags)]
