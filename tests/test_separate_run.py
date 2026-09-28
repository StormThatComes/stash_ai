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
