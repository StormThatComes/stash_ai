"""CLI parser defaults and cross-option validation."""
import pytest

from stash_ai import cli as app
from stash_ai import cli_args


def test_parse_args_uses_live_cli_defaults_and_caps_cache(monkeypatch):
    monkeypatch.setattr(app, "RAM_CACHE_GB", 12.0)
    monkeypatch.setattr(app, "RAM_BUDGET_GB", 8.0)
    monkeypatch.setattr(app, "DEFAULT_RESUME", False)

    args = cli_args.parse_args(app, [])

    assert args.ram_cache_gb == 6.0
    assert args.ram_budget_gb == 8.0
    assert args.resume is False
    assert args.model_config == app.MODELS_CONFIG_FILE
    assert args.merge_workers == 1


def test_parse_args_rejects_merged_only_mode_with_separate_output():
    with pytest.raises(SystemExit) as error:
        cli_args.parse_args(app, ["--merge-only", "--output-mode", "separate"])

    assert error.value.code == 2


def test_mode_flags_are_derived_without_runtime_mutation():
    args = cli_args.parse_args(app, ["--cleanup-only"])

    assert cli_args.mode_flags(args) == (False, True)


def test_parse_args_accepts_parallel_merge_workers():
    args = cli_args.parse_args(app, ["--merge-only", "--merge-workers", "4"])

    assert args.merge_workers == 4
