"""CLI parser defaults and cross-option validation."""
import pytest

from tests.helpers.runtime import runtime as app
from stash_ai.commands import arguments, validation


def test_parse_args_uses_live_cli_defaults_and_caps_cache(monkeypatch, tmp_path):
    monkeypatch.setattr(app, "RAM_CACHE_GB", 12.0)
    monkeypatch.setattr(app, "RAM_BUDGET_GB", 8.0)
    monkeypatch.setattr(app, "DEFAULT_RESUME", False)
    monkeypatch.setattr(app, "PREPARE_WORKERS", 3)
    configuration = tmp_path / 'models.yml'
    configuration.write_text('version: 1\n')
    monkeypatch.setattr(app, 'MODELS_CONFIG_FILE', configuration)

    args = arguments.parse_args(app, [])

    assert args.ram_cache_gb == 6.0
    assert args.ram_budget_gb == 8.0
    assert args.resume is False
    assert args.model_config == app.MODELS_CONFIG_FILE
    assert args.merge_workers == 4
    assert args.prepare_workers == 3
    assert args.profile_merge is None
    assert args.force_merge is args.rethreshold_markers is args.rethreshold_scenes is False
    assert args.abandon_pending is args.prune_empty_tags is False
    assert args.cleanup_strategy == "auto"


def test_parse_args_rejects_merged_only_mode_with_separate_output():
    with pytest.raises(SystemExit) as error:
        arguments.parse_args(app, ["--merge-only", "--output-mode", "separate"])

    assert error.value.code == 2


def test_mode_flags_are_derived_without_runtime_mutation():
    args = arguments.parse_args(app, ["--cleanup-only"])

    assert validation.mode_flags(args) == (False, True)


def test_parse_args_accepts_parallel_merge_workers():
    args = arguments.parse_args(app, ["--merge-only", "--merge-workers", "4"])

    assert args.merge_workers == 4

