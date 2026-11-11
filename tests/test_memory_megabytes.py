"""Public MB limits must reach admission and survive YAML export without unit drift."""
import re

import pytest
import yaml

from stash_ai.byte_units import BYTES_PER_MB, bytes_to_mb
from stash_ai.recognition.config import defaults, effective_data, load, plain
from stash_ai.recognition.config_layout import (
    INTERNAL_PATHS, LEGACY_BYTE_PATHS, PUBLIC_PATHS, get, leaf_items, put, v2_document,
)
from stash_ai.recognition.config_units import BYTE_LIMITS


def test_every_byte_budget_has_a_public_mb_key_and_shared_bounds():
    budgets = {path for path, _ in leaf_items(defaults()) if path.endswith('_bytes')}
    assert budgets == set(BYTE_LIMITS) == set(LEGACY_BYTE_PATHS.values())
    assert all(INTERNAL_PATHS[path].endswith('_mb') for path in budgets)
    assert not any(path.endswith('_bytes') for path in PUBLIC_PATHS)


@pytest.mark.parametrize('internal', BYTE_LIMITS)
def test_fractional_limits_round_trip_and_match_legacy_byte_inputs(internal):
    public = INTERNAL_PATHS[internal]
    supplied = {'version': 2}
    put(supplied, public, 1.25)
    _, data = effective_data(supplied)
    assert get(data, internal) == 1310720
    assert type(get(data, internal)) is int
    exported = v2_document(data)
    assert get(exported, public) == 1.25
    assert effective_data(yaml.safe_load(yaml.safe_dump(exported)))[1] == data
    old = {'version': 2}
    put(old, public[:-3] + '_bytes', 1310720)
    assert effective_data(old)[1] == data


@pytest.mark.parametrize('internal', BYTE_LIMITS)
def test_limit_boundaries_are_checked_in_mb_before_truncation(internal):
    public = INTERNAL_PATHS[internal]
    low, high = BYTE_LIMITS[internal]
    supplied = {'version': 2}
    for bound in (low, high):
        if bound is None:
            continue
        put(supplied, public, bytes_to_mb(bound))
        assert get(effective_data(supplied)[1], internal) == bound
    for invalid in (low / BYTES_PER_MB - .5 / BYTES_PER_MB,
                    None if high is None else high / BYTES_PER_MB + .5 / BYTES_PER_MB):
        if invalid is None:
            continue
        put(supplied, public, invalid)
        with pytest.raises(ValueError, match=re.escape(public) + '.*MB'):
            effective_data(supplied)


@pytest.mark.parametrize('value', [True, False, None, '1.5', '1g', float('nan'),
                                  float('inf'), -1, 10**400, 1e308, 1e-20])
def test_invalid_limits_do_not_become_disabled_or_unbounded(value):
    with pytest.raises(ValueError):
        effective_data({'version': 2, 'execution': {'scan': {'media_cache_mb': value}}})


@pytest.mark.parametrize('reverse', [False, True])
def test_legacy_and_mb_keys_cannot_silently_override_each_other(reverse):
    pairs = [('memory_mb', 1.5), ('memory_bytes', 1572864)]
    supplied = {'version': 2, 'execution': {'rebuild': dict(reversed(pairs) if reverse else pairs)}}
    with pytest.raises(ValueError, match='do not combine'):
        effective_data(supplied)


def test_migration_of_legacy_v2_preserves_nonintegral_mb_and_exact_bytes(tmp_path):
    from stash_ai.recognition.config_export import migrate

    source = tmp_path / 'old.yml'
    supplied = {'version': 2}
    for old, internal in LEGACY_BYTE_PATHS.items():
        put(supplied, old, max(BYTE_LIMITS[internal][0], BYTES_PER_MB) + 1)
    source.write_text(yaml.safe_dump(supplied))
    config = load(source)
    destination = tmp_path / 'new.yml'
    migrate(config, destination)
    assert plain(load(destination).data) == plain(config.data)
    assert '_bytes:' not in destination.read_text()
    assert 'memory_mb: 1.0000009536743164' in destination.read_text()


def test_solver_full_and_diff_export_fractional_mb_with_internal_byte_edits(tmp_path):
    from stash_ai.recognition.merge_goal_output import configuration_document
    from stash_ai.recognition.merge_report_policy import patched, proposal

    source = tmp_path / 'input.yml'
    source.write_text('version: 2\nstorage: {max_crop_mb: 0.5}\n')
    config = load(source)
    changes = {'execution.rebuild.memory_mb': 1.25, 'storage.max_crop_mb': .125}
    changed = patched(config, changes)
    recommendation = proposal(config, changes)
    assert changed.data['execution']['rebuild_memory_bytes'] == 1310720
    diff = yaml.safe_load(configuration_document(config, recommendation, {}, mode='diff'))
    assert diff == {'execution': {'rebuild': {'memory_mb': 1.25}}, 'storage': {'max_crop_mb': .125}}
    full = yaml.safe_load(configuration_document(config, recommendation, {}, mode='full'))
    assert full['execution']['rebuild']['memory_mb'] == 1.25
    assert full['storage']['max_crop_mb'] == .125
    restored = effective_data(full)[1]
    assert restored['execution'] == plain(changed.data['execution'])
    assert restored['storage']['max_crop_bytes'] == 131072


def test_decimal_fraction_normalizes_once_to_integer_bytes():
    supplied = {'version': 2, 'storage': {'max_crop_mb': .1}}
    _, data = effective_data(supplied)
    assert data['storage']['max_crop_bytes'] == 104857
    assert effective_data(v2_document(data))[1] == data


def test_solver_cli_admission_uses_fractional_mb():
    from stash_ai.recognition.cli import parser
    from stash_ai.recognition.solver_execution import cli_options

    args = parser().parse_args(['report-merge', '--performer-set', '1,2', '--solve',
                               '--solver-memory-mb', '65536.5'])
    assert args.solver_memory_mb == 65536.5
    assert cli_options(args)['memory_bytes'] == 65536 * BYTES_PER_MB + 524288
    assert type(cli_options(args)['memory_bytes']) is int


@pytest.mark.parametrize('value', ['0', '-1', 'nan', 'inf', '1e308', '1e-20'])
def test_solver_cli_rejects_invalid_memory(value):
    from stash_ai.recognition.cli import parser

    with pytest.raises(SystemExit):
        parser().parse_args(['report-merge', '--performer-set', '1,2', '--solve',
                             '--solver-memory-mb', value])


def test_tagging_cli_fractional_mb_reaches_runtime_byte_budgets():
    from stash_ai.commands.arguments import parse_args
    from stash_ai.reports.runtime import report_memory
    from tests.helpers.runtime import runtime

    args = parse_args(runtime, ['--ram-cache-mb', '.5', '--ram-budget-mb', '1.25'])
    assert args.ram_cache_mb == .5 and args.ram_budget_mb == 1.25
    assert report_memory(args) == {'cache_bytes': 524288, 'memory_budget_bytes': 1310720}


@pytest.mark.parametrize('flag', ['--ram-cache-mb', '--ram-budget-mb', '--rethreshold-spool-limit-mb'])
@pytest.mark.parametrize('value', ['nan', 'inf', '-1', '1e308', '1e-20'])
def test_tagging_memory_switches_reject_invalid_sizes(flag, value):
    from stash_ai.commands.arguments import parse_args
    from tests.helpers.runtime import runtime

    with pytest.raises(SystemExit):
        parse_args(runtime, [flag, value])
