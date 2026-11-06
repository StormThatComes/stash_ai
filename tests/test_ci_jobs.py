"""Diff selection must preserve mixed changes, renames and failed required jobs."""
import json
import subprocess

import pytest

from tests.ci_jobs import check_results, event_plan, full_plan, pr_changes, select_jobs, test_dependencies as dependencies


def suites(result):
    return {row['suite'] for row in result['matrix']['include']}


@pytest.mark.parametrize('paths,expected', [
    (['README.md', 'docs/facial-recognition.md'], set()),
    (['stash_ai/recognition/clustering.py'], set()),
    (['stash_ai/recognition/merge_solver.py'], {'core'}),
    (['stash_ai/recognition/scanning.py'], {'models', 'core', 'video'}),
    (['tests/test_recognition_model_smoke.py'], {'models', 'core'}),
    (['tests/test_recognition_video_recovery.py'], {'video'}),
    (['README.md', 'stash_ai/recognition/merge_solver.py'], {'core'}),
])
def test_pr_selects_existing_jobs_by_scope(paths, expected):
    result = select_jobs([('M', path) for path in paths])
    assert suites(result) == expected
    assert result['pytest'] == any(not (path.endswith('.md')) for path in paths)
    assert not result['dependencies'] and not result['updater']
    assert result['recognition'] == bool(expected)


@pytest.mark.parametrize('path', ['requirements-cpu.txt', 'constraints/cpu-py314.txt',
    '.github/workflows/tests.yml', 'tests/conftest.py', 'tests/helpers/recognition.py',
    'tests/ci_jobs.py', 'unclassified/new-file.toml', 'docs/executable.py'])
def test_dependency_shared_infrastructure_and_unknown_paths_run_everything(path):
    result = select_jobs([('M', path)])
    assert all(result[key] for key in ('pytest', 'updater', 'dependencies', 'recognition'))
    assert suites(result) == {'models', 'core', 'video'}


@pytest.mark.parametrize('status', ['A', 'D'])
def test_installed_module_changes_also_check_the_minimal_updater(status):
    result = select_jobs([(status, 'stash_ai/recognition/query_matrix.py')])
    assert result['pytest'] and result['updater']


def test_windows_selection_includes_imported_fixture_modules():
    result = select_jobs([('M', 'tests/test_recognition_clustering.py')])
    # This module appears in the video manifest and supplies fixtures to core tests.
    assert {'core', 'video'} <= suites(result)


def test_benchmark_fixtures_select_their_windows_consumers():
    result = select_jobs([('M', 'benchmarks/benchmark_merge_solver.py')])
    assert result['pytest'] and 'core' in suites(result)
    assert not result['dependencies']


def test_transitive_test_imports_include_relative_imports_and_cycles(tmp_path):
    folder = tmp_path / 'tests'
    folder.mkdir()
    (folder / 'first.py').write_text('from .second import fixture\n')
    (folder / 'second.py').write_text('from tests import third\n')
    (folder / 'third.py').write_text('import tests.first\n')
    assert dependencies(['tests/first.py'], tmp_path) == {'tests/first.py', 'tests/second.py', 'tests/third.py'}


def test_pr_diff_uses_merge_base_and_includes_both_sides_of_rename(tmp_path):
    def git(*args):
        return subprocess.check_output(['git', *args], cwd=tmp_path, text=True).strip()
    git('init', '-b', 'main')
    git('config', 'user.name', 'CI test')
    git('config', 'user.email', 'ci@example.invalid')
    (tmp_path / 'README.md').write_text('before\n')
    git('add', '.')
    git('commit', '-m', 'base')
    git('switch', '-c', 'feature')
    git('mv', 'README.md', 'code.py')
    (tmp_path / 'file with spaces.py').write_text('value=1\n')
    git('add', '.')
    git('commit', '-m', 'feature')
    head = git('rev-parse', 'HEAD')
    git('switch', 'main')
    (tmp_path / 'base-only.txt').write_text('new base content\n')
    git('add', '.')
    git('commit', '-m', 'advance main')
    base = git('rev-parse', 'HEAD')
    event = {'pull_request': {'base': {'sha': base}, 'head': {'sha': head}}}
    assert set(pr_changes(event, tmp_path)) == {
        ('D', 'README.md'), ('A', 'code.py'), ('A', 'file with spaces.py')}


@pytest.mark.parametrize('event', ['push', 'workflow_dispatch'])
def test_post_merge_and_manual_runs_always_run_the_full_matrix(event):
    result = event_plan(event, {})
    assert all(result[key] for key in ('pytest', 'updater', 'dependencies', 'recognition'))
    assert len(result['matrix']['include']) == 5


def test_windows_partitions_keep_scope_and_only_one_model_smoke_per_platform():
    matrix = select_jobs([('M', 'stash_ai/recognition/scanning.py')])['matrix']['include']
    windows = [row for row in matrix if row['os'] == 'windows-latest']
    assert {(row['suite'], row['shard']) for row in windows} == {
        ('core', 1), ('core', 2), ('video', 1), ('video', 2)}
    assert [row['shard'] for row in windows if row['suite'] == 'core' and row['smoke']] == [1]
    solver = select_jobs([('M', 'stash_ai/recognition/merge_solver.py')])['matrix']['include']
    assert {(row['suite'], row['shard']) for row in solver} == {('core', 1), ('core', 2)}
    assert not any(row['smoke'] for row in solver)


def test_unavailable_pr_diff_falls_back_to_full_validation():
    result = event_plan('pull_request', {})
    assert result == full_plan('PR diff unavailable (KeyError); run all checks')


def result_needs(selected):
    mapping = {'pytest': 'pytest', 'updater': 'updater', 'dependencies': 'dependency-resolution',
               'recognition': 'recognition-models'}
    return {'plan': {'result': 'success', 'outputs': {key: json.dumps(selected[key]) for key in mapping}},
            **{job: {'result': 'success' if selected[key] else 'skipped'} for key, job in mapping.items()}}


def test_result_allows_intentionally_skipped_jobs():
    check_results(result_needs(select_jobs([('M', 'README.md')])))
    check_results(result_needs(select_jobs([('M', 'stash_ai/recognition/clustering.py')])))


@pytest.mark.parametrize('status', ['failure', 'cancelled', 'skipped'])
def test_result_rejects_selected_jobs_that_did_not_succeed(status):
    needs = result_needs(full_plan('test'))
    needs['pytest']['result'] = status
    with pytest.raises(ValueError, match='pytest finished'):
        check_results(needs)


def test_failed_planner_cannot_produce_a_green_result():
    with pytest.raises(ValueError, match='selection did not succeed'):
        check_results({'plan': {'result': 'failure'}})
