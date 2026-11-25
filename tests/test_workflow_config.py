"""Catch YAML syntax errors before publishing a workflow that cannot start."""
from pathlib import Path

import pytest
import yaml


def test_workflows_parse():
    for path in (Path(__file__).resolve().parents[1] / '.github/workflows').glob('*.yml'):
        workflow = yaml.safe_load(path.read_text())
        assert isinstance(workflow['jobs'], dict), path
        for job in workflow['jobs'].values():
            for step in job['steps']:
                if 'run' in step:
                    assert isinstance(step['run'], str), (path, step)


def test_linux_matrix_covers_every_test_module_once():
    from tests.ci_shards import shard_number

    root = Path(__file__).resolve().parents[1]
    workflow = yaml.safe_load((root / '.github/workflows/tests.yml').read_text())
    shards = workflow['jobs']['pytest']['strategy']['matrix']['shard']
    for path in (root / 'tests').rglob('*.py'):
        assert shards.count(shard_number(path.relative_to(root).as_posix())) == 1, path


def test_windows_lifecycle_lists_are_disjoint_and_resolve_to_test_modules():
    from tests.ci_jobs import full_plan
    from tests.ci_windows import selected

    root = Path(__file__).resolve().parents[1]
    matrix = full_plan('test')['matrix']['include']
    files = []
    for row in matrix:
        if row['os'] == 'windows-latest':
            paths = selected(row['suite'], row['shard'], root)
            assert paths and all((root / path).is_file() for path in paths)
            files.extend(paths)
    assert files and len(files) == len(set(files))
    manifests = [path for suite in ('core', 'video') for path in
                 (root / '.github/ci' / f'windows-{suite}.txt').read_text().splitlines()]
    assert sorted(files) == sorted(manifests)


def test_windows_pr_selectors_resolve_without_overlapping_modules():
    import ast
    from tests.ci_windows import SHARDS, selected

    root = Path(__file__).resolve().parents[1]
    for suite in ('core', 'video'):
        full = {path for shard in SHARDS[suite] for path in selected(suite, shard, root)}
        focused = selected(suite, 1, root, profile='pr')
        assert len(set(focused)) == len(focused)
        for selector in focused:
            path, separator, function = selector.partition('::')
            assert path in full
            if separator:
                assert path not in focused
                module = ast.parse((root / path).read_text(encoding='utf-8'))
                assert function in {node.name for node in module.body if isinstance(node, ast.FunctionDef)}


def test_windows_rebuild_and_checkpoint_suites_use_separate_runners():
    from tests.ci_jobs import full_plan
    from tests.ci_windows import selected

    owners = {path: row['shard'] for row in full_plan('test')['matrix']['include']
              if row['os'] == 'windows-latest' and row['suite'] == 'core'
              for path in selected('core', row['shard'])}
    assert owners['tests/test_recognition_rebuild.py'] == 2
    assert owners['tests/test_recognition_replay_checkpoints.py'] == 3
    assert owners['tests/test_recognition_identity_inspector.py'] == 1


@pytest.mark.parametrize('suite,shard,profile', [
    ('core', 0, 'full'), ('core', 4, 'full'), ('video', 3, 'full'),
    ('unknown', 1, 'full'), ('core', 2, 'pr'), ('core', 3, 'pr'),
])
def test_windows_rejects_partitions_outside_the_selected_profile(suite, shard, profile):
    from tests.ci_windows import selected

    with pytest.raises(ValueError):
        selected(suite, shard, profile=profile)
