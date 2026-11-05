"""Catch YAML syntax errors before publishing a workflow that cannot start."""
from pathlib import Path

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
    root = Path(__file__).resolve().parents[1]
    workflow = yaml.safe_load((root / '.github/workflows/tests.yml').read_text())
    matrix = workflow['jobs']['recognition-models']['strategy']['matrix']['include']
    files = []
    for row in matrix:
        if row['os'] == 'windows-latest':
            paths = (root / '.github/ci' / f"windows-{row['suite']}.txt").read_text().splitlines()
            assert paths and all((root / path).is_file() for path in paths)
            files.extend(paths)
    assert files and len(files) == len(set(files))
