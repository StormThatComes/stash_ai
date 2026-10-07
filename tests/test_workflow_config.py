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
