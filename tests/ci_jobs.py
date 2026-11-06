"""Select existing CI jobs from the complete PR diff; full checks on main and manual runs."""
import argparse
import ast
from importlib.util import resolve_name
import json
import os
from pathlib import Path, PurePosixPath
import re
import subprocess


ROOT = Path(__file__).resolve().parents[1]
RECOGNITION_MATRIX = (
    {'os': 'ubuntu-latest', 'suite': 'models'},
    {'os': 'windows-latest', 'suite': 'core'},
    {'os': 'windows-latest', 'suite': 'video'},
)
# These modules use the numeric/replay contracts exercised by the full Linux suite.
# Unlisted recognition modules also run both Windows lifecycle groups and model smoke.
NUMERIC_MODULES = (
    'clustering', 'scoring', 'query_', 'cohort_', 'threshold_', 'reference_',
    'view_prototypes', 'anchor_views', 'cross_scene', 'hybrid_', 'context',
)
SOLVER_MODULES = ('merge_solver', 'merge_goal', 'merge_observation_exclusions',
                  'merge_scene_exclusions', 'solver_', 'trial_', 'body_solver')


def full_plan(reason):
    return plan({'pytest', 'updater', 'dependencies', 'models', 'core', 'video'}, {reason})


def plan(selected, reasons):
    matrix = [dict(row, smoke='models' in selected) for row in RECOGNITION_MATRIX
              if row['suite'] in selected]
    return dict(pytest='pytest' in selected, updater='updater' in selected,
                dependencies='dependencies' in selected, recognition=bool(matrix),
                matrix={'include': matrix}, reasons=sorted(reasons))


def documentation(path):
    return (path in {'README.md', 'SECURITY.md', 'THIRD_PARTY_NOTICES.md', 'LICENSE',
                     'constraints/README.md'} or
            path.startswith(('docs/', 'LICENSES/')) and PurePosixPath(path).suffix in {'.md', '.rst', '.txt'})


def test_dependencies(paths, root):
    """Include fixtures imported from other test modules, including relative imports."""
    pending, found = list(paths), set()
    while pending:
        path = pending.pop()
        if path in found:
            continue
        found.add(path)
        package = str(PurePosixPath(path).parent).replace('/', '.')
        for node in ast.walk(ast.parse((root / path).read_text(encoding='utf-8'))):
            modules = []
            if isinstance(node, ast.Import):
                modules = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                module = node.module or ''
                if node.level:
                    module = resolve_name('.' * node.level + module, package)
                modules = [module] + [module + '.' + alias.name for alias in node.names]
            for module in modules:
                dependency = module.replace('.', '/') + '.py'
                if module.startswith('tests.') and (root / dependency).is_file():
                    pending.append(dependency)
    return found


def select_jobs(changes, root=ROOT):
    if not changes:
        return full_plan('Empty PR diff; run all checks')
    windows = {row['suite']: test_dependencies((root / '.github/ci' / f"windows-{row['suite']}.txt").read_text().splitlines(), root)
               for row in RECOGNITION_MATRIX if row['os'] == 'windows-latest'}
    model_tests = test_dependencies(['tests/test_recognition_model_smoke.py'], root)
    solver_tests = test_dependencies(['tests/test_recognition_solver_concurrency.py'], root)
    selected, reasons = set(), set()
    for status, path in changes:
        if documentation(path):
            reasons.add('Documentation')
            continue
        if (path.startswith(('.github/', 'requirements', 'constraints/', 'tests/ci_', 'tests/helpers/', 'tests/fixtures/'))
                or path in {'pytest.ini', 'pyproject.toml', 'setup.cfg', 'tests/conftest.py', 'tests/__init__.py'}):
            return full_plan('CI, dependencies or shared test infrastructure changed')
        selected.add('pytest')
        if path in {'update_stash_ai.py', 'tests/test_updater.py'}:
            selected.add('updater')
            reasons.add('Updater')
        elif path.startswith('tests/test_') and path.endswith('.py'):
            selected.update(suite for suite, files in windows.items() if path in files)
            if path in model_tests:
                selected.update(('models', 'core'))
            if path in solver_tests:
                selected.add('core')
            reasons.add('Test changes and their existing Windows jobs')
        elif path.startswith('stash_ai/recognition/') and path.endswith('.py'):
            name = PurePosixPath(path).stem
            if name.startswith(SOLVER_MODULES):
                selected.add('core')
                reasons.add('Solver and Windows process/filesystem regressions')
            elif name.startswith(NUMERIC_MODULES):
                reasons.add('Recognition matching and full Linux regressions')
            else:
                selected.update(('models', 'core', 'video'))
                reasons.add('Recognition runtime, model or lifecycle changes')
            if status in {'A', 'D'}:
                selected.add('updater')
                reasons.add('Installed module added or removed')
        elif path.startswith('benchmarks/') and path.endswith('.py'):
            reasons.add('Benchmark entry points and full Linux regressions')
        elif path.startswith('stash_ai/') or path in {'stash_ai.py', 'facial_recognition.yml'}:
            selected.update(('models', 'core', 'video', 'updater'))
            reasons.add('Shared application code or configuration')
        else:
            return full_plan('Unclassified path; run all checks')
    return plan(selected, reasons)


def pr_changes(event, root=ROOT):
    pull = event['pull_request']
    base, head = pull['base']['sha'], pull['head']['sha']
    if not all(re.fullmatch(r'[0-9a-fA-F]{40}|[0-9a-fA-F]{64}', sha) for sha in (base, head)):
        raise ValueError('Invalid PR commit SHA')
    # Disabled rename detection exposes both the old and new paths as D/A.
    # NUL separators handle whitespace and filenames safely without API file limits.
    raw = subprocess.check_output(
        ['git', 'diff', '--name-status', '--no-renames', '-z', f'{base}...{head}', '--'],
        cwd=root, timeout=60)
    fields = raw.decode('utf-8', errors='surrogateescape').split('\0')
    if fields.pop() != '' or len(fields) % 2:
        raise ValueError('Malformed git diff')
    return list(zip(fields[::2], fields[1::2]))


def event_plan(event_name, event, root=ROOT):
    if event_name != 'pull_request':
        return full_plan(f'Full validation for {event_name}')
    try:
        return select_jobs(pr_changes(event, root), root)
    except (KeyError, ValueError, OSError, SyntaxError, ImportError, subprocess.SubprocessError) as error:
        return full_plan(f'PR diff unavailable ({type(error).__name__}); run all checks')


def check_results(needs):
    if needs['plan']['result'] != 'success':
        raise ValueError('CI job selection did not succeed')
    outputs = needs['plan']['outputs']
    for job, flag in [('pytest', 'pytest'), ('updater', 'updater'),
                      ('dependency-resolution', 'dependencies'), ('recognition-models', 'recognition')]:
        expected = outputs[flag]
        result = needs[job]['result']
        if expected not in {'true', 'false'}:
            raise ValueError(f'Invalid selection for {job}')
        if result != 'success' and (expected == 'true' or result != 'skipped'):
            raise ValueError(f'{job} finished as {result}')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('plan', 'check'))
    args = parser.parse_args()
    if args.command == 'check':
        check_results(json.loads(os.environ['CI_NEEDS']))
        print('All selected CI jobs passed')
        return
    event_name = os.environ['GITHUB_EVENT_NAME']
    # Push and manual full checks do not need a PR payload.
    event = json.loads(Path(os.environ['GITHUB_EVENT_PATH']).read_text()) if event_name == 'pull_request' else {}
    result = event_plan(event_name, event)
    print(json.dumps(result, indent=2))
    with Path(os.environ['GITHUB_OUTPUT']).open('a') as output:
        for key in ('pytest', 'updater', 'dependencies', 'recognition', 'matrix'):
            output.write(f'{key}={json.dumps(result[key], separators=(",", ":"))}\n')
    with Path(os.environ['GITHUB_STEP_SUMMARY']).open('a') as summary:
        summary.write('## Selected CI jobs\n\n')
        for key in ('pytest', 'updater', 'dependencies'):
            summary.write(f'- {key} — {"run" if result[key] else "skip"}\n')
        for row in result['matrix']['include']:
            summary.write(f'- recognition CPU — {row["os"]}, {row["suite"]}\n')
        summary.write('\n' + '\n'.join(result['reasons']) + '\n')


if __name__ == '__main__':
    main()
