"""Offline API fixtures plus real filesystem, helper and process-lock tests."""
import base64
import io
from pathlib import Path
import subprocess
import sys
from urllib.error import HTTPError

import pytest

import update_stash_ai as updater


COMMIT = "a" * 40
TREE = "b" * 40


class Remote(updater.GitHub):
    def __init__(self, files=None):
        self.files = files or {
            "stash_ai.py": b"print('new application')\n",
            "stash_ai/__init__.py": b"",
            "stash_ai/cli.py": b"def main(): return 0\n",
            "stash_ai/merge.py": b"VALUE = 2\n",
            "joytag_stash_markers.py": b"RETIRED LAUNCHER",
            "stash_refine.py": b"RETIRED MODULE",
            updater.UPDATER: Path(updater.__file__).read_bytes(),
            "tagger_models.yml": b"device: cuda\n",
            "marker_core_exclusions.yml": b"excluded_core_tags: []\n",
            "test_updater.py": b"DO NOT DOWNLOAD TESTS",
            "benchmark_merge.py": b"DO NOT DOWNLOAD BENCHMARKS",
            ".github/workflows/tests.yml": b"DO NOT DOWNLOAD WORKFLOWS",
            "requirements.txt": b"DO NOT DOWNLOAD REQUIREMENTS",
            "README.md": b"DO NOT DOWNLOAD DOCS",
        }
        self.commit = COMMIT
        self.calls = []
        self.mutate_tree = lambda value: value
        self.corrupt = False

    def get(self, endpoint):
        self.calls.append(endpoint)
        if endpoint == "commits/main":
            return {"sha": self.commit, "commit": {"tree": {"sha": TREE}}}
        if endpoint == "git/trees/" + TREE + "?recursive=1":
            return self.mutate_tree({"truncated": False, "tree": [
                {"path": name, "type": "blob", "mode": "100644", "size": len(data),
                 "sha": updater.blob_sha(data)} for name, data in self.files.items()]})
        if endpoint.startswith("git/blobs/"):
            for data in self.files.values():
                if updater.blob_sha(data) == endpoint.split("/")[-1]:
                    return {"encoding": "base64", "content": base64.b64encode(
                        b"corrupted" if self.corrupt else data).decode()}
        pytest.fail(f"Unexpected API request: {endpoint}")


def installed(root, remote):
    version = remote.snapshot()
    for name in version.files:
        (root / name).parent.mkdir(parents=True, exist_ok=True)
        (root / name).write_bytes(remote.files[name])
    updater.atomic_write(root / updater.STATE, updater.json_bytes({
        "format": 1, "commit": version.commit, "baseline": version.files,
        "upstream": version.files, "requirements": version.requirements}))
    remote.calls.clear()


def stage(root, remote, **kwargs):
    with updater.InstallationLock(root, exclusive=True):
        state = updater.load_state(root)
        plan = updater.plan_update(root, state, remote.snapshot(), **kwargs)
        assert not plan.conflicts
        prepared = updater.prepare(root, plan, remote)
        return prepared.identity, prepared.directory


def snapshot(root):
    return {p.relative_to(root).as_posix(): p.read_bytes() for p in root.rglob('*')
            if p.is_file() and p.name != updater.LOCK
            and updater.META not in p.parts and '__pycache__' not in p.parts}


def application_files():
    project = Path(updater.__file__).parent
    files = {path.relative_to(project).as_posix(): path.read_bytes()
             for path in (project / 'stash_ai').rglob('*.py')}
    for name in ('stash_ai.py', updater.UPDATER):
        files[name] = (project / name).read_bytes()
    return files


def flat_installation(root):
    files = {name: ('# previous ' + name + '\n').encode() for name in updater.LEGACY_MODULES}
    files['stash_ai.py'] = b"print('previous application')\n"
    files[updater.UPDATER] = b'# previous updater\n'
    for name, data in files.items():
        (root / name).write_bytes(data)
    baseline = {name: updater.blob_sha(data) for name, data in files.items()}
    updater.atomic_write(root / updater.STATE, updater.json_bytes({
        'format': 1, 'commit': 'c' * 40, 'baseline': baseline,
        'upstream': baseline, 'requirements': None}))
    return files


def test_real_helper_selective_install_and_self_update(tmp_path):
    remote = Remote()
    (tmp_path / "tagger_sources.sqlite3").write_bytes(b"pending operations and raw inference")
    (tmp_path / ".env").write_bytes(b"local credentials")
    (tmp_path / "models").mkdir()
    (tmp_path / "models/weights.bin").write_bytes(b"weights")
    assert updater.update(tmp_path, remote) == 0
    expected = {name for name in remote.files if updater.allowed(name)}
    assert expected.issubset(snapshot(tmp_path))
    assert not {"test_updater.py", "README.md", "requirements.txt", "benchmark_merge.py"} & snapshot(tmp_path).keys()
    assert not (tmp_path / ".github").exists()
    downloaded = {call.split("/")[-1] for call in remote.calls if call.startswith("git/blobs/")}
    assert downloaded == {updater.blob_sha(remote.files[name]) for name in expected}
    assert (tmp_path / "tagger_sources.sqlite3").read_bytes() == b"pending operations and raw inference"
    assert (tmp_path / ".env").read_bytes() == b"local credentials"
    assert (tmp_path / "models/weights.bin").read_bytes() == b"weights"
    assert updater.load_state(tmp_path)["commit"] == COMMIT
    remote.files[updater.UPDATER] += b"\n# self-update integration revision\n"
    remote.commit = "c" * 40
    assert updater.update(tmp_path, remote) == 0
    assert (tmp_path / updater.UPDATER).read_bytes() == remote.files[updater.UPDATER]
    result = subprocess.run([sys.executable, str(tmp_path / updater.UPDATER), "--recover"],
                            capture_output=True, text=True)
    assert result.returncode == 0 and "No incomplete" in result.stdout
    remote.calls.clear()
    assert updater.update(tmp_path, remote) == 0
    assert not any(c.startswith("git/blobs/") for c in remote.calls)


def test_installed_application_loads_extracted_modules_without_runtime_dependencies(tmp_path):
    files = application_files()
    assert updater.update(tmp_path, Remote(files)) == 0
    for name in files:
        assert (tmp_path / name).read_bytes() == files[name]
    # The updater CI matrix installs only pytest, so --help must remain lightweight.
    result = subprocess.run([sys.executable, str(tmp_path / 'stash_ai.py'), '--help'],
                            cwd=tmp_path, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert '--abandon-pending' in result.stdout
    assert '--cleanup-only' in result.stdout


def database_module_installation(root):
    files = {name: data for name, data in application_files().items()
             if not name.startswith('stash_ai/db/')}
    files['stash_ai/db.py'] = b'LEGACY_DATABASE_MODULE = True\n'
    installed(root, Remote(files))


def test_database_module_moves_to_package_and_remains_importable(tmp_path):
    database_module_installation(tmp_path)
    result = subprocess.run(
        [sys.executable, '-c', 'from stash_ai.db import LEGACY_DATABASE_MODULE'],
        cwd=tmp_path, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    cache = tmp_path / 'stash_ai/__pycache__'
    assert list(cache.glob('db.*.pyc'))

    assert updater.update(tmp_path, Remote(application_files())) == 0

    assert not (tmp_path / 'stash_ai/db.py').exists()
    assert not list(cache.glob('db.*.pyc'))
    # Exercise the existing API through the installed package with stdlib only.
    result = subprocess.run([sys.executable, '-c', '''
from stash_ai.db import check_database, compact_database, inspect_database
from stash_ai.reports.developer.report import build_report
for operation, args in (
    (check_database, ('missing.sqlite3', 'scope')),
    (compact_database, ('missing.sqlite3', 'scope')),
    (inspect_database, ('missing.sqlite3', 'scope')),
    (build_report, ('missing.sqlite3',)),
):
    try:
        operation(*args)
    except FileNotFoundError:
        pass
    else:
        raise AssertionError('Missing database must be rejected')
'''], cwd=tmp_path, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert not (tmp_path / 'missing.sqlite3').exists()


def test_reports_move_to_nested_package_and_remove_old_bytecode(tmp_path):
    files = application_files()
    previous = {name: data for name, data in files.items()
                if not name.startswith('stash_ai/reports/')}
    modules = ('cache_report', 'tag_review', 'model_report', 'developer_report')
    previous.update({f'stash_ai/{name}.py': b'PREVIOUS_REPORT = True\n' for name in modules})
    installed(tmp_path, Remote(previous))
    result = subprocess.run(
        [sys.executable, '-c', 'from stash_ai import ' + ', '.join(modules)],
        cwd=tmp_path, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    cache = tmp_path / 'stash_ai/__pycache__'
    for name in modules:
        assert list(cache.glob(name + '.*.pyc'))

    assert updater.update(tmp_path, Remote(files)) == 0

    for name in modules:
        assert not (tmp_path / f'stash_ai/{name}.py').exists()
        assert not list(cache.glob(name + '.*.pyc'))
    for name, data in files.items():
        if name.startswith('stash_ai/reports/'):
            assert (tmp_path / name).read_bytes() == data
    result = subprocess.run([sys.executable, '-c', '''
from stash_ai.reports.developer.report import build_report
try:
    build_report('missing.sqlite3')
except FileNotFoundError:
    pass
else:
    raise AssertionError('Missing database must be rejected')
'''], cwd=tmp_path, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert not (tmp_path / 'missing.sqlite3').exists()


@pytest.mark.parametrize('name', ['developer', 'model', 'tag_review'])
def test_report_subpackages_replace_modules_and_stale_bytecode(tmp_path, name):
    files = application_files()
    old_path = f'stash_ai/reports/{name}.py'
    target = 'review' if name == 'tag_review' else name
    previous = {path: data for path, data in files.items()
                if not path.startswith(f'stash_ai/reports/{target}/')}
    previous[old_path] = b'PREVIOUS_REPORT = True\n'
    installed(tmp_path, Remote(previous))
    result = subprocess.run(
        [sys.executable, '-c', f'import stash_ai.reports.{name}'],
        cwd=tmp_path, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    cache = tmp_path / 'stash_ai/reports/__pycache__'
    assert list(cache.glob(name + '.*.pyc'))

    assert updater.update(tmp_path, Remote(files)) == 0

    assert not (tmp_path / old_path).exists()
    assert not list(cache.glob(name + '.*.pyc'))
    result = subprocess.run(
        [sys.executable, '-I', '-c',
         f"import sys; sys.path.insert(0, '.'); import stash_ai.reports.{target}"],
        cwd=tmp_path, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def test_temporal_modules_move_to_evidence_and_remove_old_bytecode(tmp_path):
    files = application_files()
    previous = {path: data for path, data in files.items()
                if not path.startswith('stash_ai/evidence/')}
    modules = ('temporal', 'timeline', 'retention')
    previous.update({f'stash_ai/{name}.py': b'PREVIOUS_EVIDENCE = True\n' for name in modules})
    installed(tmp_path, Remote(previous))
    result = subprocess.run(
        [sys.executable, '-c', 'from stash_ai import ' + ', '.join(modules)],
        cwd=tmp_path, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    cache = tmp_path / 'stash_ai/__pycache__'
    for name in modules:
        assert list(cache.glob(name + '.*.pyc'))

    assert updater.update(tmp_path, Remote(files)) == 0

    for name in modules:
        assert not (tmp_path / f'stash_ai/{name}.py').exists()
        assert not list(cache.glob(name + '.*.pyc'))
    result = subprocess.run([sys.executable, '-I', '-c', '''
import sys
sys.path.insert(0, '.')
from stash_ai.evidence.temporal.policy import TemporalPolicy, marker_floor
from stash_ai.evidence.temporal.statistics import SceneAccumulator
from stash_ai.evidence import timeline, retention
policy = TemporalPolicy()
policy.validate()
assert marker_floor(policy, 0.8) == 0.8 * policy.marker_continue_ratio
assert SceneAccumulator(0.8, policy).state()['frames'] == 0
'''], cwd=tmp_path, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def test_evidence_pipeline_replaces_root_modules_and_imports_without_ml(tmp_path):
    files = application_files()
    previous = {path: data for path, data in files.items()
                if not path.startswith('stash_ai/evidence/')}
    modules = ('merge', 'fusion', 'scene', 'occurrences', 'scene_cache', 'thresholds')
    previous.update({f'stash_ai/{name}.py': b'PREVIOUS_EVIDENCE = True\n' for name in modules})
    installed(tmp_path, Remote(previous))
    result = subprocess.run(
        [sys.executable, '-c', 'from stash_ai import ' + ', '.join(modules)],
        cwd=tmp_path, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    cache = tmp_path / 'stash_ai/__pycache__'

    assert updater.update(tmp_path, Remote(files)) == 0

    for name in modules:
        assert not (tmp_path / f'stash_ai/{name}.py').exists()
        assert not list(cache.glob(name + '.*.pyc'))
    result = subprocess.run([sys.executable, '-I', '-c', '''
import sys
sys.path.insert(0, '.')
from stash_ai.evidence.pipeline import evidence
from stash_ai.evidence.frames import FrameIndex
from stash_ai.evidence.observations import ObservationIndex
from stash_ai.evidence.labels import raw_label
from stash_ai.evidence.identity import marker_segment_signature, OVERRIDE, SCENE_OVERRIDE
from stash_ai.inference.reuse import reuse
segment = {'mode': 'frames', 'meta': {}, 'items': []}
projected = {**segment, OVERRIDE: 0.2, SCENE_OVERRIDE: 0.7}
assert marker_segment_signature(segment) == marker_segment_signature(projected)
assert raw_label('plain_label') == 'plain_label'
assert not {'torch', 'numpy', 'onnxruntime'} & set(sys.modules)
'''], cwd=tmp_path, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def test_runtime_package_replaces_module_and_imports_with_stdlib_only(tmp_path):
    files = application_files()
    previous = {path: data for path, data in files.items()
                if not path.startswith('stash_ai/runtime/')}
    previous['stash_ai/runtime.py'] = b'PREVIOUS_RUNTIME = True\n'
    previous['stash_ai/diagnostics.py'] = b'PREVIOUS_DIAGNOSTICS = True\n'
    installed(tmp_path, Remote(previous))
    result = subprocess.run([sys.executable, '-c', 'from stash_ai import runtime, diagnostics'],
                            cwd=tmp_path, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    cache = tmp_path / 'stash_ai/__pycache__'
    assert list(cache.glob('runtime.*.pyc'))

    assert updater.update(tmp_path, Remote(files)) == 0

    for name in ('runtime', 'diagnostics'):
        assert not (tmp_path / f'stash_ai/{name}.py').exists()
        assert not list(cache.glob(name + '.*.pyc'))
    result = subprocess.run([sys.executable, '-I', '-c', '''
import sys
sys.path.insert(0, '.')
from stash_ai.runtime.cache import RamCache
from stash_ai.runtime.report import RunReport
from stash_ai.runtime.diagnostics import MergeDiagnostics
cache = RamCache(4096)
cache.put('key', [1, 2])
assert cache.get('key') == [1, 2]
messages = []
report = RunReport(write=messages.append)
with report.measure('stdlib'):
    pass
report.finish('Done', {'processed': 1}, cache)
assert report.timing_calls['stdlib'] == 1
assert messages[0] == 'Done'
assert not {'torch', 'numpy', 'requests'} & set(sys.modules)
'''], cwd=tmp_path, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def test_inference_module_moves_to_package_and_removes_stale_inputs(tmp_path):
    files = application_files()
    previous = {name: data for name, data in files.items()
                if not name.startswith('stash_ai/inference/')}
    modules = ('inference', 'inference_inputs')
    previous.update({f'stash_ai/{name}.py': b'PREVIOUS_INFERENCE = True\n' for name in modules})
    installed(tmp_path, Remote(previous))
    result = subprocess.run(
        [sys.executable, '-c', 'from stash_ai import ' + ', '.join(modules)],
        cwd=tmp_path, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    cache = tmp_path / 'stash_ai/__pycache__'
    for name in modules:
        assert list(cache.glob(name + '.*.pyc'))

    assert updater.update(tmp_path, Remote(files)) == 0

    for name in modules:
        assert not (tmp_path / f'stash_ai/{name}.py').exists()
        assert not list(cache.glob(name + '.*.pyc'))
    for name, data in files.items():
        if name.startswith('stash_ai/inference/'):
            assert (tmp_path / name).read_bytes() == data
    # All module imports and canonical hashing work with the updater's stdlib-only runtime.
    result = subprocess.run([sys.executable, '-c', '''
from stash_ai.inference import checkpoints, identity, inputs, prediction, preparation, sprites
from stash_ai.signatures import digest
assert digest({}) == '44136fa355b3678a1146ad16f7e8649e94fb4fc21fe77e8310c060f61caaff8a'
'''], cwd=tmp_path, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def test_cli_commands_move_to_package_and_remove_stale_bytecode(tmp_path):
    files = application_files()
    previous = {name: data for name, data in files.items()
                if not name.startswith('stash_ai/commands/')}
    modules = ('cli_args', 'cli_modes')
    previous.update({f'stash_ai/{name}.py': b'PREVIOUS_COMMANDS = True\n' for name in modules})
    installed(tmp_path, Remote(previous))
    result = subprocess.run(
        [sys.executable, '-c', 'from stash_ai import ' + ', '.join(modules)],
        cwd=tmp_path, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    cache = tmp_path / 'stash_ai/__pycache__'
    for name in modules:
        assert list(cache.glob(name + '.*.pyc'))

    assert updater.update(tmp_path, Remote(files)) == 0

    for name in modules:
        assert not (tmp_path / f'stash_ai/{name}.py').exists()
        assert not list(cache.glob(name + '.*.pyc'))
    for name, data in files.items():
        if name.startswith('stash_ai/commands/'):
            assert (tmp_path / name).read_bytes() == data
    # Database dispatch must be importable before optional runtime dependencies.
    result = subprocess.run([sys.executable, '-c', '''
from stash_ai.commands import arguments, dispatch, validation
assert callable(arguments.parse_args)
assert callable(dispatch.run_database_command)
assert callable(validation.validate_reports)
'''], cwd=tmp_path, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def test_run_modules_move_to_package_and_remove_stale_bytecode(tmp_path):
    files = application_files()
    previous = {name: data for name, data in files.items()
                if not name.startswith('stash_ai/runs/')}
    modules = ('merge_run', 'separate_run')
    previous.update({f'stash_ai/{name}.py': b'PREVIOUS_RUN = True\n' for name in modules})
    installed(tmp_path, Remote(previous))
    result = subprocess.run(
        [sys.executable, '-c', 'from stash_ai import ' + ', '.join(modules)],
        cwd=tmp_path, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    cache = tmp_path / 'stash_ai/__pycache__'
    for name in modules:
        assert list(cache.glob(name + '.*.pyc'))

    assert updater.update(tmp_path, Remote(files)) == 0

    for name in modules:
        assert not (tmp_path / f'stash_ai/{name}.py').exists()
        assert not list(cache.glob(name + '.*.pyc'))
    for name, data in files.items():
        if name.startswith('stash_ai/runs/'):
            assert (tmp_path / name).read_bytes() == data
    # Package installation must preserve the CLI's dependency-free help path.
    result = subprocess.run([sys.executable, 'stash_ai.py', '--help'], cwd=tmp_path,
                            capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert '--merge-only' in result.stdout
    assert '--output-mode' not in result.stdout


@pytest.mark.parametrize('package,modules', [
    ('publication', ('merge_sync', 'separate_sync', 'merge_publish', 'merge_parallel')),
    ('cleanup', ('merge_cleanup', 'cleanup', 'cleanup_sql', 'cleanup_sql_session', 'prune', 'legacy_prune')),
    ('storage', ('state', 'state_db', 'state_results', 'state_publications', 'state_codec',
                 'state_metadata', 'state_snapshot', 'state_thresholds', 'preview', 'progress')),
    ('backends', ('taggers', 'onnx_tagger', 'joytag', 'vocabulary')),
    ('stash_api', ('client', 'transport', 'merge_sql', 'tags')),
])
def test_domain_modules_move_to_packages_and_remove_stale_bytecode(tmp_path, package, modules):
    files = application_files()
    prefix = f'stash_ai/{package}/'
    previous = {name: data for name, data in files.items() if not name.startswith(prefix)}
    previous.update({f'stash_ai/{name}.py': b'PREVIOUS_MODULE = True\n' for name in modules})
    installed(tmp_path, Remote(previous))
    result = subprocess.run(
        [sys.executable, '-c', 'from stash_ai import ' + ', '.join(modules)],
        cwd=tmp_path, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    cache = tmp_path / 'stash_ai/__pycache__'
    assert all(list(cache.glob(name + '.*.pyc')) for name in modules)

    assert updater.update(tmp_path, Remote(files)) == 0

    for name in modules:
        assert not (tmp_path / f'stash_ai/{name}.py').exists()
        assert not list(cache.glob(name + '.*.pyc'))
    for name, data in files.items():
        if name.startswith(prefix):
            assert (tmp_path / name).read_bytes() == data
    result = subprocess.run([sys.executable, 'stash_ai.py', '--help'], cwd=tmp_path,
                            capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert '--merge-only' in result.stdout
    assert '--output-mode' not in result.stdout


def test_installed_storage_preserves_publication_round_trip_without_runtime_dependencies(tmp_path):
    assert updater.update(tmp_path, Remote(application_files())) == 0
    result = subprocess.run([sys.executable, '-c', '''
from stash_ai.storage import codec, snapshots, progress
from stash_ai.storage.database import APPLICATION_ID
from stash_ai.storage.store import StateStore
snapshot = {'scene': {'owned_ids': [], 'contributions': []}}
assert codec.decode(codec.encode(snapshot)) == snapshot
assert snapshots.expand(snapshots.compact(snapshot['scene'])) == snapshot['scene']
assert callable(progress.completed)
with StateStore('state.sqlite3', 'fixture') as store:
    assert store.connection.execute('PRAGMA application_id').fetchone()[0] == APPLICATION_ID
    store.prepare('1', snapshot)
    assert store.snapshot('1', 'scene') == (None, snapshot['scene'])
    store.publish('1', snapshot)
with StateStore('state.sqlite3', 'fixture') as store:
    assert store.snapshot('1', 'scene') == (snapshot['scene'], None)
'''], cwd=tmp_path, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def test_failed_database_package_migration_restores_module(tmp_path, monkeypatch):
    database_module_installation(tmp_path)
    before = snapshot(tmp_path)
    previous_state = (tmp_path / updater.STATE).read_bytes()
    identity, _ = stage(tmp_path, Remote(application_files()))
    original = updater.put_entry

    def fail_after_removing_module(root, entry, data):
        original(root, entry, data)
        if entry.path == 'stash_ai/db.py' and data is None:
            raise OSError('database module migration interrupted')

    monkeypatch.setattr(updater, 'put_entry', fail_after_removing_module)
    with pytest.raises(OSError, match='database module migration interrupted'):
        updater.apply_transaction(tmp_path, identity)

    assert snapshot(tmp_path) == before
    assert (tmp_path / updater.STATE).read_bytes() == previous_state
    assert not (tmp_path / updater.JOURNAL).exists()
    result = subprocess.run(
        [sys.executable, '-c', 'from stash_ai.db import LEGACY_DATABASE_MODULE'],
        cwd=tmp_path, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def test_flat_installation_moves_modules_and_preserves_user_data(tmp_path):
    flat_installation(tmp_path)
    preserved = {'tagger_sources.sqlite3': b'cached inference', '.env': b'credentials',
                 'tagger_models.yml': b'local configuration', 'stash_custom.py': b'user script'}
    for name, data in preserved.items():
        (tmp_path / name).write_bytes(data)
    (tmp_path / 'models').mkdir()
    (tmp_path / 'models/weights.bin').write_bytes(b'weights')
    assert updater.update(tmp_path, Remote(application_files()), migrate_flat_layout=True) == 0
    assert not any((tmp_path / name).exists() for name in updater.LEGACY_MODULES)
    assert all((tmp_path / name).read_bytes() == data for name, data in preserved.items())
    assert (tmp_path / 'models/weights.bin').read_bytes() == b'weights'
    # Absolute invocation from another working directory still locates the package.
    result = subprocess.run([sys.executable, str(tmp_path / 'stash_ai.py'), '--help'],
                            cwd=tmp_path.parent, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def test_modified_legacy_module_requires_force_and_is_backed_up(tmp_path):
    flat_installation(tmp_path)
    local = b'# my modifications\n'
    (tmp_path / 'stash_merge.py').write_bytes(local)
    before = snapshot(tmp_path)
    remote = Remote()
    with pytest.raises(updater.UpdateError, match='force-code'):
        updater.update(tmp_path, remote, migrate_flat_layout=True)
    assert snapshot(tmp_path) == before
    assert updater.update(tmp_path, remote, force_code=True, migrate_flat_layout=True) == 0
    assert not (tmp_path / 'stash_merge.py').exists()
    assert local in [p.read_bytes() for p in (tmp_path / updater.META / 'transactions').glob('*/old-*')]


def test_failed_layout_migration_restores_flat_installation(tmp_path, monkeypatch):
    flat_installation(tmp_path)
    before = snapshot(tmp_path)
    previous_state = (tmp_path / updater.STATE).read_bytes()
    identity, _ = stage(tmp_path, Remote(), migrate_flat_layout=True)
    original = updater.put_entry
    failed = False

    def fail_first_package_file(root, entry, data):
        nonlocal failed
        if entry.path.startswith('stash_ai/') and not failed:
            failed = True
            raise OSError('package write failed')
        original(root, entry, data)

    monkeypatch.setattr(updater, 'put_entry', fail_first_package_file)
    with pytest.raises(OSError, match='package write failed'):
        updater.apply_transaction(tmp_path, identity)
    assert snapshot(tmp_path) == before
    assert (tmp_path / updater.STATE).read_bytes() == previous_state
    assert not (tmp_path / updater.JOURNAL).exists()


def test_root_only_upgrade_explains_second_pass_then_installs_package(tmp_path):
    # A pre-package updater sees only root files in the new commit. Reproduce
    # its completed first pass; the new updater must then install missing files
    # even though the recorded commit SHA already equals main.
    files = application_files()
    root_files = {name: data for name, data in files.items() if '/' not in name}
    for name, data in root_files.items():
        (tmp_path / name).write_bytes(data)
    baseline = {name: updater.blob_sha(data) for name, data in root_files.items()}
    updater.atomic_write(tmp_path / updater.STATE, updater.json_bytes({
        'format': 1, 'commit': COMMIT, 'baseline': baseline, 'upstream': baseline, 'requirements': None}))
    result = subprocess.run([sys.executable, str(tmp_path / 'stash_ai.py'), '--help'],
                            cwd=tmp_path, capture_output=True, text=True)
    assert result.returncode == 1
    assert '--migrate-flat-layout' in result.stderr and 'update_stash_ai.py' in result.stderr
    assert 'Traceback' not in result.stderr
    assert updater.update(tmp_path, Remote(files), migrate_flat_layout=True) == 0
    result = subprocess.run([sys.executable, str(tmp_path / 'stash_ai.py'), '--help'],
                            cwd=tmp_path, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize("option", ["check", "dry_run"])
def test_previews_never_download_or_write_installation(tmp_path, option):
    (tmp_path / "tagger_models.yml").write_bytes(b"device: cpu\n")
    remote = Remote()
    before = snapshot(tmp_path)
    assert updater.update(tmp_path, remote, **{option: True}) == 0
    assert snapshot(tmp_path) == before
    assert not (tmp_path / updater.META).exists()
    assert not any(c.startswith("git/blobs/") for c in remote.calls)


def test_local_yaml_and_first_adoption_are_preserved(tmp_path):
    remote = Remote()
    local = b"device: cpu\nmodel_dir: D:/my_models\n"
    (tmp_path / "tagger_models.yml").write_bytes(local)
    assert updater.update(tmp_path, remote) == 0
    assert (tmp_path / "tagger_models.yml").read_bytes() == local
    assert (tmp_path / "tagger_models.yml.incoming").read_bytes() == remote.files["tagger_models.yml"]
    assert "tagger_models.yml" not in updater.load_state(tmp_path)["baseline"]
    assert updater.update(tmp_path, remote, overwrite_configs=True) == 0
    assert (tmp_path / "tagger_models.yml").read_bytes() == remote.files["tagger_models.yml"]
    backups = (tmp_path / updater.META / "transactions").glob("*/old-*")
    assert local in [p.read_bytes() for p in backups]


def test_only_unmodified_yaml_updates_and_retains_merge_baseline(tmp_path):
    remote = Remote()
    installed(tmp_path, remote)
    original = updater.blob_sha(remote.files["tagger_models.yml"])
    (tmp_path / "tagger_models.yml").write_bytes(b"device: cpu\n")
    remote.files["tagger_models.yml"] = b"device: auto\n"
    remote.files["marker_core_exclusions.yml"] = b"excluded_core_tags: [new]\n"
    assert updater.update(tmp_path, remote) == 0
    assert (tmp_path / "tagger_models.yml").read_bytes() == b"device: cpu\n"
    assert updater.load_state(tmp_path)["baseline"]["tagger_models.yml"] == original
    assert (tmp_path / "marker_core_exclusions.yml").read_bytes() == remote.files["marker_core_exclusions.yml"]
    (tmp_path / "tagger_models.yml.incoming").write_bytes(b"edited proposal")
    remote.files["tagger_models.yml"] = b"device: new\n"
    with pytest.raises(updater.UpdateError, match="incoming"):
        updater.update(tmp_path, remote)
    assert (tmp_path / "tagger_models.yml.incoming").read_bytes() == b"edited proposal"


@pytest.mark.parametrize("tracked", [False, True])
def test_modified_and_untracked_code_requires_force(tmp_path, tracked):
    remote = Remote()
    if tracked:
        installed(tmp_path, remote)
    local = b"LOCAL_SETTING = 999\n"
    (tmp_path / "stash_ai.py").write_bytes(local)
    with pytest.raises(updater.UpdateError, match="force-code"):
        updater.update(tmp_path, remote)
    assert (tmp_path / "stash_ai.py").read_bytes() == local
    assert not any(c.startswith("git/blobs/") for c in remote.calls)
    assert updater.update(tmp_path, remote, force_code=True) == 0
    assert (tmp_path / "stash_ai.py").read_bytes() == remote.files["stash_ai.py"]


def test_only_owned_obsolete_code_is_removed(tmp_path):
    remote = Remote()
    installed(tmp_path, remote)
    (tmp_path / "stash_custom.py").write_bytes(b"user code")
    (tmp_path / "custom.yml").write_bytes(b"user config")
    del remote.files["stash_ai/merge.py"]
    del remote.files["marker_core_exclusions.yml"]
    assert updater.update(tmp_path, remote) == 0
    assert not (tmp_path / "stash_ai/merge.py").exists()
    assert (tmp_path / "marker_core_exclusions.yml").exists()
    assert (tmp_path / "stash_custom.py").read_bytes() == b"user code"
    assert (tmp_path / "custom.yml").read_bytes() == b"user config"


@pytest.mark.parametrize("failure", ["hash", "network", "syntax"])
def test_download_failure_never_changes_installed_files(tmp_path, failure, monkeypatch):
    remote = Remote()
    installed(tmp_path, remote)
    remote.files["stash_ai.py"] = b"NEW = 1\n"
    before = snapshot(tmp_path)
    state = (tmp_path / updater.STATE).read_bytes()
    if failure == "hash":
        remote.corrupt = True
    elif failure == "network":
        monkeypatch.setattr(remote, "blob", lambda sha: (_ for _ in ()).throw(updater.UpdateError("offline")))
    else:
        remote.files["stash_ai.py"] = b"def broken(\n"
    with pytest.raises(updater.UpdateError):
        updater.update(tmp_path, remote)
    assert snapshot(tmp_path) == before
    assert (tmp_path / updater.STATE).read_bytes() == state
    assert not (tmp_path / updater.JOURNAL).exists()


@pytest.mark.parametrize("problem", ["truncated", "symlink", "case_collision", "missing_updater", "missing_package"])
def test_bad_tree_is_rejected_before_downloading(tmp_path, problem):
    remote = Remote()
    def mutate(tree):
        if problem == "truncated":
            tree["truncated"] = True
        elif problem == "symlink":
            tree["tree"][0]["mode"] = "120000"
        elif problem == "case_collision":
            entry = dict(tree["tree"][3], path="stash_ai/MERGE.py")
            tree["tree"].append(entry)
        elif problem == 'missing_updater':
            tree["tree"] = [e for e in tree["tree"] if e["path"] != updater.UPDATER]
        else:
            tree['tree'] = [e for e in tree['tree'] if not e['path'].startswith('stash_ai/')]
        return tree
    remote.mutate_tree = mutate
    with pytest.raises(updater.UpdateError):
        updater.update(tmp_path, remote)
    assert not any(c.startswith("git/blobs/") for c in remote.calls)


@pytest.mark.parametrize("path", ["stash_ai.py", "stash_ai", updater.META, "tagger_models.yml.incoming"])
def test_local_symlinks_are_not_followed(tmp_path, path):
    external = tmp_path / "external"
    if path in (updater.META, 'stash_ai'):
        external.mkdir()
    else:
        external.write_bytes(b"outside")
    try:
        (tmp_path / path).symlink_to(external, target_is_directory=external.is_dir())
    except OSError:
        pytest.skip("OS does not permit creating symlinks")
    if path.endswith("incoming"):
        (tmp_path / "tagger_models.yml").write_bytes(b"local")
    with pytest.raises(updater.UpdateError, match="linked"):
        updater.update(tmp_path, Remote())


def test_replace_failure_rolls_back_code_configs_and_state(tmp_path, monkeypatch):
    remote = Remote()
    installed(tmp_path, remote)
    before = snapshot(tmp_path)
    previous_state = (tmp_path / updater.STATE).read_bytes()
    remote.files["stash_ai.py"] = b"NEW = 1\n"
    remote.files["tagger_models.yml"] = b"device: cpu\n"
    identity, _ = stage(tmp_path, remote)
    original = updater.put_entry
    calls = []
    def fail_once(root, entry, data):
        calls.append(entry.path)
        if len(calls) == 2:
            raise OSError("simulated disk error")
        original(root, entry, data)
    monkeypatch.setattr(updater, "put_entry", fail_once)
    with pytest.raises(OSError, match="disk"):
        updater.apply_transaction(tmp_path, identity)
    assert snapshot(tmp_path) == before
    assert (tmp_path / updater.STATE).read_bytes() == previous_state
    assert not (tmp_path / updater.JOURNAL).exists()


@pytest.mark.parametrize("phase", ["prepared", "applying", "committed"])
def test_recover_all_durable_phases_offline(tmp_path, phase, monkeypatch):
    remote = Remote()
    installed(tmp_path, remote)
    old_code = (tmp_path / "stash_ai.py").read_bytes()
    remote.files["stash_ai.py"] = b"NEW = 1\n"
    identity, directory = stage(tmp_path, remote)
    entries = updater.UpdateTransaction.load(tmp_path).entries
    if phase != "prepared":
        updater.journal_write(tmp_path, identity, "applying")
        selected = entries if phase == "committed" else entries[:1]
        for i, entry in enumerate(selected):
            updater.put_entry(tmp_path, entry, updater.checked_backup(tmp_path, directory, i, entry, "new"))
    updater.journal_write(tmp_path, identity, phase)
    monkeypatch.delenv("GH_TOKEN", raising=False)
    monkeypatch.delenv("GITHUB_TOKEN", raising=False)
    assert updater.main(["--install-dir", str(tmp_path), "--recover"]) == 0
    assert (tmp_path / "stash_ai.py").read_bytes() == (remote.files["stash_ai.py"] if phase == "committed" else old_code)
    assert not (tmp_path / updater.JOURNAL).exists()


def test_staging_handoff_protects_external_edits_and_app_start(tmp_path):
    identity, _ = stage(tmp_path, Remote())
    (tmp_path / "stash_ai.py").write_bytes(b"external edit")
    with pytest.raises(updater.UpdateError, match="incomplete"):
        updater.application_guard(tmp_path)
    with pytest.raises(updater.UpdateError, match="changed before"):
        updater.apply_transaction(tmp_path, identity)
    assert updater.recover(tmp_path) == 0
    assert (tmp_path / "stash_ai.py").read_bytes() == b"external edit"


def test_corrupt_backup_prevents_partial_recovery(tmp_path):
    remote = Remote()
    installed(tmp_path, remote)
    remote.files["stash_ai.py"] = b"NEW = 1\n"
    identity, directory = stage(tmp_path, remote)
    updater.journal_write(tmp_path, identity, "applying")
    (directory / "old-0").write_bytes(b"corrupted backup")
    before = snapshot(tmp_path)
    with pytest.raises(updater.UpdateError, match="damaged"):
        updater.recover(tmp_path)
    assert snapshot(tmp_path) == before and (tmp_path / updater.JOURNAL).exists()


@pytest.mark.parametrize("which", ["old", "new"])
@pytest.mark.parametrize("damage", ["missing", "corrupt"])
def test_late_staged_damage_prevents_any_installation(tmp_path, which, damage):
    remote = Remote()
    installed(tmp_path, remote)
    remote.files["stash_ai.py"] = b"NEW = 1\n"
    identity, directory = stage(tmp_path, remote)
    transaction = updater.UpdateTransaction.load(tmp_path)
    # Damage the final metadata backup, after all application-file backups.
    target = directory / f"{which}-{len(transaction.entries) - 1}"
    if damage == "missing":
        target.unlink()
    else:
        target.write_bytes(b"damaged")
    before = snapshot(tmp_path)
    previous_state = (tmp_path / updater.STATE).read_bytes()
    previous_journal = (tmp_path / updater.JOURNAL).read_bytes()
    with pytest.raises(updater.UpdateError, match="damaged"):
        updater.apply_transaction(tmp_path, identity)
    assert snapshot(tmp_path) == before
    assert (tmp_path / updater.STATE).read_bytes() == previous_state
    assert (tmp_path / updater.JOURNAL).read_bytes() == previous_journal


def test_wrong_helper_identity_preserves_prepared_transaction(tmp_path):
    stage(tmp_path, Remote())
    before = snapshot(tmp_path)
    journal = (tmp_path / updater.JOURNAL).read_bytes()
    with pytest.raises(updater.UpdateError, match="another transaction"):
        updater.apply_transaction(tmp_path, "0" * 32)
    assert snapshot(tmp_path) == before
    assert (tmp_path / updater.JOURNAL).read_bytes() == journal


def test_standalone_helper_applies_legacy_metadata_format(tmp_path):
    # Construct format 1 directly, independently of prepare/asdict.
    root = tmp_path / "installation"
    root.mkdir()
    identity = "1" * 32
    directory = root / updater.META / "transactions" / identity
    directory.mkdir(parents=True)
    old, new = b"OLD = True\n", b"NEW = True\n"
    state = updater.json_bytes({"format": 1, "commit": COMMIT,
                                "baseline": {"stash_ai.py": updater.blob_sha(new)},
                                "upstream": {"stash_ai.py": updater.blob_sha(new)},
                                "requirements": None})
    (root / "stash_ai.py").write_bytes(old)
    (directory / "old-0").write_bytes(old)
    (directory / "new-0").write_bytes(new)
    (directory / "new-1").write_bytes(state)
    (directory / "plan.json").write_bytes(updater.json_bytes({"format": 1, "entries": [
        {"path": "stash_ai.py", "before": updater.blob_sha(old),
         "after": updater.blob_sha(new), "mode": 0o644},
        {"path": updater.STATE, "before": None, "after": updater.blob_sha(state), "mode": 0o644},
    ]}))
    (root / updater.JOURNAL).write_bytes(updater.json_bytes(
        {"format": 1, "id": identity, "phase": "prepared"}))
    helper = tmp_path / "worker.py"
    helper.write_bytes(Path(updater.__file__).read_bytes())
    result = subprocess.run([sys.executable, "-I", str(helper), "--install-dir", str(root),
                             "--_apply", identity], cwd=tmp_path, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert (root / "stash_ai.py").read_bytes() == new
    assert (root / updater.STATE).read_bytes() == state
    assert not (root / updater.JOURNAL).exists()


def test_file_tree_is_pinned_to_one_commit(tmp_path):
    remote = Remote()
    stage(tmp_path, remote)
    assert remote.calls[:2] == ["commits/main", "git/trees/" + TREE + "?recursive=1"]
    assert all(c.startswith("git/blobs/") for c in remote.calls[2:])


def test_shared_applications_exclude_updater_across_processes(tmp_path):
    # Same test exercises LockFileEx on Windows and flock on POSIX.
    program = """
import sys
from pathlib import Path
from update_stash_ai import InstallationLock, UpdateError
try:
    with InstallationLock(Path(sys.argv[1]), exclusive=sys.argv[2] == 'write'):
        print('acquired')
except UpdateError:
    sys.exit(3)
"""
    def child(mode):
        return subprocess.run([sys.executable, "-c", program, str(tmp_path), mode],
                              cwd=Path(updater.__file__).parent, capture_output=True, text=True)
    with updater.InstallationLock(tmp_path):
        assert child("read").returncode == 0
        assert child("write").returncode == 3
        assert updater.update(tmp_path, Remote(), check=True) == 0
        with pytest.raises(updater.UpdateError, match="busy"):
            updater.update(tmp_path, Remote())
    with updater.InstallationLock(tmp_path, exclusive=True):
        assert child("read").returncode == 3
        assert child("write").returncode == 3
    assert child("write").returncode == 0  # Persistent lock file is not a stale lock.


def test_launcher_stops_before_runtime_imports_when_update_pending(tmp_path):
    root = Path(updater.__file__).parent
    for name in (updater.UPDATER, "stash_ai.py"):
        (tmp_path / name).write_bytes((root / name).read_bytes())
    updater.journal_write(tmp_path, "a" * 32, "applying")
    result = subprocess.run([sys.executable, str(tmp_path / "stash_ai.py"), "--help"],
                            capture_output=True, text=True)
    assert result.returncode == 1 and "--recover" in result.stderr
    assert "ModuleNotFoundError" not in result.stderr


@pytest.mark.parametrize('module', ['stash_ai.py', 'stash_ai/merge.py'])
def test_stale_bytecode_is_removed_before_new_sources(tmp_path, module):
    remote = Remote()
    installed(tmp_path, remote)
    cache = tmp_path / Path(module).parent / "__pycache__"
    cache.mkdir()
    stale = cache / (Path(module).stem + '.cpython-314.pyc')
    stale.write_bytes(b"old cache")
    (cache / "user_custom.cpython-311.pyc").write_bytes(b"user cache")
    remote.files[module] = b"NEW = 1\n"
    assert updater.update(tmp_path, remote) == 0
    assert not stale.exists()
    assert (cache / "user_custom.cpython-311.pyc").read_bytes() == b"user cache"


class Opener:
    def __init__(self, action):
        self.action = action
        self.requests = []

    def open(self, request, **kwargs):
        self.requests.append(request)
        return self.action(request)


@pytest.mark.parametrize("status", [401, 403, 404, 500])
def test_auth_errors_never_expose_token_or_response_body(status, monkeypatch):
    client = updater.GitHub("github_pat_PRIVATE_SECRET")
    def error(request):
        raise HTTPError(request.full_url, status, "secret response", {}, io.BytesIO(b"private body"))
    client.opener = Opener(error)
    monkeypatch.setattr(updater.time, "sleep", lambda seconds: None)
    with pytest.raises(updater.UpdateError) as failure:
        client.get("commits/main")
    assert "PRIVATE_SECRET" not in str(failure.value)
    assert "secret response" not in str(failure.value)
    assert "private body" not in str(failure.value)
    assert client.opener.requests[0].get_header("Authorization") == "Bearer github_pat_PRIVATE_SECRET"


def test_redirects_never_forward_credentials():
    with pytest.raises(updater.UpdateError, match="redirect"):
        updater.NoRedirect().redirect_request(None, None, 302, "", {}, "https://other.invalid")


@pytest.mark.parametrize("token", [None, "", "bad\r\nAuthorization: secret"])
def test_missing_or_malformed_token(token):
    with pytest.raises(updater.UpdateError) as failure:
        updater.GitHub(token)
    assert "Authorization: secret" not in str(failure.value)


def test_unavailable_helper_preserves_offline_recovery(tmp_path, monkeypatch):
    remote = Remote()
    def fail(*args, **kwargs):
        assert "GH_TOKEN" not in kwargs["env"] and "GITHUB_TOKEN" not in kwargs["env"]
        raise OSError("cannot spawn")
    monkeypatch.setenv("GH_TOKEN", "private")
    monkeypatch.setattr(updater.subprocess, "Popen", fail)
    with pytest.raises(OSError):
        updater.update(tmp_path, remote)
    assert (tmp_path / updater.JOURNAL).exists()
    assert not (tmp_path / "stash_ai.py").exists()
    assert updater.recover(tmp_path) == 0


@pytest.mark.parametrize("name", sorted(updater.RETIRED_FILES))
@pytest.mark.parametrize("tracked", [False, True])
def test_retired_files_are_removed_with_history_and_backups(tmp_path, name, tracked):
    remote = Remote()
    installed(tmp_path, remote)
    old = b"# retired installation file\n"
    (tmp_path / name).parent.mkdir(parents=True, exist_ok=True)
    (tmp_path / name).write_bytes(old)
    import py_compile
    bytecode = Path(py_compile.compile(str(tmp_path / name), doraise=True))
    assert bytecode.exists()
    if tracked:
        state = updater.load_state(tmp_path)
        for key in ("baseline", "upstream"):
            state[key][name] = updater.blob_sha(old)
        updater.atomic_write(tmp_path / updater.STATE, updater.json_bytes(state))
    before = snapshot(tmp_path)
    assert updater.update(tmp_path, remote, dry_run=True, migrate_flat_layout=not tracked) == (0 if tracked else 2)
    assert snapshot(tmp_path) == before
    if not tracked:
        with pytest.raises(updater.UpdateError, match="force-code"):
            updater.update(tmp_path, remote, migrate_flat_layout=True)
        assert snapshot(tmp_path) == before
    assert updater.update(tmp_path, remote, force_code=not tracked, migrate_flat_layout=not tracked) == 0
    assert not (tmp_path / name).exists()
    assert not bytecode.exists()
    assert all(name not in updater.load_state(tmp_path)[key] for key in ("baseline", "upstream"))
    assert old in [p.read_bytes() for p in (tmp_path / updater.META / "transactions").glob("*/old-*")]
    if name in remote.files:
        assert updater.blob_sha(remote.files[name]) not in {
            c.split("/")[-1] for c in remote.calls if c.startswith("git/blobs/")}


def test_interrupted_retired_file_removal_can_roll_back(tmp_path):
    remote = Remote()
    installed(tmp_path, remote)
    name = "joytag_stash_markers.py"
    old = b"# old launcher\n"
    (tmp_path / name).parent.mkdir(parents=True, exist_ok=True)
    (tmp_path / name).write_bytes(old)
    identity, directory = stage(tmp_path, remote, force_code=True, migrate_flat_layout=True)
    updater.journal_write(tmp_path, identity, "applying")
    (tmp_path / name).unlink()
    assert updater.recover(tmp_path) == 0
    assert (tmp_path / name).read_bytes() == old


def test_publication_planning_replaces_root_modules_without_ml(tmp_path):
    files = application_files()
    previous = {path: data for path, data in files.items()
                if not path.startswith('stash_ai/publication/planning/')
                and path not in ('stash_ai/publication/groups.py', 'stash_ai/publication/index.py')}
    previous.update({f'stash_ai/{name}.py': b'PREVIOUS_PUBLICATION = True\n'
                     for name in ('marker_groups', 'marker_index')})
    installed(tmp_path, Remote(previous))
    result = subprocess.run([sys.executable, '-c', 'from stash_ai import marker_groups, marker_index'],
                            cwd=tmp_path, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert updater.update(tmp_path, Remote(files)) == 0
    for name in ('marker_groups', 'marker_index'):
        assert not (tmp_path / f'stash_ai/{name}.py').exists()
        assert not list((tmp_path / 'stash_ai/__pycache__').glob(name + '.*.pyc'))
    result = subprocess.run([sys.executable, '-I', '-c', """
import sys
sys.path.insert(0, '.')
from stash_ai.publication.planning.types import MergeSyncPlan
from stash_ai.publication.planning.merged_evidence import prepare_sync_evidence
from stash_ai.publication.planning.merged_tags import plan_scene_tags
from stash_ai.publication.planning.merged_markers import plan_markers
from stash_ai.publication.groups import group_name, managed_group, group_description
from stash_ai.publication.index import MarkerIntervalIndex
from stash_ai.commands.marker_groups import run_backfill
from stash_ai.cleanup.groups import prune_groups
assert group_name('example') == 'm:example'
assert managed_group({'name': 'm:example', 'description': group_description('example')})
assert MarkerIntervalIndex([]).overlapping('example', {'seconds': 1}) == []
assert not {'torch', 'numpy', 'onnxruntime', 'tqdm'} & set(sys.modules)
"""], cwd=tmp_path, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def test_recovery_and_fingerprints_migrate_without_ml_or_working_directory_dependency(tmp_path):
    files = application_files()
    moved = ('stash_ai/storage/recovery.py', 'stash_ai/publication/recovery.py',
             'stash_ai/commands/recovery.py', 'stash_ai/publication/fingerprints.py',
             'stash_ai/storage/segments.py', 'stash_ai/storage/merge_checks.py')
    previous = {path: data for path, data in files.items() if path not in moved}
    previous.update({f'stash_ai/{name}.py': b'PREVIOUS_STATE = True\n'
                     for name in ('performance', 'recovery')})
    installed(tmp_path, Remote(previous))
    result = subprocess.run([sys.executable, '-c', 'from stash_ai import performance, recovery'],
                            cwd=tmp_path, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert updater.update(tmp_path, Remote(files)) == 0
    for name in ('performance', 'recovery'):
        assert not (tmp_path / f'stash_ai/{name}.py').exists()
        assert not list((tmp_path / 'stash_ai/__pycache__').glob(name + '.*.pyc'))
    result = subprocess.run([sys.executable, '-I', '-c', """
import os
import sys
from copy import deepcopy
sys.path.insert(0, os.path.abspath('.'))
from stash_ai.storage.segments import StoredSegment, segment_digest
from stash_ai.publication.fingerprints import publication_code
from stash_ai.storage.recovery import abandon
from stash_ai.publication.recovery import reconcile
from stash_ai.commands.recovery import run_only
before = publication_code()
os.chdir('stash_ai/publication')
assert publication_code() == before
borrowed = StoredSegment(meta={}, items=[])
original = segment_digest(borrowed)
detached = deepcopy(borrowed)
detached['items'].append({'new': True})
assert type(detached) is dict
assert segment_digest(detached) != original
assert segment_digest(borrowed) == original
assert not {'torch', 'numpy', 'onnxruntime', 'tqdm'} & set(sys.modules)
"""], cwd=tmp_path, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def test_extracted_bootstrap_and_scene_selection_install_without_ml(tmp_path):
    files = application_files()
    previous = {path: data for path, data in files.items()
                if path not in ('stash_ai/commands/bootstrap.py', 'stash_ai/inference/scene_predictions.py')}
    installed(tmp_path, Remote(previous))
    assert updater.update(tmp_path, Remote(files)) == 0
    result = subprocess.run([sys.executable, '-I', '-c', """
import sys
sys.path.insert(0, '.')
from stash_ai.commands.bootstrap import catalogs, policies, runtime_options
from stash_ai.evidence.historical_markers import replay_markers
from stash_ai.inference.prediction import predict_scene
from stash_ai.inference.scene_predictions import retained_scores, candidate_scores
from stash_ai.evidence.temporal.policy import TemporalPolicy
assert retained_scores([0.8, 0.2], ['visible', 'retained'], [{'tags': [('retained', 0.3)]}]) == [('retained', 0.2)]
assert candidate_scores([0.8, 0.2], ['visible', 'retained'], [0, 1]) == [('visible', 0.8), ('retained', 0.2)]
assert not {'torch', 'numpy', 'onnxruntime', 'tqdm'} & set(sys.modules)
"""], cwd=tmp_path, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize('name', ['stash_custom.py', 'stash_future_compat.py', 'stash_unknown.py'])
def test_unknown_flat_paths_are_rejected_in_history_and_recovery_plans(tmp_path, name):
    remote = Remote()
    installed(tmp_path, remote)
    state = updater.load_state(tmp_path)
    state['baseline'][name] = updater.blob_sha(b'user code')
    updater.atomic_write(tmp_path / updater.STATE, updater.json_bytes(state))
    with pytest.raises(updater.UpdateError, match='Invalid installed-file history'):
        updater.load_state(tmp_path)
    transaction = {'entries': [
        {'path': name, 'before': updater.blob_sha(b'user code'), 'after': None, 'mode': 0o644},
        {'path': updater.STATE, 'before': None, 'after': updater.blob_sha(b'state'), 'mode': 0o600},
    ]}
    with pytest.raises(updater.UpdateError, match='Invalid transaction plan'):
        updater.transaction_entries(tmp_path, transaction)
    assert not updater.historical_file(name)


def test_current_update_does_not_probe_untracked_flat_or_retired_files(tmp_path, monkeypatch):
    remote = Remote()
    installed(tmp_path, remote)
    untracked = updater.LEGACY_MODULES | updater.RETIRED_FILES
    for name in untracked:
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b'# user-owned file after migration\n')
    real_sha = updater.file_sha
    def managed_only(path):
        assert path.relative_to(tmp_path).as_posix() not in untracked
        return real_sha(path)
    monkeypatch.setattr(updater, 'file_sha', managed_only)
    state = updater.load_state(tmp_path)
    plan = updater.plan_update(tmp_path, state, remote.snapshot(), force_code=True)
    assert not plan.actions and not plan.conflicts
    assert updater.update(tmp_path, remote, force_code=True) == 0
    assert all((tmp_path / name).exists() for name in untracked)


def test_tracked_flat_layout_requires_explicit_migration_before_downloading(tmp_path):
    flat_installation(tmp_path)
    remote = Remote()
    before = snapshot(tmp_path)
    state = (tmp_path / updater.STATE).read_bytes()
    with pytest.raises(updater.UpdateError, match='--migrate-flat-layout'):
        updater.update(tmp_path, remote, force_code=True)
    assert snapshot(tmp_path) == before
    assert (tmp_path / updater.STATE).read_bytes() == state
    assert not (tmp_path / updater.JOURNAL).exists()
    assert not any(call.startswith('git/blobs/') for call in remote.calls)
    assert updater.update(tmp_path, remote, dry_run=True, migrate_flat_layout=True) == 0
    assert snapshot(tmp_path) == before
    assert (tmp_path / updater.STATE).read_bytes() == state


def test_explicit_migration_is_idempotent_and_keeps_unknown_user_scripts(tmp_path):
    flat_installation(tmp_path)
    custom = tmp_path / 'stash_custom.py'
    custom.write_bytes(b'USER_CODE = True\n')
    remote = Remote()
    assert updater.update(tmp_path, remote, migrate_flat_layout=True) == 0
    state = (tmp_path / updater.STATE).read_bytes()
    remote.calls.clear()
    assert updater.update(tmp_path, remote, migrate_flat_layout=True) == 0
    assert (tmp_path / updater.STATE).read_bytes() == state
    assert not any(call.startswith('git/blobs/') for call in remote.calls)
    assert custom.read_bytes() == b'USER_CODE = True\n'
    # Normal updates after migration do not reclaim a newly created same-name file.
    (tmp_path / 'stash_merge.py').write_bytes(b'NEW_USER_CODE = True\n')
    assert updater.update(tmp_path, remote, force_code=True) == 0
    assert (tmp_path / 'stash_merge.py').read_bytes() == b'NEW_USER_CODE = True\n'


def test_known_flat_recovery_journal_can_restore_deleted_module(tmp_path):
    flat_installation(tmp_path)
    original = (tmp_path / 'stash_merge.py').read_bytes()
    identity, directory = stage(tmp_path, Remote(), migrate_flat_layout=True)
    updater.journal_write(tmp_path, identity, updater.Phase.APPLYING)
    (tmp_path / 'stash_merge.py').unlink()
    assert updater.recover(tmp_path) == 0
    assert (tmp_path / 'stash_merge.py').read_bytes() == original



def test_cli_flat_migration_preview_passes_the_explicit_option(tmp_path, monkeypatch):
    flat_installation(tmp_path)
    remote = Remote()
    monkeypatch.setattr(updater, 'GitHub', lambda token: remote)
    before = snapshot(tmp_path)
    assert updater.main(['--install-dir', str(tmp_path), '--migrate-flat-layout', '--dry-run']) == 0
    assert snapshot(tmp_path) == before
    assert not any(call.startswith('git/blobs/') for call in remote.calls)
