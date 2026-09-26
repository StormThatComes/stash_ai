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
            "stash_merge.py": b"VALUE = 2\n",
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
        if endpoint == "git/trees/" + TREE:
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
    commit, files, requirements = remote.snapshot()
    for name in files:
        (root / name).write_bytes(remote.files[name])
    updater.atomic_write(root / updater.STATE, updater.json_bytes({
        "format": 1, "commit": commit, "baseline": files,
        "upstream": files, "requirements": requirements}))
    remote.calls.clear()


def stage(root, remote, **kwargs):
    with updater.InstallationLock(root, exclusive=True):
        state = updater.load_state(root)
        commit, files, requirements = remote.snapshot()
        actions, conflicts, _, next_state = updater.plan_update(
            root, state, commit, files, requirements, **kwargs)
        assert not conflicts
        return updater.prepare(root, actions, next_state, remote)


def snapshot(root):
    return {p.name: p.read_bytes() for p in root.iterdir() if p.is_file() and p.name != updater.LOCK}


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
    project = Path(updater.__file__).parent
    files = {path.name: path.read_bytes() for path in project.glob('stash_*.py')}
    files[updater.UPDATER] = Path(updater.__file__).read_bytes()
    assert updater.update(tmp_path, Remote(files)) == 0
    for name in ('stash_domain.py', 'stash_client.py', 'stash_tags.py', 'stash_state_codec.py', 'stash_state.py'):
        assert (tmp_path / name).read_bytes() == files[name]
    # The updater CI matrix installs only pytest, so --help must remain lightweight.
    result = subprocess.run([sys.executable, str(tmp_path / 'stash_ai.py'), '--help'],
                            cwd=tmp_path, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert '--abandon-pending' in result.stdout
    assert '--cleanup-only' in result.stdout


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
    del remote.files["stash_merge.py"]
    del remote.files["marker_core_exclusions.yml"]
    assert updater.update(tmp_path, remote) == 0
    assert not (tmp_path / "stash_merge.py").exists()
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


@pytest.mark.parametrize("problem", ["truncated", "symlink", "case_collision", "missing_updater"])
def test_bad_tree_is_rejected_before_downloading(tmp_path, problem):
    remote = Remote()
    def mutate(tree):
        if problem == "truncated":
            tree["truncated"] = True
        elif problem == "symlink":
            tree["tree"][0]["mode"] = "120000"
        elif problem == "case_collision":
            entry = dict(tree["tree"][1], path="stash_MERGE.py")
            tree["tree"].append(entry)
        else:
            tree["tree"] = [e for e in tree["tree"] if e["path"] != updater.UPDATER]
        return tree
    remote.mutate_tree = mutate
    with pytest.raises(updater.UpdateError):
        updater.update(tmp_path, remote)
    assert not any(c.startswith("git/blobs/") for c in remote.calls)


@pytest.mark.parametrize("path", ["stash_ai.py", updater.META, "tagger_models.yml.incoming"])
def test_local_symlinks_are_not_followed(tmp_path, path):
    external = tmp_path / "external"
    if path == updater.META:
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
        calls.append(entry["path"])
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
    _, _, entries = updater.load_transaction(tmp_path)
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


def test_file_tree_is_pinned_to_one_commit(tmp_path):
    remote = Remote()
    stage(tmp_path, remote)
    assert remote.calls[:2] == ["commits/main", "git/trees/" + TREE]
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


def test_stale_bytecode_is_removed_before_new_sources(tmp_path):
    remote = Remote()
    installed(tmp_path, remote)
    cache = tmp_path / "__pycache__"
    cache.mkdir()
    (cache / "stash_ai.cpython-311.pyc").write_bytes(b"old cache")
    (cache / "user_custom.cpython-311.pyc").write_bytes(b"user cache")
    remote.files["stash_ai.py"] = b"NEW = 1\n"
    assert updater.update(tmp_path, remote) == 0
    assert not (cache / "stash_ai.cpython-311.pyc").exists()
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
    (tmp_path / name).write_bytes(old)
    if tracked:
        state = updater.load_state(tmp_path)
        for key in ("baseline", "upstream"):
            state[key][name] = updater.blob_sha(old)
        updater.atomic_write(tmp_path / updater.STATE, updater.json_bytes(state))
    before = snapshot(tmp_path)
    assert updater.update(tmp_path, remote, dry_run=True) == (0 if tracked else 2)
    assert snapshot(tmp_path) == before
    if not tracked:
        with pytest.raises(updater.UpdateError, match="force-code"):
            updater.update(tmp_path, remote)
        assert snapshot(tmp_path) == before
    assert updater.update(tmp_path, remote, force_code=not tracked) == 0
    assert not (tmp_path / name).exists()
    assert all(name not in updater.load_state(tmp_path)[key] for key in ("baseline", "upstream"))
    assert old in [p.read_bytes() for p in (tmp_path / updater.META / "transactions").glob("*/old-*")]
    assert updater.blob_sha(remote.files[name]) not in {
        c.split("/")[-1] for c in remote.calls if c.startswith("git/blobs/")}


def test_interrupted_retired_file_removal_can_roll_back(tmp_path):
    remote = Remote()
    installed(tmp_path, remote)
    name = "joytag_stash_markers.py"
    old = b"# old launcher\n"
    (tmp_path / name).write_bytes(old)
    identity, directory = stage(tmp_path, remote, force_code=True)
    updater.journal_write(tmp_path, identity, "applying")
    (tmp_path / name).unlink()
    assert updater.recover(tmp_path) == 0
    assert (tmp_path / name).read_bytes() == old
