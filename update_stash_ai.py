#!/usr/bin/env python
"""Selective, authenticated main-branch updater. Python 3.14.7, stdlib only."""
from __future__ import annotations

import argparse
import atexit
import base64
import hashlib
import json
import os
from pathlib import Path
import re
import runpy
import subprocess
import sys
import tempfile
import time
from urllib.error import HTTPError, URLError
from urllib.request import HTTPRedirectHandler, Request, build_opener
import uuid

REPOSITORY = "StormThatComes/stash_ai"
UPDATER = "update_stash_ai.py"
RETIRED_FILES = frozenset({"joytag_stash_markers.py", "stash_refine.py"})
LEGACY_MODULES = frozenset("stash_" + name + ".py" for name in (
    "cleanup", "cleanup_sql", "client", "db", "developer_report", "diagnostics", "domain",
    "exclusions", "fusion", "joytag", "marker_groups", "merge", "merge_sql", "model_report",
    "occurrences", "performance", "preview", "progress", "prune", "recovery", "runtime",
    "scene", "scene_cache", "state", "state_codec", "state_db", "taggers", "tags", "temporal",
    "thresholds", "timeline", "transport",
))
REQUIRED_FILES = frozenset({"stash_ai.py", UPDATER, "stash_ai/__init__.py", "stash_ai/cli.py"})
META = ".stash-ai-update"
STATE = META + "/state.json"
JOURNAL = META + "/journal.json"
LOCK = ".stash-ai.install.lock"
MAX_FILE_BYTES = 16 * 1024 * 1024
SHA = re.compile(r"[0-9a-f]{40}\Z")


class UpdateError(RuntimeError):
    pass


def allowed(name):
    """Install entry points, package Python files and root YAML configuration."""
    return bool(re.fullmatch(
        r"(?:stash_ai\.py|update_stash_ai\.py|[A-Za-z0-9][A-Za-z0-9_.-]*\.yml|"
        r"stash_ai/(?:[A-Za-z_][A-Za-z0-9_]*/)*[A-Za-z_][A-Za-z0-9_]*\.py)", name))


def historical_file(name):
    """Accept former flat modules in tracked history and recovery journals only."""
    return allowed(name) or name in RETIRED_FILES or bool(re.fullmatch(r"stash_[A-Za-z0-9_]+\.py", name))


def blob_sha(data):
    return hashlib.sha1(b"blob " + str(len(data)).encode() + b"\0" + data).hexdigest()


def safe_path(root, name):
    """Reject links, including Windows junctions, at every existing component."""
    parts = name.split("/")
    if not parts or any(p in ("", ".", "..") or "\\" in p or ":" in p for p in parts):
        raise UpdateError("Unsafe installation path.")
    path = root
    for part in parts:
        path = path / part
        if path.is_symlink() or (path.exists() and path.resolve() != path):
            raise UpdateError(f"Refusing linked installation path: {name}")
    return path


def read_bytes(path):
    try:
        return path.read_bytes()
    except FileNotFoundError:
        return None


def file_sha(path):
    data = read_bytes(path)
    return None if data is None else blob_sha(data)


def sync_dir(path):
    if os.name != "nt":
        fd = os.open(path, os.O_RDONLY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)


def atomic_write(path, data, mode=0o600):
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=".update-", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.chmod(temporary, mode)
        os.replace(temporary, path)
        sync_dir(path.parent)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def json_bytes(value):
    return (json.dumps(value, indent=2, sort_keys=True) + "\n").encode()


def read_json(path):
    raw = read_bytes(path)
    if raw is None:
        return None
    try:
        value = json.loads(raw)
    except (ValueError, UnicodeError) as exc:
        raise UpdateError(f"Invalid updater metadata: {path.name}") from exc
    if not isinstance(value, dict) or value.get("format") != 1:
        raise UpdateError(f"Unsupported updater metadata: {path.name}")
    return value


class InstallationLock:
    """Shared application locks; exclusive updater lock. Never unlink the file."""
    def __init__(self, root, *, exclusive=False):
        self.root = Path(root).resolve()
        self.exclusive = exclusive
        self.stream = None

    def __enter__(self):
        self.stream = safe_path(self.root, LOCK).open("a+b")
        try:
            if os.name == "nt":
                self._windows_lock()
            else:
                import fcntl
                fcntl.flock(self.stream.fileno(),
                            (fcntl.LOCK_EX if self.exclusive else fcntl.LOCK_SH) | fcntl.LOCK_NB)
        except OSError as exc:
            self.close()
            raise UpdateError("Installation is busy: stop the application or wait for the updater.") from exc
        return self

    def _windows_lock(self):
        import ctypes
        from ctypes import wintypes
        import msvcrt

        class Overlapped(ctypes.Structure):
            _fields_ = [("Internal", ctypes.c_size_t), ("InternalHigh", ctypes.c_size_t),
                        ("Offset", wintypes.DWORD), ("OffsetHigh", wintypes.DWORD),
                        ("hEvent", wintypes.HANDLE)]

        kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel.LockFileEx.argtypes = [wintypes.HANDLE, wintypes.DWORD, wintypes.DWORD,
                                     wintypes.DWORD, wintypes.DWORD, ctypes.POINTER(Overlapped)]
        kernel.LockFileEx.restype = wintypes.BOOL
        overlap = Overlapped()
        if not kernel.LockFileEx(msvcrt.get_osfhandle(self.stream.fileno()),
                                 1 | (2 if self.exclusive else 0), 0, 1, 0, ctypes.byref(overlap)):
            raise ctypes.WinError(ctypes.get_last_error())

    def close(self):
        if self.stream is not None:
            # Closing the handle releases both flock and LockFileEx locks.
            self.stream.close()
            self.stream = None

    def __exit__(self, *args):
        self.close()


def require_complete(root):
    if safe_path(root, JOURNAL).exists():
        raise UpdateError("An update is incomplete. Run python update_stash_ai.py --recover first.")


def application_guard(root):
    """Imported applications keep the lock for their entire process lifetime."""
    lock = InstallationLock(root).__enter__()
    try:
        require_complete(lock.root)
    except BaseException:
        lock.close()
        raise
    atexit.register(lock.close)
    return lock


def launch_application(filename):
    """Read the actual entry point again under lock, before application imports."""
    root = Path(filename).resolve().parent
    try:
        with InstallationLock(root):
            require_complete(root)
            if any(not safe_path(root, name).is_file() for name in ("stash_ai/__init__.py", "stash_ai/cli.py")):
                raise UpdateError("The stash_ai package is missing. If upgrading from the flat layout, "
                                  "run python update_stash_ai.py once more to finish the migration.")
            runpy.run_path(str(root / "stash_ai.py"), run_name="__main__",
                           init_globals={"_STASH_INSTALL_LOCKED": True})
    except UpdateError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    return 0


class NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        raise UpdateError("Unexpected GitHub redirect; refusing to forward the token.")


class GitHub:
    def __init__(self, token):
        if not token or not token.strip():
            raise UpdateError("Set GH_TOKEN (or GITHUB_TOKEN) with Contents: read access to stash_ai.")
        self.token = token.strip()
        if not re.fullmatch(r"[A-Za-z0-9_]+", self.token):
            raise UpdateError("Invalid GitHub token format; check GH_TOKEN/GITHUB_TOKEN.")
        self.opener = build_opener(NoRedirect())

    def get(self, endpoint):
        request = Request(f"https://api.github.com/repos/{REPOSITORY}/{endpoint}", headers={
            "Authorization": "Bearer " + self.token,
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
            "User-Agent": "stash-ai-updater",
        })
        for attempt in range(3):
            try:
                with self.opener.open(request, timeout=30) as response:
                    raw = response.read(MAX_FILE_BYTES * 2 + 1)
                if len(raw) > MAX_FILE_BYTES * 2:
                    raise UpdateError("GitHub response exceeds the download limit.")
                value = json.loads(raw)
                if not isinstance(value, dict):
                    raise UpdateError("Unexpected GitHub response.")
                return value
            except HTTPError as exc:
                retry = exc.code in (429, 500, 502, 503, 504)
                if retry and attempt < 2:
                    time.sleep(2 ** attempt)
                    continue
                if exc.code in (401, 403, 404):
                    raise UpdateError(f"GitHub HTTP {exc.code}: check token expiry, repository access "
                                      "and Contents: read permission (or API rate limits).") from None
                raise UpdateError(f"GitHub HTTP {exc.code}; installation was not downloaded completely.") from None
            except (URLError, TimeoutError, OSError):
                if attempt < 2:
                    time.sleep(2 ** attempt)
                    continue
                raise UpdateError("Could not reach GitHub after 3 attempts. Check network/proxy settings.") from None
            except (ValueError, UnicodeError):
                raise UpdateError("Invalid JSON from GitHub.") from None

    def snapshot(self):
        commit = self.get("commits/main")
        try:
            sha, tree_sha = commit["sha"], commit["commit"]["tree"]["sha"]
            if not SHA.fullmatch(sha) or not SHA.fullmatch(tree_sha):
                raise ValueError
            tree = self.get("git/trees/" + tree_sha + "?recursive=1")
            if tree.get("truncated") or not isinstance(tree.get("tree"), list):
                raise ValueError
            files, requirements = {}, None
            for entry in tree["tree"]:
                name = entry["path"]
                if name == "requirements.txt":
                    requirements = entry["sha"]  # Metadata only; never download this file.
                if not allowed(name):
                    continue
                if (entry["type"] != "blob" or entry["mode"] not in ("100644", "100755")
                        or not SHA.fullmatch(entry["sha"])
                        or not 0 <= entry["size"] <= MAX_FILE_BYTES):
                    raise ValueError
                if name.casefold() in {n.casefold() for n in files}:
                    raise ValueError
                files[name] = entry["sha"]
            if not REQUIRED_FILES.issubset(files):
                raise UpdateError("main does not contain the complete application package and updater.")
            return sha, files, requirements
        except (KeyError, TypeError, ValueError):
            raise UpdateError("Invalid or incomplete GitHub file tree; refusing partial installation.") from None

    def blob(self, sha):
        value = self.get("git/blobs/" + sha)
        try:
            if value["encoding"] != "base64":
                raise ValueError
            data = base64.b64decode("".join(value["content"].split()), validate=True)
            if len(data) > MAX_FILE_BYTES or blob_sha(data) != sha:
                raise ValueError
            return data
        except (KeyError, TypeError, ValueError):
            raise UpdateError("Downloaded file failed its Git blob hash check.") from None


def load_state(root):
    state = read_json(safe_path(root, STATE))
    if state is None:
        return {"format": 1, "commit": None, "baseline": {}, "upstream": {}, "requirements": None}
    try:
        for key in ("baseline", "upstream"):
            if not isinstance(state[key], dict):
                raise ValueError
            for name, sha in state[key].items():
                if not historical_file(name) or not SHA.fullmatch(sha):
                    raise ValueError
        if not SHA.fullmatch(state["commit"]):
            raise ValueError
    except (KeyError, TypeError, ValueError):
        raise UpdateError("Invalid installed-file history; refusing to overwrite local files.") from None
    return state


def plan_update(root, state, commit, remote, requirements, *, overwrite_configs=False, force_code=False):
    actions, conflicts, kept = [], [], []
    baseline = dict(state["baseline"])
    for name, sha in sorted(remote.items()):
        current = file_sha(safe_path(root, name))
        if current == sha:
            baseline[name] = sha
            continue
        modified = current is not None and current != baseline.get(name)
        if name.endswith(".yml") and modified and not overwrite_configs:
            kept.append(name)
            incoming = name + ".incoming"
            before = file_sha(safe_path(root, incoming))
            if before != sha:
                # Never clobber an edited proposal from an earlier update.
                if before is not None and before != state["upstream"].get(name):
                    conflicts.append(incoming)
                else:
                    actions.append({"path": incoming, "before": before, "after": sha})
            continue
        if modified and name.endswith(".py") and not force_code:
            conflicts.append(name)
            continue
        actions.append({"path": name, "before": current, "after": sha})
        baseline[name] = sha
    for name in sorted((set(baseline) | RETIRED_FILES | LEGACY_MODULES) - set(remote)):
        if name.endswith(".yml"):
            continue  # Retired configurations may still contain user settings.
        current = file_sha(safe_path(root, name))
        if current is not None:
            if current != baseline.get(name) and not force_code:
                conflicts.append(name)
                continue
            actions.append({"path": name, "before": current, "after": None})
        baseline.pop(name, None)
    next_state = {"format": 1, "commit": commit, "baseline": baseline,
                  "upstream": remote, "requirements": requirements}
    return actions, conflicts, kept, next_state


def journal_write(root, identity, phase):
    atomic_write(safe_path(root, JOURNAL), json_bytes({"format": 1, "id": identity, "phase": phase}))


def clear_bytecode(root, entries):
    for entry in entries:
        if entry["path"].endswith(".py"):
            module = Path(entry["path"])
            cache = safe_path(root, (module.parent / "__pycache__").as_posix())
            if not cache.exists():
                continue
            for path in cache.glob(module.stem + ".*.pyc"):
                safe_path(root, path.relative_to(root).as_posix()).unlink()


def prepare(root, actions, next_state, client):
    """Stage all downloads and backups before publishing the recovery journal."""
    identity = uuid.uuid4().hex
    directory = safe_path(root, META + "/transactions/" + identity)
    directory.mkdir(parents=True)
    entries = []
    # Updater last among code files; installed-version metadata last overall.
    ordered = sorted(actions, key=lambda entry: entry["path"] == UPDATER)
    state_data = json_bytes(next_state)
    ordered.append({"path": STATE, "before": file_sha(safe_path(root, STATE)),
                    "after": blob_sha(state_data)})
    for index, action in enumerate(ordered):
        target = safe_path(root, action["path"])
        old = read_bytes(target)
        if (blob_sha(old) if old is not None else None) != action["before"]:
            raise UpdateError(f"File changed while planning: {action['path']}")
        mode = target.stat().st_mode & 0o777 if old is not None else 0o644
        entry = dict(action, mode=mode)
        if old is not None:
            atomic_write(directory / f"old-{index}", old)
        if action["after"] is not None:
            data = state_data if action["path"] == STATE else client.blob(action["after"])
            if blob_sha(data) != action["after"]:
                raise UpdateError(f"Hash mismatch: {action['path']}")
            if action["path"].endswith(".py"):
                try:
                    compile(data, action["path"], "exec", dont_inherit=True)
                except (SyntaxError, ValueError):
                    raise UpdateError(f"Invalid Python for this interpreter: {action['path']}") from None
            atomic_write(directory / f"new-{index}", data)
        entries.append(entry)
    atomic_write(directory / "plan.json", json_bytes({"format": 1, "entries": entries}))
    # The helper is the currently trusted updater, with no application imports.
    atomic_write(directory / "worker.py", Path(__file__).read_bytes())
    sync_dir(directory.parent)
    journal_write(root, identity, "prepared")
    return identity, directory


def load_transaction(root, identity=None):
    journal = read_json(safe_path(root, JOURNAL))
    if journal is None:
        raise UpdateError("No incomplete update to process.")
    if (not re.fullmatch(r"[0-9a-f]{32}", str(journal.get("id", "")))
            or journal.get("phase") not in ("prepared", "applying", "committed")
            or (identity is not None and identity != journal["id"])):
        raise UpdateError("Invalid update journal or another transaction has replaced it.")
    directory = safe_path(root, META + "/transactions/" + journal["id"])
    plan = read_json(safe_path(root, directory.relative_to(root).as_posix() + "/plan.json"))
    try:
        entries = plan["entries"]
        names = set()
        if not isinstance(entries, list) or not entries or entries[-1]["path"] != STATE:
            raise ValueError
        for entry in entries:
            name = entry["path"]
            if not (name == STATE or historical_file(name)
                    or (name.endswith(".yml.incoming") and allowed(name[:-9]))):
                raise ValueError
            if name.casefold() in names:
                raise ValueError
            names.add(name.casefold())
            safe_path(root, name)
            for key in ("before", "after"):
                if entry[key] is not None and not SHA.fullmatch(entry[key]):
                    raise ValueError
            if not isinstance(entry["mode"], int) or not 0 <= entry["mode"] <= 0o777:
                raise ValueError
    except (KeyError, TypeError, ValueError):
        raise UpdateError("Invalid transaction plan; backups were retained.") from None
    return journal, directory, entries


def checked_backup(root, directory, index, entry, which):
    expected = entry["before" if which == "old" else "after"]
    if expected is None:
        return None
    name = (directory / f"{which}-{index}").relative_to(root).as_posix()
    data = read_bytes(safe_path(root, name))
    if data is None or blob_sha(data) != expected:
        raise UpdateError("Missing or damaged staged file/backup; transaction retained for recovery.")
    return data


def put_entry(root, entry, data):
    target = safe_path(root, entry["path"])
    if data is None:
        target.unlink(missing_ok=True)
        if target.parent.exists():
            sync_dir(target.parent)
    else:
        atomic_write(target, data, mode=0o600 if entry["path"] == STATE else entry["mode"])


def finish_journal(root):
    safe_path(root, JOURNAL).unlink()
    sync_dir(root / META)


def restore(root, directory, entries):
    # Validate every backup and destination before restoring any file.
    backups = [checked_backup(root, directory, i, e, "old") for i, e in enumerate(entries)]
    for entry in entries:
        if file_sha(safe_path(root, entry["path"])) not in (entry["before"], entry["after"]):
            raise UpdateError(f"Recovery found an external edit: {entry['path']}. Backups were retained.")
    clear_bytecode(root, entries)
    for entry, data in zip(entries, backups):
        put_entry(root, entry, data)
    finish_journal(root)


def apply_transaction(root, identity):
    with InstallationLock(root, exclusive=True):
        journal, directory, entries = load_transaction(root, identity)
        if journal["phase"] != "prepared":
            raise UpdateError("Transaction was already started; use --recover.")
        data = [checked_backup(root, directory, i, e, "new") for i, e in enumerate(entries)]
        for i, entry in enumerate(entries):
            checked_backup(root, directory, i, entry, "old")
            if file_sha(safe_path(root, entry["path"])) != entry["before"]:
                raise UpdateError(f"File changed before installation: {entry['path']}. Use --recover.")
        journal_write(root, identity, "applying")
        try:
            clear_bytecode(root, entries)
            for entry, content in zip(entries, data):
                put_entry(root, entry, content)
            journal_write(root, identity, "committed")
        except BaseException:
            restore(root, directory, entries)
            raise
        finish_journal(root)
        print(f"Update installed. Backups: {directory}", flush=True)
    return 0


def recover(root):
    with InstallationLock(root, exclusive=True):
        if not safe_path(root, JOURNAL).exists():
            print("No incomplete update.")
            return 0
        journal, directory, entries = load_transaction(root)
        if journal["phase"] == "committed":
            finish_journal(root)
            print("Completed update finalized.")
        elif journal["phase"] == "prepared":
            # No application files have been touched; keep any edits made since staging.
            finish_journal(root)
            print("Unstarted update discarded. Application files were not changed.")
        else:
            restore(root, directory, entries)
            print("Previous installation restored. SQLite and pending operations were not changed.")
    return 0


def update(root, client, *, check=False, dry_run=False, overwrite_configs=False, force_code=False):
    with InstallationLock(root, exclusive=not (check or dry_run)):
        require_complete(root)
        state = load_state(root)
        commit, remote, requirements = client.snapshot()
        print(f"Installed: {state['commit'] or 'untracked installation'}; main: {commit}")
        actions, conflicts, kept, next_state = plan_update(
            root, state, commit, remote, requirements,
            overwrite_configs=overwrite_configs, force_code=force_code)
        for entry in actions:
            print(f"{'REMOVE' if entry['after'] is None else 'UPDATE'} {entry['path']}")
        for name in kept:
            print(f"KEEP local configuration: {name}; upstream proposal: {name}.incoming")
        for name in conflicts:
            print(f"CONFLICT {name}")
        if requirements != state.get("requirements"):
            print("requirements.txt changed or has no recorded baseline. Review dependencies manually; "
                  "the updater does not download requirements or run pip.")
        if not state["commit"]:
            print("First adoption: stop any application started before updater support was installed.")
        if check or dry_run:
            print("Preview only: no application, configuration or updater-state writes.")
            return 2 if conflicts else 0
        if conflicts:
            raise UpdateError("Local or untracked Python files require --force-code; "
                              "edited .incoming proposals must be moved aside first. No files installed.")
        if not actions and state == next_state:
            print("Already up to date.")
            return 0
        identity, directory = prepare(root, actions, next_state, client)
    # Journal blocks app startup during the lock handoff. The child locks again
    # and verifies the transaction identity and every destination before applying.
    print("Installing with the temporary updater helper...", flush=True)
    environment = dict(os.environ)
    for key in ("GH_TOKEN", "GITHUB_TOKEN"):
        environment.pop(key, None)
    child = subprocess.Popen([sys.executable, str(directory / "worker.py"),
                              "--install-dir", str(root), "--_apply", identity],
                             env=environment)
    try:
        status = child.wait()
        if status and safe_path(root, JOURNAL).exists():
            print("Helper did not finish. Run python update_stash_ai.py --recover.", file=sys.stderr)
        return 0 if status == 0 else (130 if status in (-2, 130) else 1)
    except KeyboardInterrupt:
        # On Windows a console interrupt may reach both processes. Do not remove
        # the journal or race the helper with a rollback from the parent.
        print("Interrupted. Wait for the helper to exit, then use --recover if needed.", file=sys.stderr)
        return 130


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--install-dir", type=Path, default=Path(__file__).resolve().parent,
                        help="Application directory (default: next to this script).")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--check", action="store_true", help="Check main and local conflicts without downloading files.")
    mode.add_argument("--dry-run", action="store_true", help="Preview the complete update plan without installing.")
    mode.add_argument("--recover", action="store_true", help="Recover an interrupted installation, offline.")
    mode.add_argument("--_apply", help=argparse.SUPPRESS)
    parser.add_argument("--overwrite-configs", action="store_true", help="Replace local YAML files, keeping backups.")
    parser.add_argument("--force-code", action="store_true", help="Replace modified/untracked Python files, keeping backups.")
    args = parser.parse_args(argv)
    root = args.install_dir.resolve()
    if not root.is_dir():
        raise UpdateError("Installation directory does not exist.")
    if args.recover:
        return recover(root)
    if args._apply:
        return apply_transaction(root, args._apply)
    client = GitHub(os.environ.get("GH_TOKEN") or os.environ.get("GITHUB_TOKEN"))
    return update(root, client, check=args.check, dry_run=args.dry_run,
                  overwrite_configs=args.overwrite_configs, force_code=args.force_code)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        raise SystemExit(130)
    except (UpdateError, OSError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1)
