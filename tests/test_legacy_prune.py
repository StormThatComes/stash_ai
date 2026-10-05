"""Safe cleanup of legacy-import anti-duplication history."""

from copy import deepcopy
import sqlite3

import pytest

from tests.helpers.runtime import runtime as app
from stash_ai.cleanup import legacy as legacy_prune
from stash_ai.storage.store import StateStore, scope_for
from stash_ai.models import JOYTAG, WD
from tests.helpers.stash import Stash


def _seed(path, stash):
    scene_tag = stash.tag("jt:common")
    stash.scene["tags"] = [scene_tag]
    marker = stash.marker(WD, "common", "legacy-wd")
    with StateStore(path, scope_for(app.STASH_URL)) as store:
        with store.connection:
            store.legacy.remember_in_transaction("1", "scene", [deepcopy(scene_tag)])
            store.legacy.remember_in_transaction("1", "markers", [deepcopy(marker)])
            store.connection.execute(
                "INSERT INTO merge_checks VALUES (?,?,?,?)",
                (store.scope, "1", "policy", "fingerprint"),
            )
    return scene_tag, marker


def _install(monkeypatch, stash):
    monkeypatch.setattr(app, "DRY_RUN", False)
    monkeypatch.setattr(app.services, "create_stash_client", lambda *_: stash)
    monkeypatch.setattr(
        app.services, "load_model_settings",
        lambda *a, **k: pytest.fail("legacy pruning loaded model configuration"),
    )


def test_prune_removes_only_unreferenced_rows_and_preserves_merge_check(tmp_path, monkeypatch, capsys):
    path = tmp_path / "state.sqlite3"
    stash = Stash()
    _seed(path, stash)

    # Simulate an older installation where ledger writes invalidated merge_checks.
    with sqlite3.connect(path) as db:
        db.execute("""
            CREATE TRIGGER perf_legacy_imports_delete
            AFTER DELETE ON legacy_imports BEGIN
                DELETE FROM merge_checks WHERE scope=OLD.scope AND scene=OLD.scene;
            END
        """)

    from stash_ai.db import update_database
    update_database(path)
    # The scene legacy tag still exists, but its imported source marker is gone.
    stash.scene["scene_markers"] = []
    _install(monkeypatch, stash)

    assert app.main(["--prune-legacy-imports", "--state-db", str(path)]) == 0
    assert stash.create_markers is True
    assert stash.merge_read_configurations == [True]

    with sqlite3.connect(path) as db:
        assert db.execute(
            "SELECT kind FROM legacy_imports ORDER BY kind"
        ).fetchall() == [("scene",)]
        assert db.execute("SELECT COUNT(*) FROM merge_checks").fetchone() == (1,)
        assert db.execute(
            "SELECT COUNT(*) FROM sqlite_master WHERE type='trigger' "
            "AND name='perf_legacy_imports_delete'"
        ).fetchone() == (0,)

    output = capsys.readouterr().out
    assert "removed 1/2 row(s)" in output
    assert "retained 1 row(s)" in output


def test_sql_batch_path_treats_missing_scene_as_prunable(tmp_path):
    path = tmp_path / "state.sqlite3"
    with StateStore(path, scope_for(app.STASH_URL)) as store:
        with store.connection:
            store.legacy.remember_in_transaction("99", "scene", [{"id": "7", "name": "jt:gone"}])

        class SQLStash:
            merge_reader = object()
            scene_page_size = 100

            def merge_snapshots(self, identifiers):
                assert list(identifiers) == ["99"]
                return {}

        removable, retained, _ = legacy_prune.candidates(app, store, SQLStash())

    assert [(scene, kind) for scene, kind, _ in removable] == [("99", "scene")]
    assert retained == 0


def test_prune_dry_run_uses_source_db_without_modifying_it(tmp_path, monkeypatch, capsys):
    path = tmp_path / "state.sqlite3"
    stash = Stash()
    _seed(path, stash)
    stash.scene["tags"] = []
    stash.scene["scene_markers"] = []
    _install(monkeypatch, stash)

    assert app.main([
        "--prune-legacy-imports", "--dry-run", "--state-db", str(path)
    ]) == 0

    with sqlite3.connect(path) as db:
        assert db.execute("SELECT COUNT(*) FROM legacy_imports").fetchone() == (2,)

    assert "would remove 2/2 row(s)" in capsys.readouterr().out


@pytest.mark.parametrize("extra", [
    ["--output-mode", "separate"],
    ["--models", "wd"],
    ["--scene-id", "1"],
    ["--markers-only"],
    ["--abandon-pending"],
])
def test_prune_rejects_incompatible_modes_before_state_access(tmp_path, extra):
    path = tmp_path / "state.sqlite3"
    with pytest.raises(SystemExit) as error:
        app.main(["--prune-legacy-imports", "--state-db", str(path), *extra])
    assert error.value.code == 2
    assert not path.exists()


def test_prune_refuses_pending_publication(tmp_path, monkeypatch):
    path = tmp_path / "state.sqlite3"
    stash = Stash()
    _seed(path, stash)
    with StateStore(path, scope_for(app.STASH_URL)) as store:
        store.prepare("1", {"scene": {"owned_ids": [], "contributions": []}})
    _install(monkeypatch, stash)

    with pytest.raises(RuntimeError, match="Pending publication plans"):
        app.main(["--prune-legacy-imports", "--state-db", str(path)])

