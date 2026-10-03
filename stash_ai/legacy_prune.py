"""Prune legacy-import ledgers once their exact source objects are gone from Stash."""

from __future__ import annotations

from collections import defaultdict
from itertools import batched

from tqdm import tqdm

from stash_ai import separate_sync
from stash_ai import merge
from stash_ai.db.common import format_bytes
from stash_ai.performance import freeze
from stash_ai.state import StateStore, scope_for
from stash_ai.models import SOURCES


def _current_legacy(app, scene):
    """Return exact source objects that bootstrap_legacy would recognize now."""
    values = {"scene": set(), "markers": set()}
    for tag in scene.get("tags", []):
        if merge.legacy_label(tag.get("name", "")):
            values["scene"].add(freeze(tag))

    for marker in scene.get("scene_markers", []):
        parsed = merge.legacy_label((marker.get("primary_tag") or {}).get("name", ""))
        source = SOURCES.get(parsed[0]) if parsed else None
        if source is not None and separate_sync.owned_marker(marker, source):
            values["markers"].add(freeze(marker))
    return values


def _legacy_meta(store):
    return list(store.connection.execute(
        "SELECT scene,kind,length(data) FROM " + store.table("legacy_imports") +
        " WHERE scope=? ORDER BY scene,kind",
        (store.scope,),
    ))


def _tracked_scenes(stash, identifiers):
    """Use batched SQL when available, but treat missing scene IDs as deleted."""
    if getattr(stash, "merge_reader", None) is None:
        yield from stash.iter_scenes()
        return

    page_size = min(max(1, int(getattr(stash, "scene_page_size", 100))), 1000)
    for group in batched(identifiers, page_size):
        snapshots = stash.merge_snapshots(group)
        if snapshots is None:
            # SQL failed and the client disabled the reader. A full GraphQL scan
            # is safe even after earlier groups: live identities are held in sets.
            yield from stash.iter_scenes()
            return
        yield from snapshots.values()


def candidates(app, store, stash):
    """Find ledger rows with no exact imported source object left in live Stash."""
    meta = _legacy_meta(store)
    if not meta:
        return [], 0, 0

    tracked_scenes = {scene for scene, _, _ in meta}
    live = defaultdict(set)
    for scene in tqdm(_tracked_scenes(stash, list(tracked_scenes)),
                      desc="Inspect legacy imports", unit="scene"):
        scene_id = str(scene["id"])
        if scene_id not in tracked_scenes:
            continue
        for kind, items in _current_legacy(app, scene).items():
            if items:
                live[scene_id, kind].update(items)

    removable = []
    retained = 0
    query = (
        "SELECT scene,kind,data,length(data) FROM " + store.table("legacy_imports") +
        " WHERE scope=? ORDER BY scene,kind"
    )
    for scene, kind, blob, size in store.connection.execute(query, (store.scope,)):
        imported = store.unpack(blob)
        if not isinstance(imported, list):
            raise ValueError("Invalid legacy_imports payload.")
        current = live.get((scene, kind), set())
        if any(freeze(item) in current for item in imported):
            retained += 1
        else:
            removable.append((scene, kind, int(size or 0)))
    return removable, retained, sum(int(size or 0) for _, _, size in meta)


def run(app, args):
    """Remove only anti-duplication history no longer referenced by live legacy objects."""
    with StateStore(args.state_db, scope_for(app.STASH_URL), app.DRY_RUN) as store:
        if store.has_pending_publications():
            raise RuntimeError(
                "Pending publication plans exist; finish or abandon them before pruning legacy imports."
            )

        table = store.table("legacy_imports")
        rows = store.connection.execute(
            "SELECT COUNT(*),COALESCE(SUM(length(data)),0) FROM " + table + " WHERE scope=?",
            (store.scope,),
        ).fetchone()
        row_count, payload_bytes = int(rows[0]), int(rows[1] or 0)
        if not row_count:
            print("Legacy imports: nothing to prune.")
            return 0

        marker_rows = store.connection.execute(
            "SELECT 1 FROM " + table + " WHERE scope=? AND kind='markers' LIMIT 1",
            (store.scope,),
        ).fetchone() is not None
        stash = app.create_stash_client(app.DRY_RUN)
        try:
            # The normal client may omit markers when marker output is disabled.
            # Request them here only when the ledger actually contains marker rows.
            if hasattr(stash, "create_markers"):
                stash.create_markers = marker_rows
            if hasattr(stash, "configure_merge_reads"):
                stash.configure_merge_reads(args.report, merge_only=True)
            removable, retained, _ = candidates(app, store, stash)
        finally:
            stash.close()

        reclaimable = sum(size for _, _, size in removable)
        if not app.DRY_RUN and removable:
            with store.connection:
                store.connection.executemany(
                    "DELETE FROM legacy_imports WHERE scope=? AND scene=? AND kind=?",
                    [(store.scope, scene, kind) for scene, kind, _ in removable],
                )

        verb = "would remove" if app.DRY_RUN else "removed"
        print(
            f"Legacy imports: {verb} {len(removable):,}/{row_count:,} row(s), "
            f"{format_bytes(reclaimable)} of {format_bytes(payload_bytes)} payload; "
            f"retained {retained:,} row(s) still referenced by live legacy objects."
        )
        if reclaimable and not app.DRY_RUN:
            print("Run --db-compact later to return freed SQLite pages to the filesystem.")
        return 0
