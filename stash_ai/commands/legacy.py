"""Capture source-prefixed Stash objects before merged-output migration."""
from collections import defaultdict

from stash_ai import merge, separate_sync
from stash_ai.models import SOURCES


def legacy_label(name):
    for key, source in SOURCES.items():
        if name.startswith(source.prefix) and name != source.owner_tag:
            label = name[len(source.prefix):]
            if label and not label.startswith(("category:", "marker:")):
                return key, merge.raw_label(label)
    return None


def bootstrap_legacy(app, store, scene):
    """Capture all known source namespaces before migrating any associations."""
    scene_id = str(scene["id"])
    for kind, enabled in (("scene", app.APPLY_SCENE_TAGS), ("markers", app.CREATE_MARKERS)):
        if not enabled:
            continue
        existing = store.sources(scene_id, kind)
        if kind == "markers" and not existing and any(separate_sync.owned_marker(marker, merge.AI) for marker in scene.get("scene_markers", [])):
            raise RuntimeError("Merged markers exist without their source state. Restore the original --state-db before synchronization.")
        from stash_ai.performance import freeze
        imported = {freeze(item) for item in store.imported_legacy(scene_id, kind, borrow=True)}
        originals = []
        by_model = defaultdict(list)
        if kind == "scene":
            for tag in scene.get("tags", []):
                parsed = legacy_label(tag["name"])
                if parsed:
                    model, label = parsed
                    originals.append(tag)
                    if model not in existing or freeze(tag) not in imported:
                        by_model[model].append({"tag": label, "score": None, "legacy_tag_id": str(tag["id"])})
        else:
            for marker in scene.get("scene_markers", []):
                parsed = legacy_label((marker.get("primary_tag") or {}).get("name", ""))
                if not parsed or not separate_sync.owned_marker(marker, SOURCES[parsed[0]]):
                    continue
                model, label = parsed
                originals.append(marker)
                if model in existing and freeze(marker) in imported:
                    continue
                extras, manual_ids = set(), set()
                for tag in marker.get("tags", []):
                    extra = legacy_label(tag["name"])
                    if extra:
                        extras.add(extra[1])
                    elif tag["name"] not in {source.owner_tag for source in SOURCES.values()}:
                        manual_ids.add(str(tag["id"]))
                by_model[model].append(dict(tag=label, start=marker["seconds"], end=marker.get("end_seconds"),
                    score=None, hits=None, extra_tags=sorted(extras - {label}), extra_ids=sorted(manual_ids),
                    legacy_marker_id=str(marker["id"])))
        for model, items in by_model.items():
            segment = {"mode": "scene" if kind == "scene" else "spans", "items": items,
                       "meta": {"origin": "legacy", "captured_at": merge.now(), "revision": None}}
            # Fresh separate-mode outputs augment the cache. Only a successful
            # merged-mode inference replaces that model's previous evidence.
            store.put(scene_id, model, {kind: segment}, {kind: False})
        store.remember_legacy(scene_id, kind, originals)
