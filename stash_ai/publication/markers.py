"""Marker ownership, payloads and extra-tag reconciliation shared by output modes."""
from __future__ import annotations

from stash_ai.models import ModelSource

def owned_marker(marker: dict, source: ModelSource) -> bool:
    return bool(
        (marker.get("title") or "").startswith(source.title_prefix)
        and any(tag.get("name") == source.owner_tag for tag in marker.get("tags") or [])
    )

def marker_has_excluded_core(marker: dict, core_exclusions: frozenset[str],
                             source: ModelSource) -> bool:
    name = (marker.get("primary_tag") or {}).get("name") or ""
    return name.startswith(source.prefix) and name[len(source.prefix):] in core_exclusions

def marker_extra_cleanup_payload(marker: dict, scene_id: str,
                                 blocked_names: set[str]):
    """Preserve marker identity, primary, title and timing while stripping extras."""
    primary = str(marker["primary_tag"]["id"])
    extras = {
        str(tag["id"])
        for tag in marker.get("tags", [])
        if str(tag["id"]) != primary
    }
    kept = {
        str(tag["id"])
        for tag in marker.get("tags", [])
        if str(tag["id"]) != primary and tag["name"] not in blocked_names
    }
    if extras == kept:
        return None
    payload = dict(
        id=str(marker["id"]),
        scene_id=scene_id,
        title=marker["title"],
        seconds=marker["seconds"],
        primary_tag_id=primary,
        tag_ids=sorted(kept),
    )
    payload["end_seconds"] = marker.get("end_seconds")
    return payload

def marker_payload(app, spec, scene_id: str, ids: dict[str, str | None],
                   source: ModelSource) -> dict:
    primary = ids[source.prefix + spec.tag]
    extras = {ids[source.prefix + tag] for tag in spec.extra_tags}
    extras.add(ids[source.owner_tag])
    extras.discard(primary)
    extras.discard(None)
    payload = {
        "scene_id": scene_id,
        "title": source.title_prefix + spec.tag,
        "seconds": round(spec.start, 3),
        "primary_tag_id": primary,
        "tag_ids": sorted(extras),
    }
    payload["end_seconds"] = round(spec.end, 3) if app.MARKER_USE_END_TIME else None
    return payload

def marker_differs(existing: dict, wanted: dict) -> bool:
    if (
        existing["title"] != wanted["title"]
        or abs(existing["seconds"] - wanted["seconds"]) > 0.0005
    ):
        return True
    if str(existing["primary_tag"]["id"]) != wanted["primary_tag_id"]:
        return True
    # Stash may return the primary tag among associations; compare only extras.
    extra_ids = {
        str(tag["id"])
        for tag in existing["tags"]
    } - {wanted["primary_tag_id"]}
    if extra_ids != set(wanted["tag_ids"]):
        return True
    old_end, new_end = existing.get("end_seconds"), wanted["end_seconds"]
    if old_end is None or new_end is None:
        return old_end != new_end
    return abs(old_end - new_end) > 0.0005
