"""Separate-output reconciliation with explicit runtime settings and model sources."""
from __future__ import annotations

from .planning.separate_markers import plan_separate_markers
from .planning.separate_tags import _resolved_names, _scene_tag_plan

from collections import Counter

from stash_ai.models import ModelSource
from stash_ai.publication import markers as marker_ops


def _sync_statistics(existing_markers, source, core_exclusions, plan, spec_count):
    excluded_ids = {
        str(marker["id"])
        for marker in existing_markers
        if marker_ops.owned_marker(marker, source)
        and marker_ops.marker_has_excluded_core(marker, core_exclusions, source)
    }
    return Counter(
        markers_created=len(plan.creates),
        markers_updated=len(plan.updates),
        markers_deleted=len(plan.deletes),
        markers_unchanged=plan.unchanged,
        markers_skipped=spec_count - len(plan.wanted),
        markers_extras_cleaned=len(plan.extra_updates),
        markers_core_cleaned=len(excluded_ids.intersection(plan.deletes)),
    )


def _publish_plan(stash, scene_id, plan, scene_changed, final_ids):
    if stash.dry_run:
        return
    # No destructive cleanup until all new/replacement marker writes succeed.
    for payload in plan.creates:
        stash.create_marker(payload)
    for payload in plan.updates:
        stash.update_marker(payload)
    if scene_changed:
        stash.update_scene_tags(scene_id, sorted(final_ids))
    for marker_id in plan.deletes:
        stash.delete_marker(marker_id)


def sync_scene(app, stash, resolver, scene: dict,
               scene_predictions: list[tuple[str, float]], specs: list,
               core_exclusions: frozenset[str], source: ModelSource,
               scene_exclusions: frozenset[str] = frozenset()) -> Counter:
    if resolver.source != source:
        raise ValueError(
            "Resolver/source mismatch; refusing to mix model namespaces."
        )

    from stash_ai.exclusions import filter_spec

    spec_count = len(specs)
    core_exclusions = core_exclusions | scene_exclusions
    specs = [
        filter_spec(spec, scene_exclusions)
        for spec in specs
        if spec.tag not in core_exclusions
    ]
    scene_id = str(scene["id"])
    existing_tags = scene.get("tags") or []
    existing_markers = scene.get("scene_markers") or []

    scene_names, ids = _resolved_names(
        app, resolver, source, scene_predictions, specs, scene_exclusions,
    )
    if app.CREATE_MARKERS and specs and not ids.get(source.owner_tag):
        raise ValueError(
            f"Missing marker ownership tag {source.owner_tag!r}; "
            "enable CREATE_MISSING or create it in Stash."
        )

    plan = plan_separate_markers(
        app, source, scene_id, existing_markers, specs, ids,
        core_exclusions, scene_exclusions,
    )
    stats = _sync_statistics(
        existing_markers, source, core_exclusions, plan, spec_count,
    )
    final_ids, scene_changed, scene_cleaned = _scene_tag_plan(
        app, source, existing_tags, scene_names, ids, scene_exclusions,
    )
    stats["scene_tags_updated"] = int(scene_changed)
    stats["scene_tags_cleaned"] = scene_cleaned

    _publish_plan(
        stash, scene_id, plan, scene_changed, final_ids,
    )
    return stats

