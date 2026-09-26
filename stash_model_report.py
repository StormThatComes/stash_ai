"""Offline model utility and leave-one-model-out ablation report.

This report replays the normal fusion pipeline from retained state evidence. It
never runs inference and never connects to or writes to Stash.
"""

from __future__ import annotations

from collections import Counter
import json
from pathlib import Path

from stash_occurrences import is_occurrence
from stash_taggers import SOURCES


REPORT_FORMAT = 1


def _scene_labels(hits):
    return {item["tag"] for item in hits}


def _merged_markers(spans):
    from stash_merge import union_intervals
    return union_intervals(spans, borrow=True)


def _marker_key(marker):
    return (
        marker["tag"],
        round(float(marker["start"]), 3),
        None if marker["end"] is None else round(float(marker["end"]), 3),
        bool(is_occurrence(marker)),
        tuple(marker.get("extra_tags", ())),
    )


def _overlaps(left, right):
    if left["tag"] != right["tag"] or bool(is_occurrence(left)) != bool(is_occurrence(right)):
        return False
    a0 = float(left["start"])
    b0 = float(right["start"])
    a1 = float(left["end"]) if left["end"] is not None else a0
    b1 = float(right["end"]) if right["end"] is not None else b0
    if left["end"] is None and right["end"] is None:
        return abs(a0 - b0) <= .0005
    return max(a0, b0) <= min(a1, b1) + .0005


def _marker_delta(full, reduced):
    reduced_keys = {_marker_key(item) for item in reduced}
    full_keys = {_marker_key(item) for item in full}
    lost = 0
    changed = 0
    for marker in full:
        matches = [other for other in reduced if _overlaps(marker, other)]
        if not matches:
            lost += 1
        elif _marker_key(marker) not in reduced_keys:
            changed += 1
    added = sum(1 for marker in reduced if not any(_overlaps(marker, other) for other in full))
    return {
        "lost": lost,
        "changed": changed,
        "added": added,
        "exact_removed": len(full_keys - reduced_keys),
        "exact_added": len(reduced_keys - full_keys),
    }


def _extras(markers):
    return {(item["tag"], extra) for item in markers for extra in item.get("extra_tags", ())}


def _supports(hit, model):
    return hit.get("source") == model or model in hit.get("support_sources", ())


def _scene_ids(store, requested=None):
    table = store.table("results")
    if requested:
        wanted = list(dict.fromkeys(str(value) for value in requested))
        return [scene for scene in wanted if store.connection.execute(
            "SELECT 1 FROM " + table + " WHERE scope=? AND scene=? LIMIT 1",
            (store.scope, scene)).fetchone()]
    return [row[0] for row in store.connection.execute(
        "SELECT DISTINCT scene FROM " + table + " WHERE scope=? ORDER BY scene",
        (store.scope,))]


def _sources_for_scene(store, scene):
    return store.sources(scene, "scene") | store.sources(scene, "markers")


def build_report(app, store, scene_ids, core_policies, scene_policies):
    models = tuple(SOURCES)
    totals = Counter()
    utility = {model: Counter() for model in models}
    source_presence = Counter()

    for scene in scene_ids:
        present = _sources_for_scene(store, scene)
        source_presence.update(present)

        full_hits, full_spans = app.stash_merge.evidence(
            app, store, scene, core_policies, scene_policies)
        full_labels = _scene_labels(full_hits)
        full_markers = _merged_markers(full_spans)
        full_extras = _extras(full_markers)

        totals["scenes"] += 1
        totals["scene_tags"] += len(full_labels)
        totals["markers"] += len(full_markers)
        totals["occurrences"] += sum(is_occurrence(marker) for marker in full_markers)
        totals["marker_extras"] += len(full_extras)

        for model in models:
            stats = utility[model]
            stats["scenes_total"] += 1
            if model not in present:
                stats["scenes_unchanged"] += 1
                continue

            stats["scenes_with_evidence"] += 1
            stats["scene_tags_supported"] += len({
                hit["tag"] for hit in full_hits if _supports(hit, model)
            })
            stats["marker_components_supported"] += sum(
                model in marker.get("sources", ()) for marker in full_markers)
            stats["occurrences_supported"] += sum(
                is_occurrence(marker) and model in marker.get("sources", ())
                for marker in full_markers)

            selected = set(models) - {model}
            reduced_hits, reduced_spans = app.stash_merge.evidence(
                app, store, scene, core_policies, scene_policies,
                selected_sources=selected)
            reduced_labels = _scene_labels(reduced_hits)
            reduced_markers = _merged_markers(reduced_spans)
            reduced_extras = _extras(reduced_markers)

            lost_scene = full_labels - reduced_labels
            added_scene = reduced_labels - full_labels
            delta = _marker_delta(full_markers, reduced_markers)

            stats["scene_tags_lost"] += len(lost_scene)
            stats["scene_tags_added"] += len(added_scene)
            stats["marker_components_lost"] += delta["lost"]
            stats["marker_components_changed"] += delta["changed"]
            stats["marker_components_added"] += delta["added"]
            stats["marker_exact_removed"] += delta["exact_removed"]
            stats["marker_exact_added"] += delta["exact_added"]
            stats["marker_extras_lost"] += len(full_extras - reduced_extras)
            stats["marker_extras_added"] += len(reduced_extras - full_extras)
            stats["occurrences_lost"] += sum(
                is_occurrence(marker) and not any(_overlaps(marker, other) for other in reduced_markers)
                for marker in full_markers)
            changed = bool(
                lost_scene or added_scene or delta["exact_removed"] or delta["exact_added"]
                or full_extras != reduced_extras
            )
            stats["scenes_changed"] += int(changed)
            stats["scenes_unchanged"] += int(not changed)

    rows = []
    for model in models:
        stats = utility[model]
        with_evidence = stats["scenes_with_evidence"]
        rows.append({
            "model": model,
            **{key: stats[key] for key in (
                "scenes_total", "scenes_with_evidence", "scenes_changed", "scenes_unchanged",
                "scene_tags_supported", "scene_tags_lost", "scene_tags_added",
                "marker_components_supported", "marker_components_lost",
                "marker_components_changed", "marker_components_added",
                "marker_exact_removed", "marker_exact_added",
                "marker_extras_lost", "marker_extras_added",
                "occurrences_supported", "occurrences_lost",
            )},
            "changed_scene_pct_of_evidence": (
                round(stats["scenes_changed"] * 100 / with_evidence, 2) if with_evidence else 0
            ),
        })

    return {
        "format": REPORT_FORMAT,
        "mode": "leave-one-model-out",
        "semantics": {
            "scene_tag_loss": "A final scene label disappears when this model is omitted.",
            "marker_lost": "A final same-label/same-type marker has no overlapping counterpart after omission.",
            "marker_changed": "An overlapping marker remains but its time range, occurrence type, or extras differ.",
            "scene_changed": "Any final scene tag, marker semantic identity, or marker-extra set changes.",
            "source_badges_ignored_for_changed": True,
            "counterfactual_replays_consensus_temporal_extras_and_occurrence": True,
        },
        "totals": dict(totals),
        "models": rows,
    }


def format_summary(report):
    total = report["totals"]
    lines = [
        (
            f"Model utility: {total.get('scenes', 0):,} scene(s), "
            f"{total.get('scene_tags', 0):,} final scene tags, "
            f"{total.get('markers', 0):,} marker components "
            f"({total.get('occurrences', 0):,} occurrences)."
        ),
        "",
        "model     evidence  changed   scene lost  marker lost  marker changed  extras lost  occurrence lost",
    ]
    for row in report["models"]:
        lines.append(
            f"{row['model']:<9} "
            f"{row['scenes_with_evidence']:>8,} "
            f"{row['scenes_changed']:>8,} "
            f"{row['scene_tags_lost']:>11,} "
            f"{row['marker_components_lost']:>12,} "
            f"{row['marker_components_changed']:>14,} "
            f"{row['marker_extras_lost']:>11,} "
            f"{row['occurrences_lost']:>15,}"
        )
    return "\n".join(lines)


def write_report(report, destination):
    destination = Path(destination).expanduser().resolve()
    if destination.suffix.lower() != ".json":
        raise ValueError("Model utility report output must use a .json filename.")
    payload = json.dumps(report, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n"
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(payload)
    return destination


def run(app, args):
    import stash_merge
    from stash_exclusions import policies_or_empty
    from stash_temporal import load_policy

    app.stash_merge = stash_merge
    app.TEMPORAL_POLICY = load_policy(app, args.temporal_policy, not args.no_temporal)
    core_policies = app.load_exclusion_policy(args.marker_core_exclusions, "excluded_core_tags")
    scene_policies = app.load_exclusion_policy(args.scene_tag_exclusions, "excluded_scene_tags")
    core_policies = policies_or_empty(core_policies)
    scene_policies = policies_or_empty(scene_policies)

    try:
        with stash_merge.StateStore(
            args.state_db,
            stash_merge.scope_for(app.STASH_URL),
            True,
            cache_bytes=int(args.ram_cache_gb * 2**30),
            memory_budget_bytes=int(args.ram_budget_gb * 2**30),
            allow_cleanup=True,
        ) as store:
            scenes = _scene_ids(store, args.scene_id)
            if args.scene_id and len(scenes) != len(dict.fromkeys(str(value) for value in args.scene_id)):
                missing = sorted(set(map(str, args.scene_id)) - set(scenes))
                raise ValueError("No saved evidence for scene(s): " + ", ".join(missing))
            report = build_report(app, store, scenes, core_policies, scene_policies)
        destination = write_report(report, args.model_report)
        print(format_summary(report))
        print(f"\nDetailed model utility report written to {destination}. No Stash requests or inference were performed.")
        return 0
    except (OSError, ValueError, RuntimeError) as exc:
        print("Model utility report failed: " + str(exc))
        return 1
