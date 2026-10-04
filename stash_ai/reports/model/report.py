"""Offline model utility and leave-one-model-out ablation report.

This report replays the normal fusion pipeline from retained state evidence. It
never runs inference and never connects to or writes to Stash.
"""

from __future__ import annotations

from .analysis import scene_report, merge_scene_report

from collections import Counter
import json
from pathlib import Path

from tqdm import tqdm

from stash_ai.models import SOURCES
from stash_ai.storage.store import StateStore, scope_for


REPORT_FORMAT = 1


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


_MODEL_ROW_KEYS = (
    "scenes_total", "scenes_with_evidence", "scenes_changed", "scenes_unchanged",
    "scene_tags_supported", "scene_tags_sole_source", "scene_tags_multi_model",
    "scene_tags_lost", "scene_tags_added",
    "marker_components_supported", "marker_components_sole_source",
    "marker_components_multi_model", "marker_components_lost",
    "marker_components_changed", "marker_components_added",
    "marker_exact_removed", "marker_exact_added",
    "marker_extras_lost", "marker_extras_added",
    "occurrences_supported", "occurrences_lost",
)


def _model_row(model, stats):
    with_evidence = stats["scenes_with_evidence"]
    return {
        "model": model,
        **{key: stats[key] for key in _MODEL_ROW_KEYS},
        "changed_scene_pct_of_evidence": (
            round(stats["scenes_changed"] * 100 / with_evidence, 2)
            if with_evidence else 0
        ),
    }


def report_payload(totals, utility):
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
        "models": [_model_row(model, utility[model]) for model in SOURCES],
    }


def build_report(app, store, scene_ids, core_policies, scene_policies):
    """Build the report serially; kept as the deterministic library/test path."""
    totals = Counter()
    utility = {model: Counter() for model in SOURCES}
    for scene in tqdm(scene_ids, desc="Model report", unit="scene"):
        merge_scene_report(
            totals,
            utility,
            scene_report(app, store, scene, core_policies, scene_policies),
        )
    return report_payload(totals, utility)


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
    from stash_ai.exclusions import policies_or_empty
    from stash_ai.evidence.temporal.policy import load_policy

    app.APPLY_SCENE_TAGS = True
    app.CREATE_MARKERS = True
    app.TEMPORAL_POLICY = load_policy(args.temporal_policy, not args.no_temporal)
    core_policies = app.services.load_exclusion_policy(args.marker_core_exclusions, "excluded_core_tags")
    scene_policies = app.services.load_exclusion_policy(args.scene_tag_exclusions, "excluded_scene_tags")
    core_policies = policies_or_empty(core_policies)
    scene_policies = policies_or_empty(scene_policies)

    try:
        if not Path(args.state_db).expanduser().is_file():
            raise FileNotFoundError(f"State database does not exist: {args.state_db}")
        cache_bytes = int(args.ram_cache_gb * 2**30)
        memory_budget_bytes = int(args.ram_budget_gb * 2**30)
        scope = scope_for(app.STASH_URL)
        workers = max(1, int(args.model_report_workers))
        report = None
        with StateStore(
            args.state_db,
            scope,
            True,
            cache_bytes=cache_bytes,
            memory_budget_bytes=memory_budget_bytes,
            allow_cleanup=True,
        ) as store:
            scenes = _scene_ids(store, args.scene_id)
            if args.scene_id and len(scenes) != len(dict.fromkeys(str(value) for value in args.scene_id)):
                missing = sorted(set(map(str, args.scene_id)) - set(scenes))
                raise ValueError("No saved evidence for scene(s): " + ", ".join(missing))
            workers = min(workers, len(scenes)) if scenes else 1
            if workers == 1:
                if not args.scene_id:
                    store.preload_sources()
                report = build_report(
                    app, store, scenes, core_policies, scene_policies)

        if report is None:
            from stash_ai.reports.model.workers import build_report_parallel

            report = build_report_parallel(
                app,
                args.state_db,
                scope,
                scenes,
                core_policies,
                scene_policies,
                workers=workers,
                cache_bytes=cache_bytes,
                memory_budget_bytes=memory_budget_bytes,
                preload_sources=not bool(args.scene_id),
            )
        destination = write_report(report, args.model_report)
        print(format_summary(report))
        print(f"\nDetailed model utility report written to {destination}. No Stash requests or inference were performed.")
        return 0
    except (OSError, ValueError, RuntimeError) as exc:
        print("Model utility report failed: " + str(exc))
        return 1

