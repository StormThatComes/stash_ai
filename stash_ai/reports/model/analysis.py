"""Replay source support and leave-one-model-out analysis for one scene."""
from collections import Counter
from stash_ai.models import SOURCES
from stash_ai.evidence.occurrences import is_occurrence
from .comparison import _extras, _marker_delta


def _scene_labels(hits):
    return {item["tag"] for item in hits}


def _merged_markers(spans):
    from stash_ai.evidence.intervals import union_intervals
    return union_intervals(spans, borrow=True)


def _sources_for_scene(store, scene):
    return store.sources(scene, "scene") | store.sources(scene, "markers")


def _scene_baseline(app, store, scene, core_policies, scene_policies):
    from stash_ai.evidence import pipeline as stash_merge

    hits, spans = stash_merge.evidence(
        app, store, scene, core_policies, scene_policies)
    labels = _scene_labels(hits)
    markers = _merged_markers(spans)
    extras = _extras(markers)
    sources = {
        label: {
            source
            for hit in hits
            if hit["tag"] == label
            for source in (hit["source"], *hit.get("support_sources", ()))
        }
        for label in labels
    }
    return {
        "present": _sources_for_scene(store, scene),
        "labels": labels,
        "markers": markers,
        "extras": extras,
        "sources": sources,
    }


def _record_scene_totals(totals, baseline):
    totals["scenes"] += 1
    totals["scene_tags"] += len(baseline["labels"])
    totals["markers"] += len(baseline["markers"])
    totals["occurrences"] += sum(
        is_occurrence(marker) for marker in baseline["markers"])
    totals["marker_extras"] += len(baseline["extras"])


def _record_model_support(stats, model, baseline):
    stats["scenes_total"] += 1
    if model not in baseline["present"]:
        stats["scenes_unchanged"] += 1
        return False

    stats["scenes_with_evidence"] += 1
    stats["scene_tags_supported"] += sum(
        model in sources for sources in baseline["sources"].values())
    stats["scene_tags_sole_source"] += sum(
        sources == {model} for sources in baseline["sources"].values())
    stats["scene_tags_multi_model"] += sum(
        model in sources and len(sources) > 1
        for sources in baseline["sources"].values())
    stats["marker_components_supported"] += sum(
        model in marker.get("sources", ())
        for marker in baseline["markers"])
    stats["marker_components_sole_source"] += sum(
        set(marker.get("sources", ())) == {model}
        for marker in baseline["markers"])
    stats["marker_components_multi_model"] += sum(
        model in marker.get("sources", ())
        and len(marker.get("sources", ())) > 1
        for marker in baseline["markers"])
    stats["occurrences_supported"] += sum(
        is_occurrence(marker) and model in marker.get("sources", ())
        for marker in baseline["markers"])
    return True


def _record_model_ablation(app, store, scene, model, models, baseline, stats,
                           core_policies, scene_policies):
    from stash_ai.evidence import pipeline as stash_merge

    reduced_hits, reduced_spans = stash_merge.evidence(
        app,
        store,
        scene,
        core_policies,
        scene_policies,
        selected_sources=set(models) - {model},
    )
    reduced_labels = _scene_labels(reduced_hits)
    reduced_markers = _merged_markers(reduced_spans)
    reduced_extras = _extras(reduced_markers)

    lost_scene = baseline["labels"] - reduced_labels
    added_scene = reduced_labels - baseline["labels"]
    delta = _marker_delta(baseline["markers"], reduced_markers)

    stats["scene_tags_lost"] += len(lost_scene)
    stats["scene_tags_added"] += len(added_scene)
    stats["marker_components_lost"] += delta["lost"]
    stats["marker_components_changed"] += delta["changed"]
    stats["marker_components_added"] += delta["added"]
    stats["marker_exact_removed"] += delta["exact_removed"]
    stats["marker_exact_added"] += delta["exact_added"]
    stats["marker_extras_lost"] += len(
        baseline["extras"] - reduced_extras)
    stats["marker_extras_added"] += len(
        reduced_extras - baseline["extras"])
    stats["occurrences_lost"] += delta["occurrences_lost"]

    changed = bool(
        lost_scene
        or added_scene
        or delta["exact_removed"]
        or delta["exact_added"]
        or baseline["extras"] != reduced_extras
    )
    stats["scenes_changed"] += int(changed)
    stats["scenes_unchanged"] += int(not changed)


def scene_report(app, store, scene, core_policies, scene_policies):
    """Return mergeable counters for one independent scene."""
    models = tuple(SOURCES)
    totals = Counter()
    utility = {model: Counter() for model in models}

    baseline = _scene_baseline(
        app, store, scene, core_policies, scene_policies)
    _record_scene_totals(totals, baseline)

    for model in models:
        stats = utility[model]
        if not _record_model_support(stats, model, baseline):
            continue
        _record_model_ablation(
            app,
            store,
            scene,
            model,
            models,
            baseline,
            stats,
            core_policies,
            scene_policies,
        )
    return totals, utility


def merge_scene_report(totals, utility, partial):
    scene_totals, scene_utility = partial
    totals.update(scene_totals)
    for model in utility:
        utility[model].update(scene_utility[model])
