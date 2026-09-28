"""Separate-output model orchestration.

The public compatibility entry point remains stash_ai.cli.run_model.
This module owns the lifecycle of one model in separate-output mode while
calling CLI helpers dynamically through app so existing monkeypatch points
remain valid.
"""
from __future__ import annotations

from collections import Counter
from typing import Any


def _sync_hierarchy(app, resolver, vocabulary, args, scene_exclusions) -> Counter:
    from tqdm import tqdm

    source = resolver.source
    resolver.preload()
    labels = [label for label in vocabulary.labels if label not in scene_exclusions]
    selected = labels if args.create_all_tags else [
        label for label in labels if source.prefix + label in resolver.cache
    ]
    print(
        f"{'[DRY] ' if app.DRY_RUN else ''}{source.title}: "
        f"synchronizing hierarchy for {len(selected):,} tags."
    )
    for label in tqdm(selected, desc=source.key + " hierarchy", unit="tag"):
        resolver.resolve(source.prefix + label)
    app.print_hierarchy_summary(resolver)
    return Counter()


def _build_markers(app, args, config, frame_predictions, core_exclusions):
    if not app.CREATE_MARKERS:
        return [], 0

    from stash_ai.runtime import RamCache

    return app.build_markers(
        frame_predictions,
        core_exclusions,
        threshold=config.marker_threshold,
        cache=RamCache(
            int(args.ram_cache_gb * 2**30),
            rss_budget=int(getattr(args, "ram_budget_gb", 32) * 2**30),
        ),
    )


def _report_dry_run(app, stash, source, predictions, markers, report):
    if not app.DRY_RUN or not report.verbose:
        return

    from tqdm import tqdm

    if app.APPLY_SCENE_TAGS:
        tqdm.write(
            "  Scene predictions: "
            + (
                ", ".join(
                    f"{source.prefix}{tag}={score:.3f}"
                    for tag, score in predictions
                )
                or "(none)"
            )
        )
    for marker in markers:
        interval = app.format_time(marker.start)
        if app.MARKER_USE_END_TIME and stash.has_end_time:
            interval += " -> " + app.format_time(marker.end)
        tqdm.write(
            f"  {interval} | {source.prefix}{marker.tag} | "
            f"confidence={marker.score:.3f}, hits={marker.hits} | "
            f"extra={', '.join(source.prefix + tag for tag in marker.extra_tags) or '(none)'}"
        )


def _report_scene(app, stash, source, scene_id, title, cues, total_frames,
                  predictions, markers, total_markers, result, report):
    report.detail(
        f"{'[DRY] ' if app.DRY_RUN else ''}{source.key} scene {scene_id} {title!r}: "
        f"frames={len(cues)}/{total_frames}, scene_tags={len(predictions)}, "
        f"markers create={result['markers_created']} update={result['markers_updated']} "
        f"delete={result['markers_deleted']} keep={result['markers_unchanged']} "
        f"core_cleanup={result['markers_core_cleaned']} "
        f"scene_cleanup={result['scene_tags_cleaned']}"
    )
    if total_markers > len(markers):
        report.detail(
            f"  Marker cap: kept {len(markers)} of {total_markers} runs by confidence."
        )
    _report_dry_run(app, stash, source, predictions, markers, report)


def _process_scene(app, stash, tagger, resolver, scene, scene_id, config, args,
                   core_exclusions, scene_exclusions, report) -> Counter:
    source = config.source
    title = (scene.get("title") or "")[:60]
    paths = scene.get("paths") or {}

    if not paths.get("sprite") or not paths.get("vtt"):
        report.detail(
            f"SKIP {source.key} scene {scene_id}: sprite/VTT missing; "
            "generate Stash sprites first."
        )
        return Counter(skipped=1)

    if app.SKIP_EXISTING_OUTPUTS and app.outputs_already_present(scene, source):
        return Counter(skipped=1)

    cues, total_frames = app.fetch_cues(stash, paths)
    predictions, frame_predictions = app.predict_scene(
        tagger,
        stash,
        paths,
        cues,
        scene_exclusions,
        prefetch=args.prefetch_batches,
        report=report,
        prepare_workers=getattr(args, "prepare_workers", 1),
    )
    markers, total_markers = _build_markers(
        app,
        args,
        config,
        frame_predictions,
        core_exclusions,
    )
    current = stash.find_scene(scene_id)
    result = app.sync_scene(
        stash,
        resolver,
        current,
        predictions,
        markers,
        core_exclusions,
        source,
        scene_exclusions,
    )
    _report_scene(
        app,
        stash,
        source,
        scene_id,
        title,
        cues,
        total_frames,
        predictions,
        markers,
        total_markers,
        result,
        report,
    )

    stats = Counter(result)
    stats["processed"] += 1
    return stats


def _report_model_summary(app, resolver, source):
    if resolver.missing:
        verb = "Would create" if app.DRY_RUN and app.CREATE_MISSING else "Missing (skipped)"
        print(f"{source.key}: {verb} tags: {len(resolver.missing):,}")
    app.print_hierarchy_summary(resolver)


def run_model(app, stash, config, vocabulary, hierarchy, args: Any,
              core_exclusions: frozenset[str], scene_exclusions: frozenset[str]) -> Counter:
    """Run one model over selected scenes in separate-output mode."""
    from tqdm import tqdm

    source = config.source
    report = args.report
    resolver = app.TagResolver(
        stash,
        hierarchy,
        report=report.detail,
        source=source,
        exclusions=scene_exclusions,
        create_missing=app.CREATE_MISSING,
    )
    if args.sync_hierarchy_only:
        return _sync_hierarchy(app, resolver, vocabulary, args, scene_exclusions)

    stats = Counter()
    tagger = app.load_tagger(config, vocabulary, app.prepare_image)
    try:
        progress = tqdm(stash.iter_scenes(args.scene_id), desc=source.key, unit="scene")
        for scene in progress:
            scene_id = str(scene["id"])
            try:
                stats.update(_process_scene(
                    app,
                    stash,
                    tagger,
                    resolver,
                    scene,
                    scene_id,
                    config,
                    args,
                    core_exclusions,
                    scene_exclusions,
                    report,
                ))
            except Exception as exc:
                stats["errors"] += 1
                report.issue(
                    "inference",
                    source.key,
                    scene_id,
                    exc,
                    app.safe_error(exc, api_key=app.STASH_API_KEY),
                )
            finally:
                report.progress(progress, stats)
        _report_model_summary(app, resolver, source)
    finally:
        tagger.close()

    return stats
