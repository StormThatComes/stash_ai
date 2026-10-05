"""Prepare runtime options, policies and model assets in CLI phase order."""
from stash_ai.evidence.temporal.policy import TemporalPolicy
from stash_ai.models import SOURCES


def runtime_options(app, args):
    from stash_ai.commands.validation import mode_flags
    from stash_ai.runtime.report import RunReport
    from tqdm import tqdm
    args.report = RunReport(args.verbose, tqdm.write)
    app.DRY_RUN = app.DRY_RUN or args.dry_run
    if args.markers_only or args.cleanup_markers_only or args.sync_marker_groups_only:
        app.APPLY_SCENE_TAGS, app.CREATE_MARKERS = False, True
    elif args.scene_tags_only or args.cleanup_scene_tags_only:
        app.APPLY_SCENE_TAGS, app.CREATE_MARKERS = True, False
    elif args.cleanup_only:
        app.APPLY_SCENE_TAGS, app.CREATE_MARKERS = True, True
    hierarchy_only, cleanup_only = mode_flags(args)
    return hierarchy_only, cleanup_only


def selected_settings(app, args):
    settings = app.services.load_model_settings(args.model_config, args.models, args.device, args.offline)
    print("Selected models: " + ", ".join(f"{item.source.title} ({item.source.code})" for item in settings))
    return settings


def download_models(app, settings):
    for config in settings:
        app.services.load_vocabulary(config)
        app.services.ensure_files(config, app.services.weight_files(config))
        if config.source.key == "joytag":
            from stash_ai.backends.joytag import loader_path
            loader_path(config.download_missing)
        print(f"{config.source.title}: files ready in {config.model_dir}.")


def policies(app, args, *, hierarchy_only, cleanup_only):
    from stash_ai.evidence.temporal.policy import load_policy
    app.TEMPORAL_POLICY = (
        load_policy(args.temporal_policy)
        if not hierarchy_only else TemporalPolicy())
    app.services.validate_config(scene_work=not hierarchy_only and not cleanup_only and not args.explain_tag)
    empty_policies = {name: frozenset() for name in SOURCES}
    core_policies = (
        app.services.load_exclusion_policy(args.marker_core_exclusions, "excluded_core_tags")
        if app.CREATE_MARKERS and not hierarchy_only else empty_policies)
    scene_policies = app.services.load_exclusion_policy(args.scene_tag_exclusions, "excluded_scene_tags")
    return core_policies, scene_policies


def catalogs(app, settings, args, scene_policies):
    """Validate all selected vocabularies/hierarchies before any Stash writes."""
    catalogs = {}
    for config in settings:
        source = config.source
        vocabulary = app.services.load_vocabulary(config)
        catalogs[source.key] = vocabulary
        hierarchy_path = args.hierarchy if source.key == "joytag" else getattr(args, source.key + "_hierarchy")
        if args.no_hierarchy:
            hierarchy = None
        else:
            hierarchy = app.services.load_tag_hierarchy(hierarchy_path, vocabulary.labels, source)
        print(f"{source.title}: {len(vocabulary.labels):,} labels; "
              f"hierarchy={'off' if hierarchy is None else str(len(hierarchy.category_parents)) + ' categories'}; "
              f"scene threshold={config.scene_threshold}, marker threshold={config.marker_threshold}.")
    from stash_ai.commands.hierarchy import combined_hierarchy
    merged_hierarchy = combined_hierarchy(app, args, scene_policies)
    if merged_hierarchy:
        print(f"AI: {len(merged_hierarchy.leaf_parents):,} unique labels; {len(merged_hierarchy.category_parents)} shared categories.")
    return catalogs, merged_hierarchy
