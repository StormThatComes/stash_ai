"""Shared category hierarchy and hierarchy-only synchronization."""

import stash_ai.evidence.labels as merge_labels
from stash_ai.domain import TagHierarchy


def combined_hierarchy(app, args, tag_policies, hierarchy=None):
    """Use the same taxonomy for all models, filtering globally banned labels."""
    if args.no_hierarchy:
        return None
    from stash_ai.exclusions import require_policies
    tag_policies = require_policies(tag_policies)
    if hierarchy is None:
        hierarchy = app.services.load_tag_hierarchy(args.hierarchy)
    excluded = set.intersection(*(set(values) for values in tag_policies.values()))
    return TagHierarchy({label: parents for label, parents in hierarchy.leaf_parents.items()
                         if label not in excluded}, hierarchy.category_parents,
                        hierarchy.category_prefix, hierarchy.description_prefix)


def run_hierarchy(app, args, catalogs, hierarchy, tag_policies):
    from stash_ai.exclusions import require_policies
    tag_policies = require_policies(tag_policies)
    stash = app.services.create_stash_client(app.DRY_RUN)
    try:
        resolver = app.services.TagResolver(stash, hierarchy, source=merge_labels.AI, report=args.report.detail, create_missing=app.CREATE_MISSING)
        resolver.preload()
        labels = sorted({label for source, vocabulary in catalogs.items() for label in vocabulary.labels
                         if label not in tag_policies[source]})
        for label in labels:
            if args.create_all_tags or merge_labels.AI.prefix + label in resolver.cache:
                resolver.resolve(merge_labels.AI.prefix + label)
        app.services.print_hierarchy_summary(resolver)
    finally:
        stash.close()
    return 0
