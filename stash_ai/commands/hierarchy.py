"""Merged category composition and hierarchy-only synchronization."""
from collections import defaultdict

from stash_ai import merge
from stash_ai.domain import TagHierarchy
from stash_ai.models import SOURCES


def combined_hierarchy(app, args, tag_policies=None):
    """Compose the four local YAMLs into one stable ai: category namespace."""
    if args.no_hierarchy:
        return None
    from stash_ai.exclusions import policies_or_empty
    tag_policies = policies_or_empty(tag_policies)
    prefix, root = "ai:category:", "ai:category:AI"
    leaves = defaultdict(set)
    categories = {root: ()}
    for key, source in SOURCES.items():
        path = args.hierarchy if key == "joytag" else getattr(args, key + "_hierarchy")
        data = app.load_unique_yaml(path, "Hierarchy")
        if not isinstance(data, dict) or not isinstance(data.get("groups"), dict):
            raise ValueError(f"Invalid hierarchy: {path}")
        groups = data["groups"].values()
        if any(not isinstance(group, list) or any(not isinstance(label, str) for label in group) for group in groups):
            raise ValueError(f"Invalid hierarchy labels: {path}")
        labels = sorted({label for group in groups for label in group})
        hierarchy = app.load_tag_hierarchy(path, labels, source)
        def converted(name):
            return root if not hierarchy.category_parents[name] else prefix + name[len(hierarchy.category_prefix):]
        for name, parents in hierarchy.category_parents.items():
            mapped, wanted = converted(name), tuple(converted(parent) for parent in parents)
            if mapped in categories and categories[mapped] != wanted:
                raise ValueError(f"Conflicting AI hierarchy category: {mapped}")
            categories[mapped] = wanted
        for label, parents in hierarchy.leaf_parents.items():
            merge.raw_label(label)
            if label in tag_policies[key]:
                continue
            leaves[label].update(converted(parent) for parent in parents)
    fallbacks = {prefix + suffix for suffix in ("Other / Unclassified", "General", "Uncategorized")}
    for parents in leaves.values():
        if parents - fallbacks:
            parents.difference_update(fallbacks)
    if set(categories) & {merge.AI.prefix + label for label in leaves}:
        raise ValueError("A merged hierarchy category collides with a predicted tag.")
    return TagHierarchy({label: tuple(sorted(parents)) for label, parents in leaves.items()},
                            categories, prefix, merge.AI.hierarchy_description)


def run_hierarchy(app, args, catalogs, hierarchy, tag_policies=None):
    from stash_ai.exclusions import policies_or_empty
    tag_policies = policies_or_empty(tag_policies)
    stash = app.create_stash_client(app.DRY_RUN)
    try:
        resolver = app.TagResolver(stash, hierarchy, source=merge.AI, report=args.report.detail, create_missing=app.CREATE_MISSING)
        resolver.preload()
        labels = sorted({label for source, vocabulary in catalogs.items() for label in vocabulary.labels
                         if label not in tag_policies[source]})
        for label in labels:
            if args.create_all_tags or merge.AI.prefix + label in resolver.cache:
                resolver.resolve(merge.AI.prefix + label)
        app.print_hierarchy_summary(resolver)
    finally:
        stash.close()
    return 0
