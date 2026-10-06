"""Global output exclusions, separate from primary-only marker exclusions.

Raw inference records stay intact so changing YAML never requires a model run.
"""
from dataclasses import replace

from stash_ai.models import SOURCES


def require_policies(policies):
    """Internal callers provide every model, including explicitly empty bans."""
    if (not isinstance(policies, dict) or set(policies) != set(SOURCES)
            or any(not isinstance(values, (set, frozenset)) for values in policies.values())):
        raise ValueError('Exclusion policies require an explicit set for every model.')
    return policies


def selected_policies(policies, selected):
    return {source: values if source in selected else frozenset()
            for source, values in require_policies(policies).items()}


def core_policies_with_global(core_policies, tag_policies):
    return {source: core_policies[source] | tag_policies[source] for source in SOURCES}


def excluded_names(policies, *, unused=False):
    """Shared names are banned only if every source is banned.

    For unused definitions the union is safe: usage is checked before deletion.
    """
    policies = require_policies(policies)
    shared = (frozenset.union if unused else frozenset.intersection)(*policies.values())
    return {'ai:' + label for label in shared}


def filter_spec(spec, excluded):
    if not excluded:
        return spec
    return replace(spec, extra_tags=tuple(tag for tag in spec.extra_tags if tag not in excluded),
                   extra_evidence=tuple(item for item in spec.extra_evidence if item[0] not in excluded),
                   extra_support={tag: value for tag, value in spec.extra_support.items() if tag not in excluded})


def prune_excluded_definitions(app, stash, policies, report):
    """Delete only now-unused excluded definitions, after a successful full sync."""
    names = excluded_names(policies, unused=True)
    if not names or not (app.APPLY_SCENE_TAGS and app.CREATE_MARKERS):
        return {}
    from stash_ai.cleanup.prune import prune_tags
    return prune_tags(stash, report=report, names=names)
