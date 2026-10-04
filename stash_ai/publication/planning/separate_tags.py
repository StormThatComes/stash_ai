"""Resolve separate-source names and preserve allowed scene associations."""


def _resolved_names(app, resolver, source, scene_predictions, specs, scene_exclusions):
    scene_names = (
        {
            source.prefix + tag
            for tag, _ in scene_predictions
            if tag not in scene_exclusions
        }
        if app.APPLY_SCENE_TAGS
        else set()
    )
    names = set(scene_names)
    if app.CREATE_MARKERS and specs:
        names.add(source.owner_tag)
        for spec in specs:
            names.add(source.prefix + spec.tag)
            names.update(source.prefix + tag for tag in spec.extra_tags)
    ids = {name: resolver.resolve(name) for name in sorted(names)}
    return scene_names, ids


def _scene_tag_plan(app, source, existing_tags, scene_names, ids, scene_exclusions):
    existing_ids = {str(tag["id"]) for tag in existing_tags}
    preserved = {
        str(tag["id"])
        for tag in existing_tags
        if not (
            tag["name"].startswith(source.prefix)
            and (
                app.REPLACE_SCENE_TAGS
                or tag["name"][len(source.prefix):] in scene_exclusions
            )
        )
    }
    final_ids = preserved | {
        ids[name]
        for name in scene_names
        if ids[name] is not None
    }
    scene_changed = app.APPLY_SCENE_TAGS and final_ids != existing_ids
    cleaned = (
        sum(
            tag["name"].startswith(source.prefix)
            and tag["name"][len(source.prefix):] in scene_exclusions
            for tag in existing_tags
        )
        if app.APPLY_SCENE_TAGS
        else 0
    )
    return final_ids, scene_changed, cleaned
