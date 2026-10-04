"""Remove excluded model-prefixed scene associations."""
from collections import Counter
from stash_ai.models import ModelSource
from stash_ai.stash_api.client import StashClient


def cleanup_scene_tags(stash: StashClient, scene: dict, exclusions: frozenset[str],
                       source: ModelSource, report=None) -> Counter:
    existing = scene.get("tags") or []
    removed = [tag for tag in existing if tag["name"].startswith(source.prefix)
               and tag["name"][len(source.prefix):] in exclusions]
    if removed:
        if report:
            report(f"{'[DRY] ' if stash.dry_run else ''}{source.key} scene {scene['id']}: remove "
                   + ", ".join(tag["name"] for tag in removed))
        retained = {str(tag["id"]) for tag in existing} - {str(tag["id"]) for tag in removed}
        if not stash.dry_run:
            stash.update_scene_tags(str(scene["id"]), sorted(retained))
    return Counter(scene_tags_updated=int(bool(removed)), scene_tags_cleaned=len(removed))
