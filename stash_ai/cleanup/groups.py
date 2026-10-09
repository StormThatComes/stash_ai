"""Prune managed leaf groups using native usage or dry-run membership deltas."""
from collections import Counter
from copy import deepcopy
from stash_ai.publication.groups import PREFIX
from stash_ai.stash_api.protocols import ReadTagView, TagReads


class PreviewUsage(ReadTagView):
    """Apply planned membership deltas to native, global tag usage counters."""
    def __init__(self, stash: TagReads, delta):
        self.stash, self.delta = stash, delta

    def gql(self, query, variables=None):
        data = super().gql(query, variables)
        if not any(name in query for name in ('StashAIPruneCandidates', 'StashAIPruneCatalog', 'StashAIPruneTag(')):
            return data
        data = deepcopy(data)
        tags = data.get('findTags', {}).get('tags', [])
        if data.get('findTag'):
            tags = [data['findTag']]
        for tag in tags:
            if type(tag.get('scene_marker_count')) is int:
                tag['scene_marker_count'] = max(0, tag['scene_marker_count'] + self.delta.get(str(tag['id']), 0))
        return data


def prune_groups(stash, report=print, *, candidate_ids=None, usage_delta=None):
    """Only owned, globally unused leaf groups are pruned; the root is retained."""
    from stash_ai.cleanup.prune import prune_tags
    if candidate_ids is None:
        candidate_ids = [str(t['id']) for t in stash.iter_tags() if t['name'].startswith(PREFIX)]
    if not candidate_ids:
        return Counter()
    view = PreviewUsage(stash, usage_delta) if stash.dry_run and usage_delta else stash
    stats = Counter(prune_tags(view, [PREFIX], report=report, candidate_ids=candidate_ids))
    return Counter(marker_group_tags_pruned=stats['tags_pruned'],
                   marker_group_tags_retained=stats['tags_retained_after_refresh'])
