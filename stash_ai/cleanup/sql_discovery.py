"""Bounded SELECT-only cleanup candidate discovery and prune prefilters."""
from stash_ai.runtime.console import progress as console_bar
from stash_ai.runtime.console import status
import json
from itertools import batched
from tqdm import tqdm


class SQLDiscovery:
    def __init__(self, session, catalog):
        self.session, self.catalog = session, catalog
        self.stash, self.stats = session.stash, session.stats

    def candidates(self, catalog, scene_names, primary_names, extra_names):
        scenes = set()
        applicable = [(identifier, name) for identifier, name in catalog.items()
                      if name in scene_names | primary_names | extra_names]
        report = getattr(self.stash, 'report', None)
        with console_bar(tqdm, total=len(applicable), desc='Discover cleanup scenes', unit='tag') as bar:
            for identifier, name in applicable:
                for key, names, bound, params in (
                    ('primary_scenes', primary_names, self.session.bounds[0], (int(identifier),)),
                    ('extra_scenes', extra_names, self.session.bounds[1], (int(identifier),)),
                    ('scenes', scene_names, self.session.bounds[2], (int(identifier), name))):
                    if name in names:
                        status(report, relation=key, discovery_tags=f'{bar.n}/{bar.total}')
                        for rows in self.session.pages(key, params, bound):
                            scenes.update(str(row[1]) for row in rows)
                            status(report, candidates=len(scenes))
                bar.update(1)
                status(report, discovery_tags=f'{bar.n}/{bar.total}', candidates=len(scenes))
        return scenes

    def unused_candidates(self, ids, *, page_size=200, progress=None):
        """Reject definitely-used tags; native usage/ownership checks still decide deletion."""
        candidates = set()
        for group in batched(sorted(set(ids)), page_size):
            found = {str(row[0]) for row in self.session.read('unused_candidates', (json.dumps(list(map(int, group))),))}
            if not found <= set(group):
                raise ValueError('Unexpected prune candidate IDs; stopped before mutation.')
            candidates.update(found)
            if progress is not None:
                progress.set_postfix(candidates=len(candidates), refresh=False)
                progress.update(len(group))
        return candidates

    def marker_group_candidates(self, usage_delta=None):
        from stash_ai.publication.groups import PREFIX
        selected = {i for i, name in self.catalog.records.items() if name.startswith(PREFIX)}
        if not selected:
            return []
        with console_bar(tqdm, total=len(selected), desc='Find empty marker groups', unit='tag', dynamic_ncols=True) as bar:
            candidates = self.unused_candidates(selected, page_size=self.session.page_size, progress=bar)
        if self.stash.dry_run:
            # SQL sees the real database, not planned removals. Let PreviewUsage
            # check groups losing members, even when SQL still finds their tags.
            candidates.update(i for i, change in (usage_delta or {}).items() if change < 0 and i in selected)
        self.stats['marker_group_tags_checked_sql'] += len(selected)
        self.stats['marker_group_tags_candidates'] += len(candidates)
        return sorted(candidates)

