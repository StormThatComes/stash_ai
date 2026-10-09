"""Bounded dry-run cleanup effects and scene projection, without source writes."""
import json
from collections import Counter
from stash_ai.publication import markers as marker_ops
from stash_ai.stash_api.protocols import ReadTagView, SceneReads


class PreviewEffects:
    def __init__(self):
        self.names, self.cores = set(), {}
        self.group_usage_delta = Counter()

    def survivors(self, session, catalog, ids):
        """Do not count extra removals on markers already planned for deletion."""
        rows = session.read('marker_metadata', (json.dumps(ids),))
        removed = {int(i) for i, name in rows if name in self.names}
        core_groups = {}
        for i, name in rows:
            if name in self.cores:
                core_groups.setdefault(name, []).append(int(i))
        for name, group in core_groups.items():
            source = self.cores[name]
            p = source.title_prefix
            params = (catalog.ids_by_name[name], name, p, p, p, source.owner_tag, json.dumps(group))
            removed.update(int(row[0]) for row in session.read('primary_check', params))
        return [i for i in ids if i not in removed]


class SQLPreview(ReadTagView):
    """Overlay global dry-run effects without retaining millions of marker IDs."""
    def __init__(self, stash: SceneReads, app, names, cores, group_usage_delta=None):
        self.stash, self.app, self.names, self.cores = stash, app, names, cores
        self.marker_group_usage_delta = group_usage_delta or Counter()

    def find_scene(self, identifier):
        return self._project(self.stash.find_scene(identifier))

    def find_scene_for_merge(self, identifier):
        return self._project(self.stash.find_scene_for_merge(identifier))

    def _project(self, scene):
        from copy import deepcopy
        scene = deepcopy(scene)
        if self.app.APPLY_SCENE_TAGS:
            scene['tags'] = [t for t in scene.get('tags', []) if t['name'] not in self.names]
        if self.app.CREATE_MARKERS:
            scene['scene_markers'] = [m for m in scene.get('scene_markers', []) if
                m['primary_tag']['name'] not in self.names and not (
                    m['primary_tag']['name'] in self.cores and marker_ops.owned_marker(m, self.cores[m['primary_tag']['name']]))]
            for marker in scene['scene_markers']:
                marker['tags'] = [t for t in marker.get('tags', []) if t['name'] not in self.names]
        return scene

    def iter_scenes(self, ids):
        if not ids:
            return
        for scene in self.stash.iter_scenes(ids):
            yield self._project(scene)

