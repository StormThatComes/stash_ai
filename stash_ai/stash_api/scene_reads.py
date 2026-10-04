"""Coordinate SQL snapshots, GraphQL media fields and one-way read fallback."""
from itertools import batched
from typing import Iterator

from stash_ai.stash_api.sql import SceneReader
from stash_ai.stash_api.transport import safe_error


class SceneReads:
    """Own one client's read strategy; failures disable SQL until reconfigured."""
    def __init__(self, client):
        self.client = client
        self.reader, self.report, self.merge_only = None, None, False

    def _fallback(self, exc, operation):
        self.reader = None
        if self.report is not None:
            self.report.counts['merge_graphql_fallback'] += 1
            self.report.detail(operation + '; using GraphQL: ' + safe_error(exc, self.client.api_key))

    def configure(self, report, *, merge_only=False):
        self.report, self.merge_only = report, merge_only
        try:
            reader = SceneReader(self.client, self.client.create_markers, report)
            reader.read([])
            self.reader = reader
            report.counts['merge_sql_enabled'] += 1
        except Exception as exc:
            self._fallback(exc, 'Merge SQL unavailable')

    def _sql_read(self, method, *args):
        if self.reader is None:
            return None
        try:
            return getattr(self.reader, method)(*args)
        except Exception as exc:
            self._fallback(exc, 'Merge SQL ' + method + ' failed')
            return None

    def snapshots(self, ids):
        return self._sql_read('read', ids)

    def page(self, after, size):
        return self._sql_read('page', after, size)

    def find_for_merge(self, scene_id):
        scenes = self.snapshots([scene_id])
        if scenes is None:
            return self.find(scene_id)
        scene = scenes.get(str(int(scene_id)))
        if scene is None:
            raise RuntimeError(f'Scene {scene_id} no longer exists.')
        return scene

    def iter_scenes(self, ids: list[str] | None = None) -> Iterator[dict]:
        if ids:
            for group in batched(dict.fromkeys(ids), min(self.client.scene_page_size, 1000)):
                snapshots = (self.snapshots(group)
                             if self.merge_only else None)
                for scene_id in group:
                    if snapshots is None:
                        yield self.find(scene_id)
                    else:
                        scene = snapshots.get(str(int(scene_id)))
                        if scene is None:
                            raise RuntimeError(f'Scene {scene_id} no longer exists.')
                        yield scene
            return
        after = 0
        if self.merge_only and self.reader:
            while True:
                scenes = self.page(after, min(self.client.scene_page_size, 1000))
                if scenes is None:
                    break
                yield from scenes.values()
                if len(scenes) < min(self.client.scene_page_size, 1000):
                    return
                after = max(map(int, scenes))
        page = 1
        while True:
            reader = self.reader
            result = self.client.gql('''query StashAIScenes($filter: FindFilterType!) {
                findScenes(filter: $filter) {
                    scenes { ''' + ('id title paths { sprite vtt }' if reader else self.client.scene_fields()) + ''' }
                }
            }''', {"filter": {"page": page, "per_page": self.client.scene_page_size,
                              "sort": "id", "direction": "ASC"}})["findScenes"]["scenes"]
            if reader and result:
                snapshots = self.snapshots([scene['id'] for scene in result])
                for scene in result:
                    if snapshots is None:
                        yield self.find(scene['id'])
                    elif scene['id'] in snapshots:
                        yield {**scene, **snapshots[scene['id']]}
                    # A scene removed between the two read requests is skipped.
            else:
                yield from (scene for scene in result if not after or int(scene['id']) > after)
            if len(result) < self.client.scene_page_size:
                return
            page += 1

    def find(self, scene_id: str) -> dict:
        reader = self.reader
        if reader and self.merge_only:
            return self.find_for_merge(scene_id)
        data = self.client.gql("""query StashAIScene($id: ID!) {
            findScene(id: $id) {
                """ + ('id title paths { sprite vtt }' if reader else self.client.scene_fields()) + """
            }
        }""", {"id": scene_id})["findScene"]
        if not data:
            raise RuntimeError(f"Scene {scene_id} no longer exists.")
        if reader:
            scenes = self.snapshots([data['id']])
            if scenes is None:
                return self.find(scene_id)
            if data['id'] not in scenes:
                raise RuntimeError(f'Scene {scene_id} no longer exists.')
            data.update(scenes[data['id']])
        return data

