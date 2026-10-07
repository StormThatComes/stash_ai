"""Stash scene, tag and marker operations built on the shared HTTP transport."""
from __future__ import annotations

import re
from itertools import batched
from typing import Iterator

from stash_ai.stash_api.transport import StashTransport
from stash_ai.stash_api.scene_reads import SceneReads


class StashClient(StashTransport):
    """Build domain queries for Stash 0.31.1+ and manage SQL read fallback."""

    def __init__(self, dry_run: bool, *, url: str, api_key: str,
                 request_timeout: float | tuple[float, float], http_retries: int, scene_page_size: int,
                 create_markers: bool):
        super().__init__(dry_run, url=url, api_key=api_key,
                         request_timeout=request_timeout, http_retries=http_retries)
        self.scene_page_size = scene_page_size
        self.create_markers = create_markers
        self.scene_access = SceneReads(self)
        try:
            self._require_supported_version()
        except Exception:
            self.close()
            raise

    def _require_supported_version(self):
        data = self.gql('query StashAIVersion { version { version } }')
        version = (data.get('version') or {}).get('version')
        match = re.fullmatch(r'v?(\d+)\.(\d+)\.(\d+)(?:-([0-9A-Za-z.-]+))?'
                             r'(?:\+[0-9A-Za-z.-]+)?', version) if isinstance(version, str) else None
        if match:
            release = tuple(map(int, match.group(1, 2, 3)))
            suffix = match.group(4)
            # Stash development builds use git describe: v0.31.1-175-gabcdef.
            post_release = suffix is None or re.fullmatch(r'\d+-g[0-9a-f]+(?:-dirty)?|dirty', suffix)
            if release > (0, 31, 1) or (release == (0, 31, 1) and post_release):
                return
        raise RuntimeError(f'Stash >= 0.31.1 is required; server reported {version!r}. '
                           'Upgrade Stash before running commands that connect to it.')

    @property
    def merge_reader(self):
        return self.scene_access.reader

    @property
    def merge_only_reads(self):
        return self.scene_access.merge_only

    def configure_merge_reads(self, report, *, merge_only=False):
        self.report = report
        self.scene_access.configure(report, merge_only=merge_only)

    def merge_snapshots(self, ids):
        return self.scene_access.snapshots(ids)

    def find_scene_for_merge(self, scene_id):
        return self.scene_access.find_for_merge(scene_id)

    def tag_fields(self) -> str:
        return "id name description ignore_auto_tag parents { id name description }"

    def iter_tags(self) -> Iterator[dict]:
        page = 1
        while True:
            result = self.gql('''query StashAITagCatalog($filter: FindFilterType!) {
                findTags(filter: $filter) { tags { ''' + self.tag_fields() + ''' } }
            }''', {"filter": {"page": page, "per_page": 1000, "sort": "id", "direction": "ASC"}})["findTags"]["tags"]
            yield from result
            if len(result) < 1000:
                return
            page += 1

    def find_tag_by_id(self, tag_id: str, *, required: bool = True) -> dict | None:
        result = self.gql('''query StashAITagByID($id: ID!) {
            findTag(id: $id) { ''' + self.tag_fields() + ''' }
        }''', {"id": tag_id})["findTag"]
        if not result and required:
            raise RuntimeError(f"Tag {tag_id} disappeared during hierarchy synchronization.")
        return result

    def iter_scenes(self, ids: list[str] | None = None) -> Iterator[dict]:
        return self.scene_access.iter_scenes(ids)

    def scene_fields(self) -> str:
        marker_fields = ""
        if self.create_markers:
            marker_fields = "scene_markers { id title seconds end_seconds primary_tag { id name } tags { id name } }"
        return "id title paths { sprite vtt } tags { id name } " + marker_fields

    def find_scene(self, scene_id: str) -> dict:
        return self.scene_access.find(scene_id)

    def find_tag(self, name: str) -> dict | None:
        tags = self.gql('''query StashAIFindTag($name: String!) {
            findTags(tag_filter: {name: {value: $name, modifier: EQUALS}},
                     filter: {per_page: -1}) { tags { ''' + self.tag_fields() + ''' } }
        }''', {"name": name})["findTags"]["tags"]
        return next((tag for tag in tags if tag["name"] == name), None)

    def create_tag(self, name: str, *, parent_ids: list[str] | None = None,
                   description: str | None = None, ignore_auto_tag: bool | None = None) -> dict:
        payload: dict = {"name": name}
        for key, value in (("parent_ids", parent_ids), ("description", description), ("ignore_auto_tag", ignore_auto_tag)):
            if value is not None:
                payload[key] = value
        result = self.gql('''mutation StashAICreateTag($input: TagCreateInput!) {
            tagCreate(input: $input) { ''' + self.tag_fields() + ''' }
        }''', {"input": payload})["tagCreate"]
        if not result:
            raise RuntimeError(f"Stash did not create tag {name!r}.")
        return result

    def update_tag_organization(self, tag_id: str, parent_ids: list[str], ignore_auto_tag: bool) -> dict:
        result = self.gql('''mutation StashAIMarkerGroupTag($input: TagUpdateInput!) {
            tagUpdate(input: $input) { ''' + self.tag_fields() + ''' }
        }''', {"input": {"id": tag_id, "parent_ids": parent_ids, "ignore_auto_tag": ignore_auto_tag}})["tagUpdate"]
        if not result:
            raise RuntimeError(f"Stash did not update marker group tag {tag_id}.")
        return result

    def update_tag_parents(self, tag_id: str, parent_ids: list[str]) -> dict:
        result = self.gql('''mutation StashAITagParents($input: TagUpdateInput!) {
            tagUpdate(input: $input) { ''' + self.tag_fields() + ''' }
        }''', {"input": {"id": tag_id, "parent_ids": parent_ids}})["tagUpdate"]
        if not result:
            raise RuntimeError(f"Stash did not update parents of tag {tag_id}.")
        return result

    def update_scene_tags(self, scene_id: str, tag_ids: list[str]) -> None:
        result = self.gql('''mutation StashAISceneTags($input: SceneUpdateInput!) {
            sceneUpdate(input: $input) { id }
        }''', {"input": {"id": scene_id, "tag_ids": tag_ids}})["sceneUpdate"]
        if not result:
            raise RuntimeError("Stash did not update scene tags.")

    def create_marker(self, payload: dict) -> str:
        result = self.gql('''mutation StashAICreateMarker($input: SceneMarkerCreateInput!) {
            sceneMarkerCreate(input: $input) { id }
        }''', {"input": payload})["sceneMarkerCreate"]
        if not result:
            raise RuntimeError("Stash did not create the scene marker.")
        return str(result['id'])

    def update_marker(self, payload: dict) -> None:
        result = self.gql('''mutation StashAIUpdateMarker($input: SceneMarkerUpdateInput!) {
            sceneMarkerUpdate(input: $input) { id }
        }''', {"input": payload})["sceneMarkerUpdate"]
        if not result:
            raise RuntimeError("Stash did not update the scene marker.")

    def delete_marker(self, marker_id: str) -> None:
        result = self.gql('''mutation StashAIDeleteMarker($id: ID!) {
            sceneMarkerDestroy(id: $id)
        }''', {"id": marker_id})["sceneMarkerDestroy"]
        if not result:
            raise RuntimeError(f"Stash did not delete marker {marker_id}.")

    def delete_markers(self, marker_ids):
        """Bounded native mutations; ambiguous responses leave durable recovery."""
        for group in batched(dict.fromkeys(map(str, marker_ids)), 200):
            result = self.gql('''mutation StashAIDeleteMarkers($ids: [ID!]!) {
                sceneMarkersDestroy(ids: $ids)
            }''', {'ids': list(group)})['sceneMarkersDestroy']
            if result is not True:
                raise RuntimeError('Marker batch deletion was not confirmed; resume the pending merge.')
