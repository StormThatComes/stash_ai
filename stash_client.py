"""Stash HTTP/GraphQL access with explicit per-client settings."""
from __future__ import annotations

import re
from typing import Iterator
from urllib.parse import urljoin, urlparse


def absolute_stash_url(path_or_url: str, base_url: str) -> str:
    return urljoin(base_url.rstrip("/") + "/", path_or_url)


def origin(url: str) -> tuple[str, str | None, int]:
    parsed = urlparse(url)
    return parsed.scheme, parsed.hostname, parsed.port or (443 if parsed.scheme == "https" else 80)


def safe_error(exc: Exception, api_key: str = "") -> str:
    """Avoid exposing API keys embedded in Stash's media URLs or HTTP errors."""
    message = str(exc)
    if api_key:
        message = message.replace(api_key, "[redacted]")
    return re.sub(r"(?i)([?&](?:apikey|api_key|token)=)[^&\s\"'<>]+", r"\1[redacted]", message)


class StashClient:
    """Small GraphQL client with separate media retries and a write guard."""

    def __init__(self, dry_run: bool, *, url: str, api_key: str,
                 request_timeout: float, http_retries: int, scene_page_size: int,
                 create_markers: bool):
        import requests
        from requests.adapters import HTTPAdapter
        from urllib3.util.retry import Retry

        self.url = url
        self.api_key = api_key
        self.request_timeout = request_timeout
        self.http_retries = http_retries
        self.scene_page_size = scene_page_size
        self.create_markers = create_markers
        self.dry_run = dry_run
        self.has_end_time = False
        self.has_tag_hierarchy = False
        self.merge_reader = None
        self.merge_only_reads = False
        self.report = None
        self.api = requests.Session()  # No POST retries, including connection retries.
        self.media = requests.Session()
        if self.api_key:
            self.api.headers["ApiKey"] = self.api_key
            self.media.headers["ApiKey"] = self.api_key
        adapter = HTTPAdapter(max_retries=Retry(
            total=http_retries, backoff_factor=0.5,
            status_forcelist=(429, 500, 502, 503, 504),
            allowed_methods=frozenset({"GET"}),
        ))
        self.media.mount("http://", adapter)
        self.media.mount("https://", adapter)

    def close(self) -> None:
        self.api.close()
        self.media.close()

    @staticmethod
    def _read_only(query: str, variables: dict | None = None) -> bool:
        if not query.lstrip().startswith('mutation'):
            return True
        from stash_cleanup_sql import is_read_only_cleanup
        from stash_merge_sql import is_read_only_merge
        return is_read_only_cleanup(query, variables) or is_read_only_merge(query, variables)

    def gql(self, query: str, variables: dict | None = None) -> dict:
        report = getattr(self, 'report', None)
        if report is None:
            return self._gql(query, variables)
        read = self._read_only(query, variables)
        # Names are from our fixed operations, never from variables or payloads.
        match = re.search(r'\b(?:query|mutation)\s+(StashAI[A-Za-z0-9_]*)\b', query)
        operation = match.group(1) if match else ('read' if read else 'write')
        if operation == 'StashAIMergeReadSQL':
            from stash_merge_sql import SQL
            operation += '.' + next((mode for mode, sql in SQL.items() if sql == (variables or {}).get('sql')), 'unknown')
        with report.request(operation, read):
            return self._gql(query, variables)

    def _gql(self, query: str, variables: dict | None = None) -> dict:
        read = self._read_only(query, variables)
        if self.dry_run and query.lstrip().startswith("mutation") and not read:
            raise RuntimeError("Internal error: mutation attempted during dry-run.")
        from requests import ConnectionError as RequestsConnectionError, Timeout as RequestsTimeout
        from stash_runtime import stage
        import time
        report = getattr(self, 'report', None)
        attempts = self.http_retries + 1 if read else 1
        retry_statuses = {429, 500, 502, 503, 504}
        for attempt in range(attempts):
            try:
                with stage(report, 'stash_http'):
                    response = self.api.post(
                        absolute_stash_url("graphql", self.url),
                        json={"query": query, "variables": variables or {}},
                        timeout=self.request_timeout, allow_redirects=False,
                    )
            except (RequestsConnectionError, RequestsTimeout):
                if attempt + 1 == attempts:
                    raise
                time.sleep(0.5 * 2 ** attempt)
                continue
            if report is not None and isinstance(getattr(response, 'content', None), bytes):
                report.response_size(len(response.content))
            if response.status_code not in retry_statuses or attempt + 1 == attempts:
                break
            time.sleep(0.5 * 2 ** attempt)
        response.raise_for_status()
        if 300 <= response.status_code < 400:
            raise RuntimeError("Stash redirected GraphQL; configure STASH_URL to its final URL.")
        with stage(report, 'stash_json_decode'):
            payload = response.json()
        if payload.get("errors"):
            raise RuntimeError("GraphQL: " + "; ".join(
                error.get("message", "Unknown error") for error in payload["errors"]
            ))
        if not isinstance(payload.get("data"), dict):
            raise RuntimeError("Stash returned no GraphQL data.")
        return payload["data"]

    def configure_merge_reads(self, report, *, merge_only=False):
        from stash_merge_sql import SceneReader
        self.report = report
        self.merge_only_reads = merge_only
        try:
            reader = SceneReader(self, self.create_markers, report)
            reader.read([])  # Verify the exact SELECT/schema/JSON support, once.
            self.merge_reader = reader
            report.counts['merge_sql_enabled'] += 1
        except Exception as exc:
            self.merge_reader = None
            report.counts['merge_graphql_fallback'] += 1
            report.detail('Merge SQL unavailable; using GraphQL: ' + safe_error(exc, self.api_key))

    def merge_snapshots(self, ids):
        reader = getattr(self, 'merge_reader', None)
        if reader is None:
            return None
        try:
            return reader.read(ids)
        except Exception as exc:
            self.merge_reader = None
            if self.report is not None:
                self.report.counts['merge_graphql_fallback'] += 1
                self.report.detail('Merge SQL read failed; using GraphQL: ' + safe_error(exc, self.api_key))
            return None

    def find_scene_for_merge(self, scene_id):
        scenes = self.merge_snapshots([scene_id])
        if scenes is None:
            return self.find_scene(scene_id)
        scene = scenes.get(str(int(scene_id)))
        if scene is None:
            raise RuntimeError(f'Scene {scene_id} no longer exists.')
        return scene

    def detect_marker_capabilities(self) -> None:
        data = self.gql('''query StashAIMarkerCapabilities {
            create: __type(name: "SceneMarkerCreateInput") { inputFields { name } }
            update: __type(name: "SceneMarkerUpdateInput") { inputFields { name } }
            marker: __type(name: "SceneMarker") { fields { name } }
        }''')
        if not all(data.get(name) for name in ("create", "update", "marker")):
            raise RuntimeError("This Stash server does not expose the scene marker API.")
        self.has_end_time = all(
            any(field["name"] == "end_seconds" for field in data[name][key])
            for name, key in (("create", "inputFields"), ("update", "inputFields"), ("marker", "fields"))
        )

    def detect_tag_hierarchy(self) -> None:
        data = self.gql('''query StashAIHierarchyCapabilities {
            create: __type(name: "TagCreateInput") { inputFields { name } }
            update: __type(name: "TagUpdateInput") { inputFields { name } }
            tag: __type(name: "Tag") { fields { name } }
        }''')
        self.has_tag_hierarchy = all(
            required <= {field["name"] for field in (data.get(name) or {}).get(key, [])}
            for name, key, required in (
                ("create", "inputFields", {"parent_ids", "description", "ignore_auto_tag"}),
                ("update", "inputFields", {"parent_ids", "ignore_auto_tag"}),
                ("tag", "fields", {"parents", "description", "ignore_auto_tag"}),
            )
        )
        if not self.has_tag_hierarchy:
            raise RuntimeError("Stash does not expose the required tag hierarchy/settings API. Update Stash; "
                               "merged marker groups require hierarchy support even with --no-hierarchy.")

    def tag_fields(self) -> str:
        return "id name description ignore_auto_tag parents { id name description }" if self.has_tag_hierarchy else "id name"

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
        if ids:
            for scene_id in dict.fromkeys(ids):
                yield (self.find_scene_for_merge(scene_id) if getattr(self, 'merge_only_reads', False)
                       else self.find_scene(scene_id))
            return
        after = 0
        if getattr(self, 'merge_only_reads', False) and getattr(self, 'merge_reader', None):
            while True:
                try:
                    scenes = self.merge_reader.page(after, min(self.scene_page_size, 1000))
                except Exception as exc:
                    self.merge_reader = None
                    self.report.counts['merge_graphql_fallback'] += 1
                    self.report.detail('Merge SQL page failed; using GraphQL: ' + safe_error(exc, self.api_key))
                    break
                yield from scenes.values()
                if len(scenes) < min(self.scene_page_size, 1000):
                    return
                after = max(map(int, scenes))
        page = 1
        while True:
            reader = getattr(self, 'merge_reader', None)
            result = self.gql('''query StashAIScenes($filter: FindFilterType!) {
                findScenes(filter: $filter) {
                    scenes { ''' + ('id title paths { sprite vtt }' if reader else self.scene_fields()) + ''' }
                }
            }''', {"filter": {"page": page, "per_page": self.scene_page_size,
                              "sort": "id", "direction": "ASC"}})["findScenes"]["scenes"]
            if reader and result:
                snapshots = self.merge_snapshots([scene['id'] for scene in result])
                for scene in result:
                    if snapshots is None:
                        yield self.find_scene(scene['id'])
                    elif scene['id'] in snapshots:
                        yield {**scene, **snapshots[scene['id']]}
                    # A scene removed between the two read requests is skipped.
            else:
                yield from (scene for scene in result if not after or int(scene['id']) > after)
            if len(result) < self.scene_page_size:
                return
            page += 1

    def scene_fields(self) -> str:
        marker_fields = ""
        if self.create_markers:
            end = "end_seconds" if self.has_end_time else ""
            marker_fields = f"scene_markers {{ id title seconds {end} primary_tag {{ id name }} tags {{ id name }} }}"
        return "id title paths { sprite vtt } tags { id name } " + marker_fields

    def find_scene(self, scene_id: str) -> dict:
        reader = getattr(self, 'merge_reader', None)
        data = self.gql("""query StashAIScene($id: ID!) {
            findScene(id: $id) {
                """ + ('id title paths { sprite vtt }' if reader else self.scene_fields()) + """
            }
        }""", {"id": scene_id})["findScene"]
        if not data:
            raise RuntimeError(f"Scene {scene_id} no longer exists.")
        if reader:
            scenes = self.merge_snapshots([data['id']])
            if scenes is None:
                return self.find_scene(scene_id)
            if data['id'] not in scenes:
                raise RuntimeError(f'Scene {scene_id} no longer exists.')
            data.update(scenes[data['id']])
        return data

    def get_media(self, url: str) -> bytes:
        # ApiKey is a custom header; requests does not remove it on redirects.
        # Follow same-origin redirects explicitly, never forwarding it elsewhere.
        for _ in range(6):
            if origin(url) != origin(self.url):
                raise ValueError("Media URL uses a different Stash origin; check STASH_URL.")
            response = self.media.get(url, timeout=self.request_timeout, allow_redirects=False)
            if response.status_code in (301, 302, 303, 307, 308):
                location = response.headers.get("Location")
                if not location:
                    raise RuntimeError("Media redirect has no Location header.")
                url = urljoin(url, location)
                continue
            response.raise_for_status()
            return response.content
        raise RuntimeError("Too many media redirects.")

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
