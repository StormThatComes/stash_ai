"""Read capabilities used by cleanup plans, independent of a concrete client."""
from typing import Iterable, Protocol


class QueryReads(Protocol):
    dry_run: bool

    def gql(self, query: str, variables: dict | None = None) -> dict: ...


class TagReads(QueryReads, Protocol):
    def iter_tags(self) -> Iterable[dict]: ...
    def find_tag(self, name: str) -> dict | None: ...
    def find_tag_by_id(self, identifier: str, *, required: bool = True) -> dict | None: ...


class SceneReads(TagReads, Protocol):
    def find_scene(self, identifier: str) -> dict: ...
    def find_scene_for_merge(self, identifier: str) -> dict: ...
    def iter_scenes(self, identifiers: Iterable[str] | None = None) -> Iterable[dict]: ...


class ReadTagView:
    """An explicit preview surface; no delegated mutation/transport attributes."""
    dry_run = True

    def gql(self, query, variables=None):
        if not query.lstrip().startswith('query '):
            raise ValueError('Preview accepts only named GraphQL read queries.')
        return self.stash.gql(query, variables)

    def iter_tags(self):
        return self.stash.iter_tags()

    def find_tag(self, name):
        return self.stash.find_tag(name)

    def find_tag_by_id(self, identifier, *, required=True):
        return self.stash.find_tag_by_id(identifier, required=required)
