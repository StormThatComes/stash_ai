"""Resolve model tags and synchronize their managed hierarchy."""
from __future__ import annotations

from collections import Counter
from typing import TYPE_CHECKING

from stash_ai.domain import TagHierarchy
from stash_ai.models import ModelSource

if TYPE_CHECKING:
    from stash_ai.client import StashClient


class TagResolver:
    def __init__(self, stash: StashClient, hierarchy: TagHierarchy | None = None, report=None,
                 *, source: ModelSource, exclusions: frozenset[str] = frozenset(),
                 create_missing: bool = True):
        self.stash = stash
        self.create_missing = create_missing
        self.hierarchy = hierarchy
        self.source = source
        self.exclusions = exclusions
        self.report = report
        self.cache: dict[str, str | None] = {}
        self.records: dict[str, dict | None] = {}
        self.names_by_id: dict[str, str | None] = {}
        self.missing: set[str] = set()
        self.synced: set[str] = set()
        self.catalog_loaded = False
        self.hierarchy_stats = Counter()

    def preload(self) -> None:
        """Read all tags for a hierarchy-only backfill, avoiding 5,813 lookups."""
        for tag in self.stash.iter_tags():
            self._remember(tag)
        self.catalog_loaded = True

    def _remember(self, tag: dict) -> dict:
        self.records[tag["name"]] = tag
        self.cache[tag["name"]] = str(tag["id"])
        self.names_by_id[str(tag['id'])] = tag['name']
        return tag

    def name_for_id(self, identifier: str) -> str | None:
        if identifier not in self.names_by_id:
            tag = self.stash.find_tag_by_id(identifier, required=False)
            if tag is None:
                self.names_by_id[identifier] = None
            else:
                self._remember(tag)
        return self.names_by_id[identifier]

    def _find(self, name: str) -> dict | None:
        if name not in self.records:
            self.records[name] = None if self.catalog_loaded else self.stash.find_tag(name)
            if self.records[name] is not None:
                self._remember(self.records[name])
        return self.records[name]

    def _create(self, name: str, parents: list[dict] | None = None, category: bool = False) -> dict | None:
        if not self.create_missing:
            self.cache[name] = None
            self.missing.add(name)
            return None
        payload: dict = {}
        if parents is not None:
            payload["parent_ids"] = sorted({str(tag["id"]) for tag in parents})
        if category:
            payload.update(description=f"{self.hierarchy.description_prefix} Organizational category from the tag hierarchy.",
                           ignore_auto_tag=True)
        if self.stash.dry_run:
            tag = {"id": f"dry-run:{name}", "name": name, "parents": parents or [],
                   "description": payload.get("description")}
            self.missing.add(name)
        else:
            tag = self.stash.create_tag(name, **payload)
        if self.hierarchy:
            self.hierarchy_stats["categories_created" if category else "tags_created"] += 1
        if self.stash.dry_run and self.report and parents is not None:
            self.report(f"[DRY] Create tag {name!r}; parents: " + (", ".join(p["name"] for p in parents) or "(none)"))
        return self._remember(tag)

    def _sync_parents(self, tag: dict, desired: list[dict], category: bool = False) -> dict:
        def final_parents(current: dict) -> dict[str, dict]:
            retained = {str(parent["id"]): parent for parent in current.get("parents", [])
                        if not self.hierarchy.is_managed_category(parent)}
            retained.update({str(parent["id"]): parent for parent in desired})
            return retained

        wanted = final_parents(tag)
        if set(wanted) == {str(parent["id"]) for parent in tag.get("parents", [])}:
            self.hierarchy_stats["parents_unchanged"] += 1
            return tag
        # Refresh before writing a changed relationship so manual parents added
        # after a catalog read are retained. Stash has no tag version/CAS input.
        if not self.stash.dry_run:
            tag = self.stash.find_tag_by_id(str(tag["id"]))
            if category and not self.hierarchy.is_managed_category(tag):
                raise RuntimeError(f"Category ownership changed for {tag['name']!r}; stopped before overwriting it.")
            wanted = final_parents(tag)
            if set(wanted) == {str(parent["id"]) for parent in tag.get("parents", [])}:
                self.hierarchy_stats["parents_unchanged"] += 1
                return self._remember(tag)
        if self.report and self.stash.dry_run:
            self.report(f"[DRY] Set parents of {tag['name']!r}: " + (", ".join(p["name"] for p in wanted.values()) or "(none)"))
        if self.stash.dry_run:
            updated = {**tag, "parents": list(wanted.values())}
        else:
            updated = self.stash.update_tag_parents(str(tag["id"]), sorted(wanted))
        self.hierarchy_stats["tags_reparented"] += 1
        return self._remember(updated)

    def _category(self, name: str) -> dict | None:
        if name in self.synced:
            return self.records[name]
        parent_names = self.hierarchy.category_parents[name]
        parents = [self._category(parent) for parent in parent_names]
        if any(parent is None for parent in parents):
            self.missing.add(name)
            return None
        tag = self._find(name)
        if tag is not None and not self.hierarchy.is_managed_category(tag):
            raise ValueError(f"Category name {name!r} already belongs to an unrelated tag. Choose another category_prefix in YAML.")
        if tag is None:
            tag = self._create(name, parents, category=True)
        else:
            tag = self._sync_parents(tag, parents, category=True)
        if tag is not None:
            self.synced.add(name)
        return tag

    def resolve(self, name: str) -> str | None:
        raw = name[len(self.source.prefix):] if name.startswith(self.source.prefix) else None
        if raw in self.exclusions:
            return None
        if name in self.synced:
            return self.cache.get(name)
        parent_names = self.hierarchy.leaf_parents.get(raw) if self.hierarchy else None
        parents = [self._category(parent) for parent in parent_names] if parent_names else None
        missing_parents = parents is not None and any(parent is None for parent in parents)
        tag = self._find(name)
        if tag is None:
            tag = self._create(name, parents if not missing_parents else None)
        elif parents is not None and not missing_parents:
            tag = self._sync_parents(tag, parents)
        if missing_parents:
            # Never prune existing relationships based on an incomplete set.
            self.hierarchy_stats["hierarchy_skipped"] += 1
        self.synced.add(name)
        return str(tag["id"]) if tag else None
