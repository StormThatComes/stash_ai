"""Prune unused leaf tag definitions through the Stash API, without model assets."""
from __future__ import annotations

from collections import Counter, deque
from contextlib import contextmanager
from itertools import batched
from threading import Event, Thread
from time import monotonic

from tqdm import tqdm

DEFAULT_PREFIXES = ("jt:", "wd:", "dtq:", "op:", "ai:", "m:")
REQUIRED_COUNTS = {"scene_count", "scene_marker_count", "image_count", "gallery_count", "performer_count", "child_count"}


def prefixes_checked(values):
    values = tuple(dict.fromkeys(values if values is not None else DEFAULT_PREFIXES))
    if not values or any(not isinstance(value, str) or value != value.strip() or
                         len(value) < 2 or not value.endswith(":") or any(c.isspace() for c in value) for value in values):
        raise ValueError("Prune prefixes must be explicit non-empty namespaces ending with ':', e.g. op: ai:.")
    return values


@contextmanager
def _waiting_for_stash(progress, batch, total_batches):
    """Refresh only the display; API calls and validation stay on the caller thread."""
    stopped = Event()
    started = monotonic()
    status = f"batch={batch}/{total_batches}"

    def refresh():
        progress.set_postfix_str(f"{status}, waiting for Stash: {int(monotonic() - started)}s")

    def tick():
        while not stopped.wait(1):
            refresh()

    refresh()
    ticker = Thread(target=tick, name="stash-prune-progress", daemon=True)
    ticker.start()
    try:
        yield
    finally:
        stopped.set()
        ticker.join()
        progress.set_postfix_str(status, refresh=False)


class PruneClient:
    def __init__(self, stash):
        self.stash = stash
        data = stash.gql('''query StashAIPruneCapabilities {
            tag: __type(name: "Tag") { fields(includeDeprecated: true) { name } }
            studio: __type(name: "Studio") { fields { name } }
            group: __type(name: "Group") { fields { name } }
            movie: __type(name: "Movie") { fields { name } }
            mutation: __type(name: "Mutation") { fields { name } }
        }''')
        fields = {f["name"] for f in (data.get("tag") or {}).get("fields", [])}
        required = REQUIRED_COUNTS | {"id", "name", "description", "parents", "children"}
        for entity in ("studio", "group", "movie"):
            entity_fields = {f["name"] for f in (data.get(entity) or {}).get("fields", [])}
            if "tags" in entity_fields and not (entity == "movie" and "group_count" in fields):
                required.add(entity + "_count")
        if not required <= fields or "tagDestroy" not in {f["name"] for f in (data.get("mutation") or {}).get("fields", [])}:
            raise RuntimeError("Stash does not expose all tag usage/child counters needed for safe pruning; nothing was deleted.")
        # Include new usage counters exposed by future compatible servers too.
        self.counts = sorted(name for name in fields if name.endswith("_count") and name != "parent_count")
        self.fields = "id name description parents { id } children { id } " + " ".join(self.counts)

    def catalog(self, ids=None):
        if ids is not None:
            result = {}
            ids = list(ids)
            if not ids:
                return result
            total_batches = (len(ids) + 99) // 100
            with tqdm(total=len(ids), desc="Check tag usage", unit="tag", dynamic_ncols=True) as progress:
                for number, group in enumerate(batched(ids, 100), 1):
                    with _waiting_for_stash(progress, number, total_batches):
                        response = self.stash.gql('''query StashAIPruneCandidates($ids: [ID!]!) {
                            findTags(ids: $ids) { tags { ''' + self.fields + ''' } }
                        }''', {'ids': list(group)})['findTags']['tags']
                    for tag in response:
                        self.validate(tag)
                        if str(tag['id']) not in group:
                            raise ValueError('Unexpected prune tag IDs; stopped before mutation.')
                        result[str(tag['id'])] = tag
                    # Missing IDs may have been deleted meanwhile; their lookup
                    # still completed. Failed/invalid batches never advance.
                    progress.update(len(group))
                    progress.refresh()
            return result
        page, result = 1, {}
        with tqdm(desc="Read tag catalog", unit="tag", dynamic_ncols=True) as progress:
            while True:
                # The first page supplies the total without a separate API request.
                count_field = "count" if page == 1 else ""
                response = self.stash.gql('''query StashAIPruneCatalog($filter: FindFilterType!) {
                    findTags(filter: $filter) { ''' + count_field + ''' tags { ''' + self.fields + ''' } }
                }''', {"filter": {"page": page, "per_page": 1000, "sort": "id", "direction": "ASC"}})["findTags"]
                batch = response["tags"]
                if page == 1 and type(response.get("count")) is int and response["count"] >= 0:
                    progress.total = response["count"]
                for tag in batch:
                    self.validate(tag)
                    result[str(tag["id"])] = tag
                if progress.total is not None:
                    progress.total = max(progress.total, progress.n + len(batch))
                progress.set_postfix(page=page, refresh=False)
                progress.update(len(batch))
                if len(batch) < 1000:
                    # The catalog may change while its pages are being read.
                    progress.total = progress.n
                    return result
                page += 1

    def validate(self, tag):
        if not isinstance(tag.get("name"), str) or not isinstance(tag.get("children"), list) or not isinstance(tag.get("parents"), list):
            raise ValueError("Incomplete tag relationships returned by Stash; pruning stopped.")
        if any(type(tag.get(name)) is not int or tag[name] < 0 for name in self.counts):
            raise ValueError("Unknown/invalid tag usage count; pruning stopped.")

    def unused_leaf(self, tag):
        self.validate(tag)
        return not tag["children"] and all(tag[name] == 0 for name in self.counts)

    def refresh(self, identifier):
        return self.stash.gql('''query StashAIPruneTag($id: ID!) {
            findTag(id: $id) { ''' + self.fields + ''' }
        }''', {"id": identifier})["findTag"]

    def delete(self, identifier):
        result = self.stash.gql('''mutation StashAIPruneTag($input: TagDestroyInput!) {
            tagDestroy(input: $input)
        }''', {"input": {"id": identifier}})["tagDestroy"]
        if result is not True:
            raise RuntimeError(f"Stash did not confirm deletion of tag {identifier}; rerun to reconcile.")


def prune_tags(stash, prefixes=None, report=print, *, names=None, candidate_ids=None, skip_marker_groups=False):
    prefixes = prefixes_checked(prefixes)
    # Standalone and post-merge callers use print; keep their logs above the bars.
    report = tqdm.write if report is print else report
    names = frozenset(names) if names is not None else None
    def selected(tag):
        from stash_ai.publication.groups import PREFIX, ROOT, managed_group
        if tag['name'] == ROOT or (tag['name'].startswith(PREFIX) and (skip_marker_groups or not managed_group(tag))):
            return False
        return tag["name"] in names if names is not None else tag["name"].startswith(prefixes)
    report("Pruning unused excluded tag definitions." if names is not None else
           "Empty-tag pruning, global prefixes: " + " ".join(prefixes))
    client = PruneClient(stash)
    catalog = client.catalog(candidate_ids)  # Finish pagination before deleting anything.
    pending = deque(identifier for identifier, tag in catalog.items()
                    if selected(tag) and client.unused_leaf(tag))
    stats = Counter(tags_pruned=0, tags_retained_after_refresh=0)
    if names is not None:
        stats['excluded_tags_still_used_or_parent'] = sum(selected(tag) and not client.unused_leaf(tag)
                                                        for tag in catalog.values())
    description = "Preview empty tags" if stash.dry_run else "Prune empty tags"
    with tqdm(total=len(pending), desc=description, unit="tag", dynamic_ncols=True) as progress:
        while pending:
            identifier = pending.popleft()
            try:
                previous = catalog.get(identifier)
                if previous is None:
                    continue
                tag = previous if stash.dry_run else client.refresh(identifier)
                if tag is None:
                    catalog.pop(identifier, None)
                    continue
                if not selected(tag) or not client.unused_leaf(tag):
                    stats["tags_retained_after_refresh"] += 1
                    continue
                report(f"{'[DRY] Would delete' if stash.dry_run else 'Deleting'} unused leaf tag {tag['name']!r} (id={identifier}).")
                if not stash.dry_run:
                    client.delete(identifier)
                stats["tags_pruned"] += 1
                catalog.pop(identifier, None)
                for parent_id in {str(parent["id"]) for parent in [*previous["parents"], *tag["parents"]]}:
                    parent = catalog.get(parent_id)
                    if not parent:
                        continue
                    parent["children"] = [child for child in parent["children"] if str(child["id"]) != identifier]
                    parent["child_count"] = len(parent["children"])
                    if selected(parent) and client.unused_leaf(parent):
                        pending.append(parent_id)
                        progress.total += 1
            finally:
                progress.set_postfix({"planned" if stash.dry_run else "deleted": stats["tags_pruned"],
                                      "kept": stats["tags_retained_after_refresh"], "pending": len(pending)}, refresh=False)
                progress.update(1)
    if not progress.n:
        report("No unused leaf tags found in the selected prefixes.")
    return stats


def run_prune(app, args):
    # The same writer lock prevents pruning against a concurrent local merge.
    from stash_ai.storage.store import StateStore, scope_for
    with StateStore(args.state_db, scope_for(app.STASH_URL), app.DRY_RUN):
        stash = app.create_stash_client(app.DRY_RUN)
        try:
            stats = prune_tags(stash, args.prune_prefixes, report=args.report.detail)
        finally:
            stash.close()
    print("Pruning" + (" (planned)" if app.DRY_RUN else "") + ": " + ", ".join(f"{key}={value}" for key,value in sorted(stats.items())))
    return 0
