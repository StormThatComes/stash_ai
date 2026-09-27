"""Reconciliation of one scene into the merged AI namespace."""
from __future__ import annotations

from collections import Counter, defaultdict

from stash_ai import merge
from stash_ai.occurrences import OCCURRENCE_TAG, is_occurrence
from stash_ai.runtime import stage
from stash_ai.taggers import SOURCES


class MergedSceneSynchronizer:
    """Plan and publish one scene while preserving manual and recovery state."""

    def __init__(self, app, stash, resolver, store, *, migrate=True, cleanup=False,
                 report=None, tag_policies=None, publication_key=None):
        self.app = app
        self.stash = stash
        self.resolver = resolver
        self.store = store
        self.migrate = migrate
        self.cleanup = cleanup
        self.report = report or getattr(store, 'report', None)
        self.tag_policies = tag_policies
        self.publication_key = publication_key

        from stash_ai.exclusions import excluded_names
        from stash_ai.marker_groups import for_resolver
        self.blocked_names = excluded_names(tag_policies)
        self.groups = for_resolver(app, resolver)

        self.scene = None
        self.scene_id = None
        self.scene_hits = None
        self.spans = None
        self.markers = None
        self.scene_labels = None
        self.ids = None
        self.previous_ranges = []
        self.suppressed = set()
        self.snapshots = {}
        self.current_tags = []
        self.existing_ids = set()
        self.desired_ids = set()
        self.final_ids = set()
        self.migrated_scene_ids = set()
        self.wanted = []
        self.creates = []
        self.updates = []
        self.deletes = set()
        self.unchanged = 0

    def sync(self, scene, scene_hits, spans):
        self.scene = scene
        self.scene_id = str(scene["id"])
        self.scene_hits = scene_hits
        self.spans = spans

        self._load_previous_ranges()
        self._prepare_evidence()
        self._resolve_tags_and_refresh()
        self._plan_scene_tags()
        self._plan_markers()
        if self.app.CREATE_MARKERS:
            self.snapshots["markers"] = {"contributions": self.spans}
        stats = self._statistics()
        return self._publish(stats)

    def _load_previous_ranges(self):
        app = self.app
        if not app.CREATE_MARKERS or (self.stash.has_end_time and app.MARKER_USE_END_TIME):
            return
        from stash_ai.recovery import publications
        snapshots = (
            *self.store.snapshot(self.scene_id, "markers", borrow=True),
            *publications(self.store, self.scene_id, 'markers'),
        )
        for snapshot in snapshots:
            if snapshot:
                self.previous_ranges.extend(
                    merge.union_intervals(snapshot.get("contributions", []), borrow=True))

    def temporal_marker(self, marker):
        """Restore logical end time for point-only Stash markers when known."""
        ends = [
            item["end"] for item in self.previous_ranges
            if item["end"] is not None
            and merge.AI.prefix + item["tag"] == marker["primary_tag"]["name"]
            and abs(round(item["start"], 3) - marker["seconds"]) <= 0.0005
        ]
        return {**marker, "end_seconds": max(ends)} if ends else marker

    def _prepare_evidence(self):
        app = self.app
        with stage(self.report, 'marker_union'):
            self.markers = (
                merge.union_intervals(self.spans, borrow=True)
                if app.CREATE_MARKERS else [])
        if self.report is not None:
            self.report.record_counts(
                marker_contributions=len(self.spans),
                union_markers=len(self.markers),
                scene_contributions=len(self.scene_hits))
        self.scene_labels = (
            {merge.raw_label(hit["tag"]) for hit in self.scene_hits}
            if app.APPLY_SCENE_TAGS else set())
        if not self.cleanup:
            return

        # Cleanup never restores outputs that the user removed since the last sync.
        self.scene_labels.intersection_update(
            tag["name"][len(merge.AI.prefix):]
            for tag in self.scene.get("tags", [])
            if tag["name"].startswith(merge.AI.prefix))
        self.scene_hits = [
            hit for hit in self.scene_hits if hit["tag"] in self.scene_labels]
        present = [
            marker for marker in self.scene.get("scene_markers", [])
            if app.owned_marker(marker, merge.AI)]
        self.markers = [
            marker for marker in self.markers
            if any(
                old["primary_tag"]["name"] == merge.AI.prefix + marker["tag"]
                and merge.overlaps(
                    self.temporal_marker(old),
                    {"seconds": marker["start"], "end_seconds": marker["end"]}, 0)
                for old in present)]
        self.spans = [
            item for marker in self.markers for item in marker["contributors"]]

    def _resolve_tags_and_refresh(self):
        app = self.app
        names = {merge.AI.prefix + label for label in self.scene_labels}
        for marker in self.markers:
            names.update(
                merge.AI.prefix + label
                for label in [marker["tag"], *marker["extra_tags"]])
        if self.markers:
            names.add(merge.AI.owner_tag)
        if any(is_occurrence(marker) for marker in self.markers):
            names.add(OCCURRENCE_TAG)

        with stage(self.report, 'tag_resolution'):
            self.ids = {
                name: self.resolver.resolve(name) for name in sorted(names)}
        if self.markers and not self.ids.get(merge.AI.owner_tag):
            raise ValueError("Missing AI marker ownership tag; enable CREATE_MISSING.")

        # Refresh after tag creation/reparenting, preserving newly added manual tags.
        with stage(self.report, 'scene_refresh'):
            self.scene = (
                self.stash.find_scene_for_merge(self.scene_id)
                if hasattr(self.stash, 'find_scene_for_merge')
                else self.stash.find_scene(self.scene_id))
        if self.report is not None:
            self.report.record_counts(
                existing_markers=len(self.scene.get('scene_markers', [])),
                existing_scene_tags=len(self.scene.get('tags', [])))

        # A manually maintained ai: marker already supplies a location. It remains
        # untouched, and must not receive a duplicate automatically generated point.
        manual_labels = {
            marker['primary_tag']['name']
            for marker in self.scene.get('scene_markers', [])
            if not app.owned_marker(marker, merge.AI)}
        self.suppressed = {
            marker['tag'] for marker in self.markers
            if is_occurrence(marker)
            and merge.AI.prefix + marker['tag'] in manual_labels}
        if self.suppressed:
            self.markers = [
                marker for marker in self.markers
                if not (is_occurrence(marker) and marker['tag'] in self.suppressed)]
            self.spans = [
                span for span in self.spans
                if not (
                    span.get('derived') == 'occurrence'
                    and span['tag'] in self.suppressed)]

        # Imported/manual extras may retain IDs in old namespaces. Resolve each
        # unknown ID at most once through the shared resolver, not per marker.
        for tag in self.scene.get('tags', []):
            self.resolver.names_by_id[str(tag['id'])] = tag['name']
        for marker in self.scene.get('scene_markers', []):
            for tag in [marker['primary_tag'], *marker.get('tags', [])]:
                self.resolver.names_by_id[str(tag['id'])] = tag['name']

    def _allowed_id(self, identifier):
        name = self.resolver.name_for_id(str(identifier))
        return (
            name is not None
            and name not in self.blocked_names
            and not self.groups.owns_id(identifier))

    def _manual_extra(self, tag):
        return (
            not tag['name'].startswith(merge.AI.prefix)
            and tag['name'] not in self.blocked_names
            and not self.groups.owns(tag))

    def _plan_scene_tags(self):
        from stash_ai.recovery import publications

        self.current_tags = self.scene.get("tags", [])
        self.migrated_scene_ids = (
            {
                str(tag["id"]) for tag in self.current_tags
                if merge.legacy_label(tag["name"])
            }
            if self.migrate else set())
        published, pending = self.store.snapshot(
            self.scene_id, "scene", borrow=True)
        owned_before = (
            set((published or {}).get("owned_ids", []))
            | set((pending or {}).get("owned_ids", [])))
        owned_before.update(
            identifier
            for snapshot in publications(self.store, self.scene_id, 'scene')
            for identifier in snapshot.get('owned_ids', []))

        self.existing_ids = {
            str(tag["id"]) for tag in self.current_tags}
        self.desired_ids = {
            self.ids[merge.AI.prefix + label]
            for label in self.scene_labels
            if self.ids.get(merge.AI.prefix + label)}
        unresolved_labels = {
            label for label in self.scene_labels
            if not self.ids.get(merge.AI.prefix + label)}

        # CREATE_MISSING=False must not remove a legacy association lacking a replacement.
        self.migrated_scene_ids.difference_update(
            str(tag["id"]) for tag in self.current_tags
            if (
                (parsed := merge.legacy_label(tag["name"]))
                and parsed[1] in unresolved_labels))
        forbidden_ids = {
            str(tag['id']) for tag in self.current_tags
            if tag['name'] in self.blocked_names}
        preserved = (
            self.existing_ids
            - owned_before
            - self.migrated_scene_ids
            - forbidden_ids)
        self.final_ids = preserved | self.desired_ids
        if self.app.APPLY_SCENE_TAGS:
            self.snapshots["scene"] = {
                "owned_ids": sorted(self.desired_ids - preserved),
                "contributions": self.scene_hits,
            }

    def _plan_markers(self):
        app = self.app
        with stage(self.report, 'marker_matching'):
            resolved_marker_labels = set()
            for marker in self.markers:
                primary = self.ids.get(merge.AI.prefix + marker["tag"])
                if not primary:
                    continue
                resolved_marker_labels.add(marker["tag"])
                extras = {
                    self.ids[merge.AI.prefix + label]
                    for label in marker["extra_tags"]
                    if self.ids.get(merge.AI.prefix + label)}
                extras.update(
                    identifier for identifier in marker["extra_ids"]
                    if self._allowed_id(identifier))
                extras.add(self.ids[merge.AI.owner_tag])
                extras.add(
                    self.groups.resolve(marker['tag'], is_occurrence(marker)))
                if is_occurrence(marker) and self.ids.get(OCCURRENCE_TAG):
                    extras.add(self.ids[OCCURRENCE_TAG])
                extras.discard(primary)
                payload = {
                    "scene_id": self.scene_id,
                    "primary_tag_id": primary,
                    "tag_ids": sorted(extras),
                    "title": (
                        merge.AI.title_prefix + marker["tag"] + " ["
                        + "+".join(merge.source_codes(marker["sources"])) + "]"),
                    "seconds": round(marker["start"], 3),
                }
                if is_occurrence(marker):
                    payload['title'] += ' · wykrycie'
                if self.stash.has_end_time:
                    payload["end_seconds"] = (
                        round(marker["end"], 3)
                        if marker["end"] is not None and app.MARKER_USE_END_TIME
                        else None)
                self.wanted.append((
                    payload,
                    {"seconds": marker["start"], "end_seconds": marker["end"]}))

            current_markers = self.scene.get("scene_markers", [])
            owned = [
                marker for marker in current_markers
                if app.owned_marker(marker, merge.AI)]
            owned_by_primary = defaultdict(list)
            for marker in owned:
                owned_by_primary[str(marker["primary_tag"]["id"])].append(marker)
            available = {
                str(marker["id"]): marker for marker in owned}

            for payload, interval in self.wanted:
                same_primary = owned_by_primary[payload["primary_tag_id"]]
                related = [
                    marker for marker in same_primary
                    if merge.overlaps(
                        self.temporal_marker(marker), interval, 0)]
                # Keep manual extras from every marker absorbed by the union,
                # including duplicates that will be deleted. A split carries
                # them to both pieces.
                payload["tag_ids"] = sorted(
                    set(payload["tag_ids"])
                    | {
                        str(tag["id"])
                        for marker in related
                        for tag in marker.get("tags", [])
                        if self._manual_extra(tag)})
                matches = [
                    marker for marker in same_primary
                    if str(marker["id"]) in available
                    and merge.overlaps(
                        self.temporal_marker(marker), interval,
                        app.MARKER_MATCH_TOLERANCE_SECONDS)]
                if not matches:
                    self.creates.append(payload)
                    continue

                def match_key(marker):
                    marker = self.temporal_marker(marker)
                    old_end = (
                        marker.get("end_seconds")
                        if marker.get("end_seconds") is not None
                        else marker["seconds"])
                    new_end = (
                        interval["end_seconds"]
                        if interval["end_seconds"] is not None
                        else interval["seconds"])
                    shared = max(
                        0,
                        min(old_end, new_end)
                        - max(marker["seconds"], payload["seconds"]))
                    return (
                        -shared,
                        abs(marker["seconds"] - payload["seconds"]),
                        str(marker["id"]))

                current = min(matches, key=match_key)
                marker_id = str(current["id"])
                del available[marker_id]
                # Additional manually attached tags outside the generated
                # namespace survive.
                payload["tag_ids"] = sorted(
                    set(payload["tag_ids"])
                    | {
                        str(tag["id"])
                        for tag in current.get("tags", [])
                        if self._manual_extra(tag)})
                if app.marker_differs(current, payload):
                    self.updates.append({**payload, "id": marker_id})
                else:
                    self.unchanged += 1

            self.deletes = (
                set(available) if app.CREATE_MARKERS else set())
            if self.migrate and app.CREATE_MARKERS:
                predicted_labels = {
                    marker["tag"] for marker in self.markers}
                for marker in current_markers:
                    parsed = merge.legacy_label(
                        (marker.get("primary_tag") or {}).get("name", ""))
                    if (
                        parsed
                        and app.owned_marker(marker, SOURCES[parsed[0]])
                        and (
                            parsed[1] not in predicted_labels
                            or parsed[1] in resolved_marker_labels)):
                        self.deletes.add(str(marker["id"]))

            if app.CREATE_MARKERS and self.blocked_names:
                self.deletes.update(
                    str(marker['id']) for marker in current_markers
                    if marker['primary_tag']['name'] in self.blocked_names)
                changed_ids = (
                    self.deletes
                    | {payload['id'] for payload in self.updates})
                for marker in current_markers:
                    if str(marker['id']) in changed_ids:
                        continue
                    payload = app.marker_extra_cleanup_payload(
                        marker, self.scene_id, self.blocked_names,
                        self.stash.has_end_time)
                    if payload is not None:
                        self.updates.append(payload)

    def _statistics(self):
        app = self.app
        scene_changed = (
            app.APPLY_SCENE_TAGS
            and self.final_ids != self.existing_ids)
        self.scene_changed = scene_changed
        stats = Counter(
            scene_tags_updated=int(scene_changed),
            legacy_scene_tags_migrated=(
                len(self.migrated_scene_ids)
                if app.APPLY_SCENE_TAGS else 0),
            markers_created=len(self.creates),
            markers_updated=len(self.updates),
            markers_deleted=len(self.deletes),
            markers_unchanged=self.unchanged,
            merged_markers=len(self.wanted),
            merged_scene_tags=len(self.desired_ids),
        )
        stats['short_extras'] = sum(
            item.get('qualification') == 'short_episode'
            for marker in self.markers
            for item in marker.get('extra_evidence', {}).values())
        stats['occurrence_markers'] = sum(
            is_occurrence(marker) for marker in self.markers)
        stats['occurrence_manual_locations'] = len(self.suppressed)
        stats['cross_model_extras'] = sum(
            bool(set(item.get('source_evidence', {})) - set(marker['sources']))
            for marker in self.markers
            for tag, item in marker.get('extra_evidence', {}).items()
            if tag in marker['extra_tags'])
        return stats

    def _publish(self, stats):
        app = self.app
        if self.stash.dry_run:
            self.groups.record_preview(
                self.scene, self.creates, self.updates, self.deletes)
            write = self.report.detail if self.report is not None else print
            write(
                f"[DRY] scene {self.scene_id}: ai scene tags={len(self.desired_ids)}, "
                f"markers create={len(self.creates)} update={len(self.updates)} "
                f"delete={len(self.deletes)}; "
                f"migrate scene tags={stats['legacy_scene_tags_migrated']}.")
            if app.APPLY_SCENE_TAGS and (
                    self.report is None or self.report.verbose):
                write(
                    "  Scene tags: "
                    + (
                        ", ".join(
                            merge.AI.prefix + label
                            for label in sorted(self.scene_labels))
                        or "(none)"))
            for marker in (
                    self.markers
                    if self.report is None or self.report.verbose else ()):
                end = (
                    app.format_time(marker["end"])
                    if marker["end"] is not None
                    else "point (end unknown)")
                write(
                    f"  {app.format_time(marker['start'])} -> {end}: "
                    f"ai:{marker['tag']} "
                    f"[{'+'.join(merge.source_codes(marker['sources']))}]")
            self.store.publish(
                self.scene_id, self.snapshots)  # temporary preview overlay only
            return stats

        with stage(self.report, 'publication_prepare'):
            self.store.prepare(self.scene_id, self.snapshots)
        created = []
        with stage(self.report, 'publication_mutations'):
            for payload in self.creates:
                created.append(
                    (payload, self.stash.create_marker(payload)))
            for payload in self.updates:
                self.stash.update_marker(payload)
            if self.scene_changed:
                self.stash.update_scene_tags(
                    self.scene_id, sorted(self.final_ids))
            for marker_id in sorted(self.deletes):
                self.stash.delete_marker(marker_id)

        publication = None
        with stage(self.report, 'publication_fingerprint'):
            if (
                    self.publication_key is not None
                    and not self.resolver.missing):
                from stash_ai.performance import projected_scene, scene_fingerprint
                observed = projected_scene(
                    self.scene, self.resolver, created,
                    self.updates, self.deletes,
                    self.final_ids
                    if app.APPLY_SCENE_TAGS else self.existing_ids)
                if observed is None:
                    observed = self.stash.find_scene(self.scene_id)
                publication = (
                    self.publication_key,
                    scene_fingerprint(observed))
        with stage(self.report, 'publication_commit'):
            self.store.commit_prepared(
                self.scene_id, self.snapshots, publication=publication)
        return stats
