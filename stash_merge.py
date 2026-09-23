"""Exact-label fusion, interval union and durable per-source evidence.

No inference adapter depends on this module. The caller supplies the application
module so CLI and imported use share the same configuration and Stash client.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from contextlib import AbstractContextManager, nullcontext
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import sqlite3
from urllib.parse import urlparse

from stash_domain import Cue, FramePrediction, TagHierarchy
from stash_taggers import ModelSource, SOURCES
from stash_runtime import measured, stage
from stash_state_codec import encode, decode
from stash_occurrences import OCCURRENCE_TAG

AI = ModelSource("ai", "AI", "ai:", "ai:marker:stash-taggers", "[AI] ", "[AI hierarchy v1]")
APPLICATION_ID = 0x53544D47
KINDS = ("scene", "markers")


def now():
    return datetime.now(timezone.utc).isoformat()


def scope_for(url):
    parsed = urlparse(url)
    port = parsed.port or (443 if parsed.scheme == "https" else 80)
    canonical = f"{parsed.scheme.lower()}://{parsed.hostname.lower()}:{port}{parsed.path.rstrip('/')}"
    return hashlib.sha256(canonical.encode()).hexdigest()


class StateStore(AbstractContextManager):
    """Results commit after each successful inference; API writes have an intent log.

    A dry run uses a read-only source plus a sparse temporary overlay, no source lock.
    A writer holds a process lock for its whole run, including Stash requests.
    """
    def __init__(self, path, scope, dry_run=False, cache_bytes=0, *, allow_cleanup=False, cleanup_preview=False,
                 memory_budget_bytes=0):
        from stash_runtime import RamCache
        self.cache = RamCache(cache_bytes, rss_budget=memory_budget_bytes)
        self.path, self.scope = Path(path).expanduser().resolve(), scope
        self.lock_file, self.connection = None, None
        self.preview_source = False
        try:
            if dry_run:
                # Empty filename = SQLite-managed temporary DB, deleted at close.
                # Its bounded page cache prevents a long preview consuming RAM
                # proportional to the entire inference database.
                self.connection = sqlite3.connect("", uri=True)
                if self.path.exists():
                    source = sqlite3.connect(self.path.as_uri() + "?mode=ro", uri=True)
                    try:
                        source_version = source.execute('PRAGMA user_version').fetchone()[0]
                        if (source_version in (5, 6, 7) and
                                source.execute('PRAGMA application_id').fetchone()[0] == APPLICATION_ID):
                            # Modern schemas can be read directly; only preview
                            # changes occupy the temporary overlay.
                            self.connection.execute('ATTACH DATABASE ? AS cleanup_source',
                                                    (self.path.as_uri() + '?mode=ro',))
                            self.connection.execute(f'PRAGMA user_version={source_version}')
                            self.preview_source = True
                            self.preview_data_version = self.connection.execute(
                                'PRAGMA cleanup_source.data_version').fetchone()[0]
                        else:
                            source.backup(self.connection)
                    finally:
                        source.close()
            else:
                self.path.parent.mkdir(parents=True, exist_ok=True)
                self.lock_file = open(str(self.path) + ".lock", "a+b")
                if self.lock_file.seek(0, os.SEEK_END) == 0:
                    self.lock_file.write(b"0")
                    self.lock_file.flush()
                self.lock_file.seek(0)
                try:
                    if os.name == "nt":
                        import msvcrt
                        msvcrt.locking(self.lock_file.fileno(), msvcrt.LK_NBLCK, 1)
                    else:
                        import fcntl
                        fcntl.flock(self.lock_file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                except OSError as exc:
                    raise RuntimeError("Another merged-output process is using this state DB.") from exc
                self.connection = sqlite3.connect(self.path, timeout=30)
            db = self.connection
            application = db.execute("PRAGMA application_id").fetchone()[0]
            version = db.execute("PRAGMA user_version").fetchone()[0]
            tables = db.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
            if application not in (0, APPLICATION_ID) or version not in (0, 1, 2, 3, 4, 5, 6, 7) or (tables and application != APPLICATION_ID):
                raise ValueError("Not a supported stash-taggers state DB; choose another --state-db.")
            db.executescript(f"""
                PRAGMA application_id={APPLICATION_ID};
                CREATE TABLE IF NOT EXISTS results (
                    scope TEXT NOT NULL, scene TEXT NOT NULL, model TEXT NOT NULL,
                    kind TEXT NOT NULL, data BLOB NOT NULL,
                    PRIMARY KEY(scope, scene, model, kind));
                CREATE TABLE IF NOT EXISTS snapshots (
                    scope TEXT NOT NULL, scene TEXT NOT NULL, kind TEXT NOT NULL,
                    published BLOB, pending BLOB,
                    PRIMARY KEY(scope, scene, kind));
                CREATE TABLE IF NOT EXISTS legacy_imports (
                    scope TEXT NOT NULL, scene TEXT NOT NULL, kind TEXT NOT NULL,
                    data BLOB NOT NULL, PRIMARY KEY(scope, scene, kind));
            """)
            from stash_progress import initialize_progress
            initialize_progress(self, version)
            from stash_cleanup import initialize
            initialize(self, version)
            if self.preview_source:
                self.connection.execute('INSERT INTO cleanup_state SELECT * FROM cleanup_source.cleanup_state WHERE scope=?', (self.scope,))
                self.connection.commit()
                initialize(self, version)
            from stash_performance import initialize as initialize_performance
            initialize_performance(self)
            from stash_recovery import initialize as initialize_recovery
            initialize_recovery(self)
            from stash_preview import initialize as initialize_preview
            initialize_preview(self)
            if self.cleanup_pending and not allow_cleanup:
                raise RuntimeError('Global cleanup is interrupted; rerun the same --models all cleanup mode '
                                   'with the same exclusion policies, or use --abandon-pending with current policies.')
        except BaseException:
            self.close()
            raise

    def close(self):
        self.cache.clear()
        if self.connection is not None:
            self.connection.close()
            self.connection = None
        if self.lock_file is not None:
            # Closing the descriptor releases the advisory lock on both platforms.
            self.lock_file.close()
            self.lock_file = None

    def __exit__(self, *args):
        self.close()

    def cached(self, key):
        # A transaction may still roll back: never cache its provisional view.
        from stash_preview import check_source
        check_source(self)
        return None if self.connection.in_transaction else self.cache.get(key)

    def table(self, name):
        from stash_preview import table
        return table(self, name)

    @measured('state_cache')
    def retain(self, key, value):
        if not self.connection.in_transaction:
            self.cache.put(key, value)

    @measured('state_sql_read')
    def read_rows(self, sql, params):
        return self.connection.execute(sql, params).fetchall()

    def pack(self, value):
        return encode(value, getattr(self, 'report', None))

    def unpack(self, value):
        return decode(value, getattr(self, 'report', None))

    @measured('state_read')
    def records(self, scene, kind, model=None, *, borrow=False):
        """Public reads are detached; internal read-only consumers may borrow."""
        key = ('records', str(scene), kind)
        records = self.cached(key)
        if records is None:
            records = {model: self.unpack(data) for model, data in self.read_rows(
                "SELECT model,data FROM " + self.table('results') + " WHERE scope=? AND scene=? AND kind=?",
                (self.scope, str(scene), kind))}
            # Retired video samples are never inputs. Remove them from the decoded
            # view without scanning/recompressing every scene during DB migration.
            for segments in records.values():
                from stash_performance import StoredSegment
                for index, segment in enumerate(segments):
                    segment.pop("refinement", None)
                    segments[index] = StoredSegment(segment)
            self.retain(key, records)
        result = {model: records[model]} if model in records else {} if model else records
        return result if borrow else deepcopy(result)

    def sources(self, scene, kind):
        key = ('sources', str(scene), kind)
        value = self.cached(key)
        if value is None:
            value = {row[0] for row in self.connection.execute(
                "SELECT model FROM " + self.table('results') + " WHERE scope=? AND scene=? AND kind=?", (self.scope, str(scene), kind))}
            self.retain(key, value)
        return set(value)

    def preload_sources(self):
        if not self.cache.limit:
            return
        last, values = None, set()
        for scene, kind, model in self.connection.execute(
                'SELECT scene,kind,model FROM ' + self.table('results') + ' WHERE scope=? ORDER BY scene,kind', (self.scope,)):
            key = ('sources', scene, kind)
            if key != last:
                if last is not None:
                    self.cache.put(last, values)
                last, values = key, set()
            values.add(model)
        if last is not None:
            self.cache.put(last, values)

    @measured('state_write')
    def put(self, scene, model, outputs, replace=None):
        with self.connection:
            self._put(scene, model, outputs, replace)

    def _put(self, scene, model, outputs, replace=None):
        if model not in SOURCES or set(outputs) - set(KINDS):
            raise ValueError("Unknown result source or output kind.")
        for kind, segment in outputs.items():
            segment = {key: value for key, value in segment.items() if key != "refinement"}
            old = self.records(scene, kind, model, borrow=True).get(model, []) if not (replace or {}).get(kind, True) else []
            # Repeating an identical run with replacement disabled need not
            # accumulate identical frame streams merely due to its timestamp.
            keep = []
            if old:
                # Only append mode needs comparison. A shallow metadata copy
                # avoids duplicating all retained frame observations.
                def semantic(value):
                    data = dict(value)
                    if 'meta' in data:
                        data['meta'] = {key: item for key, item in data['meta'].items() if key != 'captured_at'}
                    # JSON normalizes tuples in new results to stored lists.
                    return json.dumps(data, sort_keys=True, allow_nan=False)
                current = semantic(segment)
                keep = [previous for previous in old if semantic(previous) != current]
            self.replace_records(scene, model, kind, keep + [segment])

    def replace_records(self, scene, model, kind, segments):
        """Replace a segment list inside the caller's transaction, including previews."""
        from stash_progress import index_result
        self.cache.discard(('records', str(scene), kind))
        self.cache.discard(('sources', str(scene), kind))
        self.cache.discard(('status', str(scene), model))
        self.connection.execute("INSERT INTO results VALUES (?,?,?,?,?) "
            "ON CONFLICT(scope,scene,model,kind) DO UPDATE SET data=excluded.data",
            (self.scope, str(scene), model, kind, self.pack(segments)))
        index_result(self, scene, model, kind, segments)

    @measured('state_read')
    def imported_legacy(self, scene, kind, *, borrow=False):
        key = ('legacy', str(scene), kind)
        cached = self.cached(key)
        if cached is not None:
            return cached if borrow else deepcopy(cached)
        rows = self.read_rows("SELECT data FROM " + self.table('legacy_imports') + " WHERE scope=? AND scene=? AND kind=?",
                              (self.scope, str(scene), kind))
        values = self.unpack(rows[0][0]) if rows else []
        self.retain(key, values)
        return values if borrow else deepcopy(values)

    @measured('state_write')
    def remember_legacy(self, scene, kind, data):
        from stash_performance import freeze
        previous = self.imported_legacy(scene, kind, borrow=True)
        known = {freeze(item) for item in previous}
        fresh = [item for item in data if freeze(item) not in known]
        if not fresh:
            return
        previous = previous + fresh
        self.cache.discard(('legacy', str(scene), kind))
        with self.connection:
            self.connection.execute("INSERT INTO legacy_imports VALUES (?,?,?,?) "
                "ON CONFLICT(scope,scene,kind) DO UPDATE SET data=excluded.data",
                (self.scope, str(scene), kind, self.pack(previous)))

    @measured('state_read')
    def snapshot(self, scene, kind, *, borrow=False):
        key = ('snapshot', str(scene), kind)
        cached = self.cached(key)
        if cached is not None:
            return cached if borrow else deepcopy(cached)
        rows = self.read_rows("SELECT published,pending FROM " + self.table('snapshots') + " WHERE scope=? AND scene=? AND kind=?",
                              (self.scope, str(scene), kind))
        from stash_cleanup import mask_snapshot
        values = tuple(mask_snapshot(self, self.unpack(item), kind) if item is not None else None
                       for item in rows[0]) if rows else (None, None)
        self.retain(key, values)
        return values if borrow else deepcopy(values)

    def has_pending_publications(self):
        return self.connection.execute('SELECT 1 FROM ' + self.table('snapshots') +
            ' WHERE scope=? AND pending IS NOT NULL LIMIT 1', (self.scope,)).fetchone() is not None

    @measured('state_write')
    def prepare(self, scene, snapshots):
        from stash_cleanup import stamp_snapshot
        with self.connection:
            for kind, data in snapshots.items():
                self.cache.discard(('snapshot', str(scene), kind))
                if self.preview_source:
                    self.connection.execute('INSERT OR IGNORE INTO snapshots SELECT * FROM '
                        'cleanup_source.snapshots WHERE scope=? AND scene=? AND kind=?',
                        (self.scope, str(scene), kind))
                self.connection.execute("INSERT INTO snapshots VALUES (?,?,?,NULL,?) "
                    "ON CONFLICT(scope,scene,kind) DO UPDATE SET pending=excluded.pending",
                    (self.scope, str(scene), kind, self.pack(stamp_snapshot(self, data))))

    @measured('state_write')
    def publish(self, scene, snapshots):
        from stash_cleanup import stamp_snapshot
        from stash_recovery import clear_publications
        with self.connection:
            for kind, data in snapshots.items():
                self.cache.discard(('snapshot', str(scene), kind))
                self.connection.execute("INSERT INTO snapshots VALUES (?,?,?,?,NULL) "
                    "ON CONFLICT(scope,scene,kind) DO UPDATE SET published=excluded.published,pending=NULL",
                    (self.scope, str(scene), kind, self.pack(stamp_snapshot(self, data))))
                clear_publications(self, scene, kind)


    @measured('state_write')
    def commit_prepared(self, scene, kinds, *, publication=None):
        """Promote durable intent without encoding large evidence a second time."""
        from stash_recovery import clear_publications
        with self.connection:
            for kind in kinds:
                self.cache.discard(('snapshot', str(scene), kind))
                changed = self.connection.execute('UPDATE snapshots SET published=pending,pending=NULL '
                    'WHERE scope=? AND scene=? AND kind=? AND pending IS NOT NULL',
                    (self.scope, str(scene), kind)).rowcount
                if changed != 1:
                    raise RuntimeError('Publication intent disappeared before commit.')
                clear_publications(self, scene, kind)
            if publication is not None:
                from stash_performance import record_check
                record_check(self, scene, *publication)


def raw_label(value):
    if (not isinstance(value, str) or not value or value != value.strip() or "\n" in value or "\r" in value
            or AI.prefix + value in (AI.owner_tag, OCCURRENCE_TAG) or value.startswith("category:")):
        raise ValueError(f"Invalid or reserved merged label: {value!r}")
    return value


def legacy_label(name):
    for key, source in SOURCES.items():
        if name.startswith(source.prefix) and name != source.owner_tag:
            label = name[len(source.prefix):]
            if label and not label.startswith(("category:", "marker:")):
                return key, raw_label(label)
    return None


def source_codes(sources):
    return [source.prefix.rstrip(":") for key, source in SOURCES.items() if key in sources]


def union_intervals(contributions, *, borrow=False):
    """One component per exact label. Positive gaps are never bridged here.

    Unknown legacy end times remain points, unless covered by a known interval.
    Scores are kept on contributors; there is no artificial ensemble probability.
    """
    grouped = defaultdict(list)
    memo = {}
    for contribution in contributions:
        item = contribution if borrow else deepcopy(contribution, memo)
        tag = raw_label(item["tag"])
        if item["source"] not in SOURCES:
            raise ValueError("Unknown interval source.")
        start, end = item["start"], item["end"]
        if (isinstance(start, bool) or not isinstance(start, (int, float)) or not math.isfinite(start) or start < 0
                or (end is not None and (isinstance(end, bool) or not isinstance(end, (int, float))
                    or not math.isfinite(end) or end < start))):
            raise ValueError(f"Invalid marker interval for {tag!r}.")
        grouped[tag].append(item)
    merged = []
    for tag, items in sorted(grouped.items()):
        items.sort(key=lambda item: (item["start"], item["end"] if item["end"] is not None else item["start"], item["source"]))
        components = []
        for item in items:
            end = item["end"] if item["end"] is not None else item["start"]
            if not components or item["start"] > components[-1]["bound"]:
                components.append(dict(tag=tag, start=item["start"], bound=end, contributors=[]))
            component = components[-1]
            component["bound"] = max(component["bound"], end)
            if item not in component["contributors"]:
                component["contributors"].append(item)
        for component in components:
            contributors = component["contributors"]
            bound = component.pop("bound")
            component["end"] = bound if any(item["end"] is not None for item in contributors) else None
            component["sources"] = [key for key in SOURCES if any(item["source"] == key or
                                    key in item.get("support_sources", []) for item in contributors)]
            component["extra_tags"] = sorted({raw_label(tag) for item in contributors for tag in item.get("extra_tags", [])}
                                             - {tag})
            component["extra_ids"] = sorted({str(i) for item in contributors for i in item.get("extra_ids", [])})
            from stash_temporal import merged_extras
            merged_extras(component)
            component["extra_tags"] = sorted({raw_label(label) for label in component["extra_tags"]})
            merged.append(component)
    return sorted(merged, key=lambda item: (item["start"], item["tag"]))


def combined_hierarchy(app, args, tag_policies=None):
    """Compose the four local YAMLs into one stable ai: category namespace."""
    if args.no_hierarchy:
        return None
    from stash_exclusions import policies_or_empty
    tag_policies = policies_or_empty(tag_policies)
    prefix, root = "ai:category:", "ai:category:AI"
    leaves = defaultdict(set)
    categories = {root: ()}
    for key, source in SOURCES.items():
        path = args.hierarchy if key == "joytag" else getattr(args, key + "_hierarchy")
        data = app.load_unique_yaml(path, "Hierarchy")
        if not isinstance(data, dict) or not isinstance(data.get("groups"), dict):
            raise ValueError(f"Invalid hierarchy: {path}")
        groups = data["groups"].values()
        if any(not isinstance(group, list) or any(not isinstance(label, str) for label in group) for group in groups):
            raise ValueError(f"Invalid hierarchy labels: {path}")
        labels = sorted({label for group in groups for label in group})
        hierarchy = app.load_tag_hierarchy(path, labels, source)
        def converted(name):
            return root if not hierarchy.category_parents[name] else prefix + name[len(hierarchy.category_prefix):]
        for name, parents in hierarchy.category_parents.items():
            mapped, wanted = converted(name), tuple(converted(parent) for parent in parents)
            if mapped in categories and categories[mapped] != wanted:
                raise ValueError(f"Conflicting AI hierarchy category: {mapped}")
            categories[mapped] = wanted
        for label, parents in hierarchy.leaf_parents.items():
            raw_label(label)
            if label in tag_policies[key]:
                continue
            leaves[label].update(converted(parent) for parent in parents)
    fallbacks = {prefix + suffix for suffix in ("Other / Unclassified", "General", "Uncategorized")}
    for parents in leaves.values():
        if parents - fallbacks:
            parents.difference_update(fallbacks)
    if set(categories) & {AI.prefix + label for label in leaves}:
        raise ValueError("A merged hierarchy category collides with a predicted tag.")
    return TagHierarchy({label: tuple(sorted(parents)) for label, parents in leaves.items()},
                            categories, prefix, "replace_managed", AI.hierarchy_description)


def bootstrap_legacy(app, store, scene):
    """Capture all known source namespaces before migrating any associations."""
    scene_id = str(scene["id"])
    for kind, enabled in (("scene", app.APPLY_SCENE_TAGS), ("markers", app.CREATE_MARKERS)):
        if not enabled:
            continue
        existing = store.sources(scene_id, kind)
        if kind == "markers" and not existing and any(app.owned_marker(marker, AI) for marker in scene.get("scene_markers", [])):
            raise RuntimeError("Merged markers exist without their source state. Restore the original --state-db before synchronization.")
        from stash_performance import freeze
        imported = {freeze(item) for item in store.imported_legacy(scene_id, kind, borrow=True)}
        originals = []
        by_model = defaultdict(list)
        if kind == "scene":
            for tag in scene.get("tags", []):
                parsed = legacy_label(tag["name"])
                if parsed:
                    model, label = parsed
                    originals.append(tag)
                    if model not in existing or freeze(tag) not in imported:
                        by_model[model].append({"tag": label, "score": None, "legacy_tag_id": str(tag["id"])})
        else:
            for marker in scene.get("scene_markers", []):
                parsed = legacy_label((marker.get("primary_tag") or {}).get("name", ""))
                if not parsed or not app.owned_marker(marker, SOURCES[parsed[0]]):
                    continue
                model, label = parsed
                originals.append(marker)
                if model in existing and freeze(marker) in imported:
                    continue
                extras, manual_ids = set(), set()
                for tag in marker.get("tags", []):
                    extra = legacy_label(tag["name"])
                    if extra:
                        extras.add(extra[1])
                    elif tag["name"] not in {source.owner_tag for source in SOURCES.values()}:
                        manual_ids.add(str(tag["id"]))
                by_model[model].append(dict(tag=label, start=marker["seconds"], end=marker.get("end_seconds"),
                    score=None, hits=None, extra_tags=sorted(extras - {label}), extra_ids=sorted(manual_ids),
                    legacy_marker_id=str(marker["id"])))
        for model, items in by_model.items():
            segment = {"mode": "scene" if kind == "scene" else "spans", "items": items,
                       "meta": {"origin": "legacy", "captured_at": now(), "revision": None}}
            # Fresh separate-mode outputs augment the cache. Only a successful
            # merged-mode inference replaces that model's previous evidence.
            store.put(scene_id, model, {kind: segment}, {kind: False})
        store.remember_legacy(scene_id, kind, originals)


def evidence(app, store, scene_id, core_policies, scene_policies, metrics=None):
    """Retain source markers and enrich them with compatible shared observations."""
    from stash_temporal import scene_reason
    from stash_fusion import FrameIndex
    from stash_scene import shared_scene_contributions
    report = getattr(store, 'report', None)
    scene_hits, spans = [], []
    need_scene = app.APPLY_SCENE_TAGS or (app.CREATE_MARKERS and app.TEMPORAL_POLICY.enabled
                                        and app.TEMPORAL_POLICY.occurrence_markers)
    scene_records = store.records(scene_id, 'scene', borrow=True) if need_scene else {}
    for model, segments in scene_records.items():
        for segment in segments:
            eligible = []
            for entry in segment["items"]:
                label = raw_label(entry["tag"])
                reason = scene_reason(entry["score"], entry.get("statistics"),
                                      segment["meta"].get("scene_threshold", 0.0), app.TEMPORAL_POLICY)
                if label not in scene_policies[model] and reason:
                    eligible.append({**entry, "source": model, "meta": segment["meta"],
                                     **({"qualification": reason} if entry.get("statistics") else {})})
            # Cap each inference pass separately. With replacement disabled,
            # historical detections and their original confidence remain intact.
            eligible.sort(key=lambda item: (-(item["score"] if item["score"] is not None else -1), item["tag"]))
            scene_hits.extend(eligible[:app.MAX_TAGS] if app.MAX_TAGS and segment["meta"]["origin"] != "legacy" else eligible)
    records = store.records(scene_id, 'markers', borrow=True)
    from stash_thresholds import marker_context
    app = marker_context(app, records)
    if report is not None:
        segments = [segment for values in records.values() for segment in values]
        report.record_counts(input_models=len(records), input_segments=len(segments),
            frame_rows=sum(len(s['items']) for s in segments if s['mode'] == 'frames'),
            retained_detections=sum(len(row['tags']) for s in segments if s['mode'] == 'frames' for row in s['items']))
    with stage(report, 'frame_index'):
        index = FrameIndex(app, records, core_policies, cache=store.cache, tag_policies=scene_policies)
        index.report = report
    with index:
        if app.CREATE_MARKERS:
            spans = marker_contributions(app, records, core_policies, index=index, metrics=metrics, tag_policies=scene_policies)
        if need_scene:
            with stage(report, 'scene_fusion'):
                scene_hits.extend(shared_scene_contributions(app, index, scene_records, scene_policies, scene_hits, metrics))
    if app.CREATE_MARKERS:
        from stash_occurrences import occurrence_contributions
        report = getattr(store, 'report', None)
        with report.measure('occurrence_compute') if report is not None else nullcontext():
            spans.extend(occurrence_contributions(app, scene_hits, records, spans, core_policies, scene_policies, metrics))
    return sorted(scene_hits, key=lambda item: (item["tag"], item["source"])), spans


def marker_segment_signature(segment):
    # Cleanup must use exactly the evidence that supported published markers.
    from stash_performance import StoredSegment, segment_digest
    from stash_thresholds import OVERRIDE
    if OVERRIDE not in segment:
        return segment_digest(segment)
    if isinstance(segment, StoredSegment) and hasattr(segment, "_evidence_digest"):
        return segment._evidence_digest
    signature = segment_digest({k: v for k, v in segment.items() if k != OVERRIDE})
    if isinstance(segment, StoredSegment):
        segment._evidence_digest = signature
    return signature


def marker_contributions(app, records, core_policies, *, index=None, metrics=None, cache=None, tag_policies=None):
    from stash_fusion import FrameIndex
    from stash_thresholds import marker_context
    app = marker_context(app, records)
    if index is not None:
        return _marker_contributions(app, records, core_policies, index, metrics, tag_policies)
    with FrameIndex(app, records, core_policies, cache=cache, tag_policies=tag_policies) as index:
        return _marker_contributions(app, records, core_policies, index, metrics, tag_policies)


def _marker_contributions(app, records, core_policies, index, metrics, tag_policies):
    from stash_temporal import consensus_seeds, build_options, build_markers as temporal_markers
    from stash_exclusions import policies_or_empty, core_policies_with_global
    tag_policies = policies_or_empty(tag_policies if tag_policies is not None else getattr(index, 'tag_policies', None))
    effective_cores = core_policies_with_global(core_policies, tag_policies)
    report = getattr(index, 'report', None)
    with stage(report, 'marker_seeds'):
        seeds = consensus_seeds(records, effective_cores, app.MARKER_TAG_BLOCKLIST, app.TEMPORAL_POLICY,
                                thresholds=app.MARKER_THRESHOLDS)
    signatures = index.signatures
    spans = []
    with stage(report, 'marker_build'):
        for model, segments in records.items():
            for segment in segments:
                if segment["mode"] == "frames":
                    frames = [FramePrediction(Cue(frame["start"], frame["end"], "", (0, 0, 1, 1)),
                               tuple((raw_label(tag), score) for tag, score in frame["tags"]
                                     if tag not in app.MARKER_TAG_BLOCKLIST and tag not in tag_policies[model]))
                              for frame in segment["items"]]
                    from stash_thresholds import marker_threshold
                    kwargs = dict(threshold=marker_threshold(segment, app.MARKER_THRESHOLDS), seeds=seeds,
                                  identity=segment["meta"].get("frame_input_signature"))
                    defer_extras = app.TEMPORAL_POLICY.shared_marker_extras and signatures[id(segment)] in index.by_signature
                    markers, _ = (temporal_markers(app, frames, effective_cores[model], **kwargs,
                                  collect_extras=not defer_extras,
                                  extra_group=index.by_signature.get(signatures[id(segment)], signatures[id(segment)]),
                                  extra_cache=index.cache) if app.TEMPORAL_POLICY.enabled else
                                  app.build_markers(frames, effective_cores[model], **kwargs))
                    options = {**build_options(app), "core_exclusions": {k: sorted(v) for k, v in core_policies.items()},
                               "tag_exclusions": {k: sorted(v) for k, v in tag_policies.items()},
                               "blocklist": sorted(app.MARKER_TAG_BLOCKLIST)}
                    values = [dict(tag=marker.tag, start=marker.start, end=marker.end, score=marker.score,
                                   hits=marker.hits, extra_tags=list(marker.extra_tags),
                                   **(dict(extra_evidence=list(marker.extra_evidence), strong_hits=marker.strong_hits,
                                           support_sources=list(marker.support_sources), extra_support=marker.extra_support, build_options=options,
                                           evidence_signature=signatures[id(segment)],
                                           support_context={source: sorted({signatures[id(s)] for s in records[source]
                                               if s["meta"].get("frame_input_signature") == segment["meta"].get("frame_input_signature")})
                                               for source in marker.support_sources})
                                      if app.TEMPORAL_POLICY.enabled else {})) for marker in markers]
                elif segment["mode"] == "spans":
                    values = segment["items"]
                else:
                    raise ValueError("Unsupported stored marker evidence.")
                for item in values:
                    if item["tag"] not in effective_cores[model] and item["tag"] not in app.MARKER_TAG_BLOCKLIST:
                        from stash_fusion import prune_blocked_extras
                        spans.append(prune_blocked_extras({**item, "source": model, "meta": segment["meta"]},
                                                         app.MARKER_TAG_BLOCKLIST | tag_policies[model]))
    from stash_fusion import enrich
    with stage(report, 'marker_fusion'):
        return enrich(app, records, core_policies, spans, signatures, index, metrics)


def cleanup_support(app, store, scene_id, spans, selected, core_policies, tag_policies=None):
    """Re-evaluate revoked votes, bounded by the already-published source ranges."""
    from types import SimpleNamespace
    from stash_temporal import TemporalPolicy
    from stash_fusion import prune_blocked_extras, refresh_clipped_evidence
    from stash_exclusions import policies_or_empty
    tag_policies = policies_or_empty(tag_policies)
    records, rebuilt, available_by_source = None, {}, {}
    cleaned = []
    for span in spans:
        if span.get('derived') == 'occurrence':
            from stash_occurrences import cleanup_occurrence
            point = cleanup_occurrence(app, store, scene_id, span, selected, core_policies, tag_policies)
            if point is not None:
                cleaned.append(point)
            continue
        own_blocked = app.MARKER_TAG_BLOCKLIST | (tag_policies[span['source']] if span['source'] in selected else frozenset())
        if span.get('derived') != 'shared_frames' and span['source'] in selected and (
                span['tag'] in core_policies[span['source']] or span['tag'] in own_blocked):
            continue
        core_affected = any(source in selected and (span["tag"] in core_policies[source] or
                            span["tag"] in tag_policies[source] or span["tag"] in app.MARKER_TAG_BLOCKLIST)
                            for source in span.get("support_sources", []))
        extra_labels = set(span.get('extra_tags', [])) | set(span.get('extra_support', {}))
        extra_sources = set(span.get('frame_context', {})) | {span['source']}
        extras_affected = any(extra_labels & tag_policies[source] for source in extra_sources & selected)
        affected = core_affected or extras_affected
        if not affected:
            cleaned.append(prune_blocked_extras(span, app.MARKER_TAG_BLOCKLIST))
            continue
        if not span.get('build_options') or not span.get('evidence_signature'):
            # Imported ranges have no frame proofs. Their extras belong to the
            # captured source and can still be removed without loading assets.
            cleaned.append(prune_blocked_extras(span, own_blocked))
            continue
        records = store.records(scene_id, "markers", borrow=True) if records is None else records
        required = span.get("frame_context") or {**span.get("support_context", {}), span["source"]: [span["evidence_signature"]]}
        matching = {}
        for source, signatures in required.items():
            if source not in available_by_source:
                available_by_source[source] = {marker_segment_signature(s): s for s in records.get(source, [])}
            available = available_by_source[source]
            if not set(signatures) <= set(available):
                raise RuntimeError("Consensus sources changed since publication; run --merge-only before cleanup.")
            matching[source] = [available[signature] for signature in signatures]
        options = span["build_options"]
        key = json.dumps([options, required], sort_keys=True)
        if key not in rebuilt:
            context = SimpleNamespace(**vars(app))
            context.TEMPORAL_POLICY = TemporalPolicy.from_snapshot(options["policy"])
            context.MARKER_THRESHOLDS = options.get("marker_thresholds", {})
            for name, value in options["settings"].items():
                setattr(context, name, value)
            context.MARKER_TAG_BLOCKLIST = set(options["blocklist"]) | app.MARKER_TAG_BLOCKLIST
            policies = {source: frozenset(options["core_exclusions"][source]) |
                        (core_policies[source] if source in selected else frozenset()) for source in SOURCES}
            frozen_tags = {source: frozenset(options.get('tag_exclusions', {}).get(source, [])) |
                           (tag_policies[source] if source in selected else frozenset()) for source in SOURCES}
            # Cleanup can revoke old evidence but cannot add newly cached votes.
            rebuilt[key] = (marker_contributions(context, matching, policies, tag_policies=frozen_tags),
                            context, policies, matching, frozen_tags)
        candidates, context, policies, matching, frozen_tags = rebuilt[key]
        for item in candidates:
            if item["tag"] != span["tag"]:
                continue
            if span.get("derived") != "shared_frames" and (item["source"] != span["source"] or item.get("derived")):
                continue
            start, end = max(span["start"], item["start"]), min(span["end"], item["end"])
            if end <= start:
                continue
            clipped = {**item, "start": start, "end": end}
            if (start, end) != (item["start"], item["end"]):
                clipped.update(hits=None, strong_hits=None, extra_evidence=[], cleanup_clipped=True)
                if clipped.get("derived") == "shared_frames":
                    clipped.update(shared_hits=None, shared_strong_hits=None)
                # Legacy proofs can only be clipped; compatible frame records
                # below allow exact scores and hit proofs to be rebuilt.
                clipped["extra_tags"] = sorted(set(item.get("extra_tags", [])) & set(span.get("extra_tags", [])))
                clipped["extra_support"] = {label: dict(
                    intervals=[[max(a,start), min(b,end)] for a,b in evidence["intervals"] if min(b,end)>max(a,start)],
                    sample_times=[t for t in evidence["sample_times"] if start <= t < end])
                    for label,evidence in item.get("extra_support", {}).items()}
                if not refresh_clipped_evidence(context, clipped, matching, policies, frozen_tags):
                    continue
            # Cleanup only revokes published extras; a normal merge can admit
            # additional labels freed by a changed per-frame cap.
            new_extras = (set(clipped.get('extra_tags', [])) | set(clipped.get('extra_support', {}))) - extra_labels
            clipped = prune_blocked_extras(clipped, new_extras)
            clipped = prune_blocked_extras(clipped, app.MARKER_TAG_BLOCKLIST)
            if clipped not in cleaned:
                cleaned.append(clipped)
    return cleaned


def overlaps(marker, wanted, tolerance):
    start, end = marker["seconds"], marker.get("end_seconds")
    end = start if end is None else end
    target_end = wanted.get("end_seconds")
    target_end = wanted["seconds"] if target_end is None else target_end
    return (max(start, wanted["seconds"]) <= min(end, target_end)
            or abs(start - wanted["seconds"]) <= tolerance)


def sync_merged(app, stash, resolver, store, scene, scene_hits, spans, *, migrate=True, cleanup=False, report=None,
                tag_policies=None, publication_key=None):
    """Reconcile only script-owned markers and recorded scene associations."""
    scene_id = str(scene["id"])
    report = report or getattr(store, 'report', None)
    from stash_exclusions import excluded_names
    from stash_occurrences import OCCURRENCE_TAG, is_occurrence
    from stash_marker_groups import for_resolver
    groups = for_resolver(app, resolver)
    from stash_recovery import publications
    blocked_names = excluded_names(tag_policies)
    previous_ranges = []
    if app.CREATE_MARKERS and (not stash.has_end_time or not app.MARKER_USE_END_TIME):
        for snapshot in (*store.snapshot(scene_id, "markers", borrow=True), *publications(store, scene_id, 'markers')):
            if snapshot:
                previous_ranges.extend(union_intervals(snapshot.get("contributions", []), borrow=True))

    def temporal_marker(marker):
        # Point-only Stash still needs the previous logical extent for a cleanup
        # that moves the start, and for preserving IDs/manual tags across a split.
        ends = [item["end"] for item in previous_ranges if item["end"] is not None
                and AI.prefix + item["tag"] == marker["primary_tag"]["name"]
                and abs(round(item["start"], 3) - marker["seconds"]) <= 0.0005]
        return {**marker, "end_seconds": max(ends)} if ends else marker

    with stage(report, 'marker_union'):
        markers = union_intervals(spans, borrow=True) if app.CREATE_MARKERS else []
    if report is not None:
        report.record_counts(marker_contributions=len(spans), union_markers=len(markers), scene_contributions=len(scene_hits))
    scene_labels = {raw_label(hit["tag"]) for hit in scene_hits} if app.APPLY_SCENE_TAGS else set()
    if cleanup:
        # Cleanup never restores outputs that the user removed since the last sync.
        scene_labels.intersection_update(tag["name"][len(AI.prefix):] for tag in scene.get("tags", [])
                                         if tag["name"].startswith(AI.prefix))
        scene_hits = [hit for hit in scene_hits if hit["tag"] in scene_labels]
        present = [marker for marker in scene.get("scene_markers", []) if app.owned_marker(marker, AI)]
        markers = [marker for marker in markers if any(old["primary_tag"]["name"] == AI.prefix + marker["tag"]
                   and overlaps(temporal_marker(old), {"seconds": marker["start"], "end_seconds": marker["end"]}, 0) for old in present)]
        spans = [item for marker in markers for item in marker["contributors"]]
    names = {AI.prefix + label for label in scene_labels}
    for marker in markers:
        names.update(AI.prefix + label for label in [marker["tag"], *marker["extra_tags"]])
    if markers:
        names.add(AI.owner_tag)
    if any(is_occurrence(marker) for marker in markers):
        names.add(OCCURRENCE_TAG)
    with stage(report, 'tag_resolution'):
        ids = {name: resolver.resolve(name) for name in sorted(names)}
    if markers and not ids.get(AI.owner_tag):
        raise ValueError("Missing AI marker ownership tag; enable CREATE_MISSING.")
    # Refresh after tag creation/reparenting, preserving newly added manual tags.
    with stage(report, 'scene_refresh'):
        scene = (stash.find_scene_for_merge(scene_id) if hasattr(stash, 'find_scene_for_merge')
                 else stash.find_scene(scene_id))
    if report is not None:
        report.record_counts(existing_markers=len(scene.get('scene_markers', [])), existing_scene_tags=len(scene.get('tags', [])))
    # A manually maintained ai: marker already supplies a location. It remains
    # untouched, and must not receive a duplicate automatically generated point.
    manual_labels = {m['primary_tag']['name'] for m in scene.get('scene_markers', [])
                     if not app.owned_marker(m, AI)}
    suppressed = {m['tag'] for m in markers if is_occurrence(m) and AI.prefix + m['tag'] in manual_labels}
    if suppressed:
        markers = [m for m in markers if not (is_occurrence(m) and m['tag'] in suppressed)]
        spans = [s for s in spans if not (s.get('derived') == 'occurrence' and s['tag'] in suppressed)]
    # Imported/manual extras may retain IDs in old namespaces. Resolve each
    # unknown ID at most once through the shared resolver, not per marker.
    for tag in scene.get('tags', []):
        resolver.names_by_id[str(tag['id'])] = tag['name']
    for marker in scene.get('scene_markers', []):
        for tag in [marker['primary_tag'], *marker.get('tags', [])]:
            resolver.names_by_id[str(tag['id'])] = tag['name']
    def allowed_id(identifier):
        name = resolver.name_for_id(str(identifier))
        return name is not None and name not in blocked_names and not groups.owns_id(identifier)
    def manual_extra(tag):
        return (not tag['name'].startswith(AI.prefix) and tag['name'] not in blocked_names
                and not groups.owns(tag))
    current_tags = scene.get("tags", [])
    snapshots = {}
    migrated_scene_ids = {str(tag["id"]) for tag in current_tags if legacy_label(tag["name"])} if migrate else set()
    published, pending = store.snapshot(scene_id, "scene", borrow=True)
    owned_before = set((published or {}).get("owned_ids", [])) | set((pending or {}).get("owned_ids", []))
    owned_before.update(i for snapshot in publications(store, scene_id, 'scene') for i in snapshot.get('owned_ids', []))
    existing_ids = {str(tag["id"]) for tag in current_tags}
    desired_ids = {ids[AI.prefix + label] for label in scene_labels if ids.get(AI.prefix + label)}
    unresolved_labels = {label for label in scene_labels if not ids.get(AI.prefix + label)}
    # CREATE_MISSING=False must not remove a legacy association lacking a replacement.
    migrated_scene_ids.difference_update(str(tag["id"]) for tag in current_tags
        if (parsed := legacy_label(tag["name"])) and parsed[1] in unresolved_labels)
    forbidden_ids = {str(tag['id']) for tag in current_tags if tag['name'] in blocked_names}
    preserved = existing_ids - owned_before - migrated_scene_ids - forbidden_ids
    final_ids = preserved | desired_ids
    if app.APPLY_SCENE_TAGS:
        snapshots["scene"] = {"owned_ids": sorted(desired_ids - preserved), "contributions": scene_hits}
    with stage(report, 'marker_matching'):
        wanted = []
        resolved_marker_labels = set()
        for marker in markers:
            primary = ids.get(AI.prefix + marker["tag"])
            if not primary:
                continue
            resolved_marker_labels.add(marker["tag"])
            extras = {ids[AI.prefix + label] for label in marker["extra_tags"] if ids.get(AI.prefix + label)}
            extras.update(identifier for identifier in marker["extra_ids"] if allowed_id(identifier))
            extras.add(ids[AI.owner_tag])
            extras.add(groups.resolve(marker['tag'], is_occurrence(marker)))
            if is_occurrence(marker) and ids.get(OCCURRENCE_TAG):
                extras.add(ids[OCCURRENCE_TAG])
            extras.discard(primary)
            payload = {"scene_id": scene_id, "primary_tag_id": primary, "tag_ids": sorted(extras),
                       "title": AI.title_prefix + marker["tag"] + " [" + "+".join(source_codes(marker["sources"])) + "]",
                       "seconds": round(marker["start"], 3)}
            if is_occurrence(marker):
                payload['title'] += ' · wykrycie'
            if stash.has_end_time:
                payload["end_seconds"] = round(marker["end"], 3) if marker["end"] is not None and app.MARKER_USE_END_TIME else None
            wanted.append((payload, {"seconds": marker["start"], "end_seconds": marker["end"]}))
        current_markers = scene.get("scene_markers", [])
        owned = [marker for marker in current_markers if app.owned_marker(marker, AI)]
        owned_by_primary = defaultdict(list)
        for marker in owned:
            owned_by_primary[str(marker["primary_tag"]["id"])].append(marker)
        available = {str(marker["id"]): marker for marker in owned}
        creates, updates, unchanged = [], [], 0
        for payload, interval in wanted:
            same_primary = owned_by_primary[payload["primary_tag_id"]]
            related = [marker for marker in same_primary if overlaps(temporal_marker(marker), interval, 0)]
            # Keep manual extras from every marker absorbed by the union, including
            # duplicates that will be deleted. A split carries them to both pieces.
            payload["tag_ids"] = sorted(set(payload["tag_ids"]) | {str(tag["id"]) for marker in related
                for tag in marker.get("tags", []) if manual_extra(tag)})
            matches = [marker for marker in same_primary if str(marker["id"]) in available
                       and overlaps(temporal_marker(marker), interval, app.MARKER_MATCH_TOLERANCE_SECONDS)]
            if not matches:
                creates.append(payload)
                continue
            def match_key(marker):
                marker = temporal_marker(marker)
                old_end = marker.get("end_seconds") if marker.get("end_seconds") is not None else marker["seconds"]
                new_end = interval["end_seconds"] if interval["end_seconds"] is not None else interval["seconds"]
                shared = max(0, min(old_end, new_end) - max(marker["seconds"], payload["seconds"]))
                return (-shared, abs(marker["seconds"] - payload["seconds"]), str(marker["id"]))
            current = min(matches, key=match_key)
            marker_id = str(current["id"])
            del available[marker_id]
            # Additional manually attached tags outside the generated namespace survive.
            payload["tag_ids"] = sorted(set(payload["tag_ids"]) | {str(tag["id"]) for tag in current.get("tags", [])
                                                                  if manual_extra(tag)})
            if app.marker_differs(current, payload):
                updates.append({**payload, "id": marker_id})
            else:
                unchanged += 1
        deletes = set(available) if app.CREATE_MARKERS else set()
        if migrate and app.CREATE_MARKERS:
            predicted_labels = {marker["tag"] for marker in markers}
            for marker in current_markers:
                parsed = legacy_label((marker.get("primary_tag") or {}).get("name", ""))
                if parsed and app.owned_marker(marker, SOURCES[parsed[0]]):
                    if parsed[1] not in predicted_labels or parsed[1] in resolved_marker_labels:
                        deletes.add(str(marker["id"]))
        if app.CREATE_MARKERS and blocked_names:
            deletes.update(str(marker['id']) for marker in current_markers
                           if marker['primary_tag']['name'] in blocked_names)
            changed_ids = deletes | {payload['id'] for payload in updates}
            for marker in current_markers:
                if str(marker['id']) in changed_ids:
                    continue
                payload = app.marker_extra_cleanup_payload(marker, scene_id, blocked_names, stash.has_end_time)
                if payload is not None:
                    updates.append(payload)
    if app.CREATE_MARKERS:
        snapshots["markers"] = {"contributions": spans}
    scene_changed = app.APPLY_SCENE_TAGS and final_ids != existing_ids
    stats = Counter(scene_tags_updated=int(scene_changed), legacy_scene_tags_migrated=len(migrated_scene_ids) if app.APPLY_SCENE_TAGS else 0,
                    markers_created=len(creates), markers_updated=len(updates), markers_deleted=len(deletes),
                    markers_unchanged=unchanged, merged_markers=len(wanted), merged_scene_tags=len(desired_ids))
    stats['short_extras'] = sum(item.get('qualification') == 'short_episode'
        for marker in markers for item in marker.get('extra_evidence', {}).values())
    stats['occurrence_markers'] = sum(is_occurrence(marker) for marker in markers)
    stats['occurrence_manual_locations'] = len(suppressed)
    stats['cross_model_extras'] = sum(bool(set(item.get('source_evidence', {})) - set(marker['sources']))
        for marker in markers for tag, item in marker.get('extra_evidence', {}).items() if tag in marker['extra_tags'])
    if stash.dry_run:
        groups.record_preview(scene, creates, updates, deletes)
        write = report.detail if report is not None else print
        write(f"[DRY] scene {scene_id}: ai scene tags={len(desired_ids)}, markers create={len(creates)} "
              f"update={len(updates)} delete={len(deletes)}; migrate scene tags={stats['legacy_scene_tags_migrated']}.")
        if app.APPLY_SCENE_TAGS:
            if report is None or report.verbose:
                write("  Scene tags: " + (", ".join(AI.prefix + label for label in sorted(scene_labels)) or "(none)"))
        for marker in markers if report is None or report.verbose else ():
            end = app.format_time(marker["end"]) if marker["end"] is not None else "point (end unknown)"
            write(f"  {app.format_time(marker['start'])} -> {end}: ai:{marker['tag']} "
                  f"[{'+'.join(source_codes(marker['sources']))}]")
        store.publish(scene_id, snapshots)  # temporary preview overlay only
        return stats
    with stage(report, 'publication_prepare'):
        store.prepare(scene_id, snapshots)
    created = []
    with stage(report, 'publication_mutations'):
        for payload in creates:
            created.append((payload, stash.create_marker(payload)))
        for payload in updates:
            stash.update_marker(payload)
        if scene_changed:
            stash.update_scene_tags(scene_id, sorted(final_ids))
        for marker_id in sorted(deletes):
            stash.delete_marker(marker_id)
    publication = None
    with stage(report, 'publication_fingerprint'):
        if publication_key is not None and not resolver.missing:
            from stash_performance import projected_scene, scene_fingerprint
            observed = projected_scene(scene, resolver, created, updates, deletes,
                                       final_ids if app.APPLY_SCENE_TAGS else existing_ids)
            if observed is None:
                observed = stash.find_scene(scene_id)
            publication = publication_key, scene_fingerprint(observed)
    with stage(report, 'publication_commit'):
        store.commit_prepared(scene_id, snapshots, publication=publication)
    return stats


def capture_outputs(app, config, vocabulary, cues, predictions, frames, analysis=None):
    from stash_temporal import marker_floor
    cue_data = [[cue.start, cue.end, cue.image, list(cue.rect)] for cue in cues]
    meta = {"origin": "inference", "captured_at": now(), "repo": config.source.repo,
            "revision": config.source.revision, "vocabulary_sha256": app.vocabulary_digest(vocabulary.labels),
            "input_cues_sha256": hashlib.sha256(json.dumps(cue_data, ensure_ascii=False).encode()).hexdigest(),
            "scene_threshold": config.scene_threshold, "marker_threshold": config.marker_threshold,
            "aggregation": app.AGGREGATION, "top_k_frames": app.TOP_K_FRAMES}
    if app.TEMPORAL_POLICY.enabled:
        meta.update(analysis_format=3, marker_score_floor=marker_floor(app.TEMPORAL_POLICY, config.marker_threshold),
                    scene_support_floor=config.scene_threshold * app.TEMPORAL_POLICY.scene_support_ratio)
    outputs = {}
    if app.APPLY_SCENE_TAGS:
        outputs["scene"] = {"mode": "scene", "meta": meta,
                            "items": [dict(tag=tag, score=score,
                                           **({"statistics": analysis["scene_statistics"][tag]} if analysis and
                                              tag in analysis.get("scene_statistics", {}) else {})) for tag, score in predictions]}
    if app.CREATE_MARKERS:
        outputs["markers"] = {"mode": "frames", "meta": meta, "items": [
            dict(start=frame.cue.start, end=frame.cue.end, tags=list(frame.tags)) for frame in frames]}
    return outputs


def has_outputs(app, store, scene_id, model):
    from stash_progress import completed, output_kinds
    return completed(store, scene_id, model, output_kinds(app))


def run_merged(app, settings, args, catalogs, hierarchy, core_policies, scene_policies):
    from stash_diagnostics import MergeDiagnostics
    from stash_runtime import RunReport
    from tqdm import tqdm
    report = getattr(args, 'report', None) or RunReport(getattr(args, 'verbose', False), tqdm.write)
    args.report = report
    with MergeDiagnostics(report, getattr(args, 'profile_merge', None), dry_run=app.DRY_RUN,
                          merge_only=args.merge_only, force=getattr(args, 'force_merge', False)):
        return _run_merged(app, settings, args, catalogs, hierarchy, core_policies, scene_policies)


def _run_merged(app, settings, args, catalogs, hierarchy, core_policies, scene_policies):
    from tqdm import tqdm
    import stash_progress as progress
    import stash_performance as performance
    stats = Counter()
    from stash_runtime import RunReport
    report = getattr(args, 'report', None) or RunReport(getattr(args, 'verbose', False), tqdm.write)
    with StateStore(args.state_db, scope_for(app.STASH_URL), app.DRY_RUN,
                    cache_bytes=int(getattr(args, 'ram_cache_gb', 0) * 2**30),
                    allow_cleanup=getattr(args, 'abandon_pending', False),
                    memory_budget_bytes=int(getattr(args, 'ram_budget_gb', 32) * 2**30)) as store:
        store.report = report
        if getattr(args, 'abandon_pending', False):
            from stash_recovery import abandon
            stats.update(abandon(store, report.write))
        stash = app.create_stash_client(app.DRY_RUN)
        stash.report = report
        try:
            if hierarchy or app.CREATE_MARKERS:
                stash.detect_tag_hierarchy()
            if app.CREATE_MARKERS:
                stash.detect_marker_capabilities()
                if not stash.has_end_time:
                    print("Stash supports point markers only; full merged ranges remain in the source state.")
            if hasattr(stash, 'configure_merge_reads'):
                stash.configure_merge_reads(report, merge_only=args.merge_only)
            scenes = {}
            resolver = app.TagResolver(stash, hierarchy, source=AI, report=report.detail, create_missing=app.CREATE_MISSING)
            use_catalog = bool(hierarchy or app.CREATE_MARKERS)
            if use_catalog or not args.scene_id or len(args.scene_id) > 1:
                resolver.preload()
            base_key = performance.publication_key(app, args, hierarchy, core_policies, scene_policies, stash)
            merge_key = (performance.digest([base_key, performance.catalog_fingerprint(resolver)])
                         if use_catalog else base_key)
            store.preload_sources()
            with stage(report, 'initial_scan'):
                for current in tqdm(stash.iter_scenes(args.scene_id), desc="Read sources", unit="scene"):
                    scene_id = str(current["id"])
                    try:
                        with stage(report, 'scan_fingerprint'):
                            fingerprint = performance.scene_fingerprint(current)
                        if performance.checked(store, scene_id, merge_key, fingerprint):
                            stats['bootstrap_reused'] += 1
                        else:
                            with report.measure('bootstrap'):
                                bootstrap_legacy(app, store, current)
                        # Keep only input paths/ID between passes; marker bodies need
                        # no collection-sized in-memory cache and are refreshed at sync.
                        scenes[scene_id] = {"id": scene_id, "paths": current.get("paths") or {},
                                            'fingerprint': fingerprint}
                    except Exception as exc:
                        stats["errors"] += 1
                        report.issue('read sources', '', scene_id, exc, app.safe_error(exc, api_key=app.STASH_API_KEY))
            if getattr(args, 'rethreshold_markers', False):
                from stash_thresholds import rethreshold
                if stats['errors']:
                    report.finish('Marker thresholds (not applied)', stats, store.cache)
                    return 1
                stats.update(rethreshold(app, store, scenes, settings, report))
                if stats['errors']:
                    report.finish('Marker thresholds (not applied)', stats, store.cache)
                    return 1
            kinds = progress.output_kinds(app)
            inputs = progress.RunInputs(stash, store.cache, report)
            if not args.merge_only:
                if args.restart:
                    progress.request_refresh(store, scenes, [c.source.key for c in settings], kinds)
                progress.preload_status(store)
                for config in settings:
                    key = config.source.key
                    tagger = None
                    identity = progress.model_identity(app, config, catalogs[key])
                    try:
                        bar = tqdm(scenes.items(), desc=key + " inference", unit="scene")
                        for scene_id, current in bar:
                            report.progress(bar, stats)
                            paths = current["paths"]
                            if not paths.get("sprite") or not paths.get("vtt"):
                                stats["skipped"] += 1
                                continue
                            if app.SKIP_EXISTING_OUTPUTS and not args.restart and has_outputs(app, store, scene_id, key):
                                stats["skipped"] += 1
                                continue
                            try:
                                store.cache.pressure(force=True)
                                cues, media, signature = inputs.inspect(app, paths, identity)
                                if args.resume and progress.completed(store, scene_id, key, kinds, signature):
                                    stats["reused"] += 1
                                    continue
                                if args.resume and not args.restart:
                                    from stash_thresholds import reuse
                                    if reuse(app, store, scene_id, config, identity, cues, media.hashes, kinds):
                                        stats['marker_cache_reused'] += 1
                                        continue
                                if tagger is None:
                                    try:
                                        tagger = app.load_tagger(config, catalogs[key], app.prepare_image)
                                    except Exception as exc:
                                        stats["errors"] += 1
                                        report.issue('load model', key, None, exc, app.safe_error(exc, api_key=app.STASH_API_KEY))
                                        break
                                    # Auto-download may just have supplied missing weight files.
                                    identity = progress.model_identity(app, config, catalogs[key])
                                    signature = progress.digest({"model": identity,
                                        "cues": [[c.start,c.end,c.image,list(c.rect)] for c in cues], "media": media.hashes})
                                if not args.resume:
                                    with store.connection:
                                        progress.clear_checkpoint(store, scene_id, key)
                                checkpoint = progress.Checkpoint(store, scene_id, key,
                                    progress.digest({"input": signature, "kinds": kinds}), len(cues),
                                    len(catalogs[key].labels), args.checkpoint_batches)
                                analysis = {}
                                predictions, frames = app.predict_scene(tagger, media, paths, cues, raw_scene=True,
                                    checkpoint=checkpoint, prefetch=args.prefetch_batches, analysis=analysis, report=report,
                                    prepare_workers=getattr(args, 'prepare_workers', 1))
                                outputs = capture_outputs(app, config, catalogs[key], cues, predictions, frames, analysis)
                                for segment in outputs.values():
                                    segment["meta"].update(inference_signature=signature,
                                        frame_input_signature=progress.digest({"cues": [[c.start,c.end,c.image,list(c.rect)] for c in cues],
                                                                               "media": media.hashes}))
                                progress.finish(store, scene_id, key, outputs,
                                    {"scene": app.REPLACE_SCENE_TAGS, "markers": app.REPLACE_MARKERS})
                                stats["processed"] += 1
                            except Exception as exc:
                                stats["errors"] += 1
                                report.issue('inference', key, scene_id, exc, app.safe_error(exc, api_key=app.STASH_API_KEY))
                    finally:
                        if tagger is not None:
                            tagger.close()
            bar = tqdm(scenes.items(), desc="Merge to ai:", unit="scene")
            for scene_id, current in bar:
                with report.diagnostics.scene(scene_id) as sample:
                    try:
                        with stage(report, 'publication_check'):
                            reason = ('forced' if getattr(args, 'force_merge', False) else
                                      performance.check_reason(store, scene_id, merge_key, current['fingerprint']))
                        sample['reason'] = reason
                        if reason == 'reused':
                            sample['status'] = 'reused'
                            stats['merge_reused'] += 1
                            report.progress(bar, stats)
                            continue
                        stats['merge_' + reason] += 1
                        diagnostics = Counter()
                        cache_before = store.cache.stats.copy()
                        with report.measure('merge_compute'):
                            scene_hits, spans = evidence(app, store, scene_id, core_policies, scene_policies, diagnostics)
                        with report.measure('merge_plan'):
                            result = sync_merged(app, stash, resolver, store, current, scene_hits, spans, report=report,
                                                 tag_policies=scene_policies, publication_key=merge_key)
                        stats.update(result)
                        sample['statistics'] = dict(result + diagnostics)
                        report.record_counts(**{'cache_' + key: value-cache_before[key]
                                                for key, value in store.cache.stats.items()})
                        stats.update(diagnostics)
                        stats["scenes_synced"] += 1
                    except Exception as exc:
                        sample['status'] = 'error'
                        stats["errors"] += 1
                        report.issue('merge', '', scene_id, exc, app.safe_error(exc, api_key=app.STASH_API_KEY))
                    report.progress(bar, stats)
            if use_catalog:
                final_key = performance.digest([base_key, performance.catalog_fingerprint(resolver)])
                if final_key != merge_key:
                    with store.connection:
                        store.connection.execute('UPDATE merge_checks SET policy=? WHERE scope=? AND policy=?',
                                                 (final_key, store.scope, merge_key))
            app.print_hierarchy_summary(resolver)
            if args.prune_empty_tags and not stats["errors"]:
                from stash_prune import prune_tags
                # Groups use the projected marker counts below, including in dry-run.
                stats.update(prune_tags(stash, args.prune_prefixes, report=report.detail,
                                        skip_marker_groups=app.CREATE_MARKERS))
            elif not stats['errors']:
                from stash_exclusions import prune_excluded_definitions
                stats.update(prune_excluded_definitions(app, stash, scene_policies, report.detail))
            if app.CREATE_MARKERS and not stats['errors']:
                from stash_marker_groups import for_resolver, prune_groups
                stats.update(prune_groups(stash, report.detail,
                    usage_delta=for_resolver(app, resolver).usage_delta))
        finally:
            stash.close()
        report.finish('Merged' + (' (planned)' if app.DRY_RUN else ''), stats, store.cache)
    return int(bool(stats["errors"]))


def run_hierarchy(app, args, catalogs, hierarchy, tag_policies=None):
    from stash_exclusions import policies_or_empty
    tag_policies = policies_or_empty(tag_policies)
    stash = app.create_stash_client(app.DRY_RUN)
    try:
        stash.detect_tag_hierarchy()
        resolver = app.TagResolver(stash, hierarchy, source=AI, report=args.report.detail, create_missing=app.CREATE_MISSING)
        resolver.preload()
        labels = sorted({label for source, vocabulary in catalogs.items() for label in vocabulary.labels
                         if label not in tag_policies[source]})
        for label in labels:
            if args.create_all_tags or AI.prefix + label in resolver.cache:
                resolver.resolve(AI.prefix + label)
        app.print_hierarchy_summary(resolver)
    finally:
        stash.close()
    return 0


def cleanup_scene(app, stash, resolver, store, current, settings, selected, args,
                  core_policies, scene_policies, scoped_tags, report):
    """Reconcile one affected scene; refresh only after actual writes."""
    from stash_exclusions import excluded_names
    from stash_scene import cleanup_scene_support
    scene_id, stats = str(current["id"]), Counter()
    if getattr(args, 'output_mode', 'merged') == 'merged':
        published = {}
        for kind, enabled in (("scene", app.APPLY_SCENE_TAGS), ("markers", app.CREATE_MARKERS)):
            if not enabled:
                continue
            snapshot, pending = store.snapshot(scene_id, kind, borrow=True)
            if pending is not None:
                raise RuntimeError("An interrupted merge is pending; run --merge-only before cleanup.")
            from stash_recovery import reconcile
            published[kind], recovered = reconcile(app, stash, store, current, kind, snapshot)
            stats['publications_reconciled'] += int(recovered)
        if app.CREATE_MARKERS and published.get("markers") is None and any(
                app.owned_marker(marker, AI) for marker in current.get("scene_markers", [])):
            raise RuntimeError("AI marker source state is missing; restore the original --state-db.")
        hits = (published.get("scene") or {}).get("contributions", [])
        spans = (published.get("markers") or {}).get("contributions", [])
        clean_hits = cleanup_scene_support(app, store, scene_id, hits, selected, scene_policies)
        clean_spans = cleanup_support(app, store, scene_id, spans, selected, core_policies, scene_policies)
        blocked_names = excluded_names(scoped_tags)
        forbidden_present = (app.APPLY_SCENE_TAGS and any(tag['name'] in blocked_names for tag in current.get('tags', []))) or (
            app.CREATE_MARKERS and any(tag['name'] in blocked_names for marker in current.get('scene_markers', [])
                                      for tag in [marker['primary_tag'], *marker.get('tags', [])]))
        if clean_hits != hits or clean_spans != spans or forbidden_present:
            result = sync_merged(app, stash, resolver, store, current, clean_hits, clean_spans,
                                 migrate=False, cleanup=True, report=report, tag_policies=scoped_tags)
            stats.update(result)
            if not stash.dry_run and any(result[name] for name in (
                    'scene_tags_updated', 'markers_created', 'markers_updated', 'markers_deleted')):
                current = stash.find_scene(scene_id)
    # Most scenes have no old source-prefixed outputs. Reading the same scene
    # once for every configured model previously dominated no-op cleanup.
    for config in settings:
        result, source = Counter(), config.source
        if app.APPLY_SCENE_TAGS:
            result.update(app.cleanup_scene_tags(stash, current, scene_policies[source.key], source, report.detail))
        if app.CREATE_MARKERS:
            result.update(app.cleanup_scene_marker_cores(stash, current, core_policies[source.key], report.detail,
                                                        source, scene_policies[source.key]))
        stats.update(result)
        if not stash.dry_run and any(result[name] for name in ('scene_tags_cleaned', 'markers_deleted', 'markers_updated')):
            current = stash.find_scene(scene_id)
    return stats


def run_cleanup(app, settings, args, core_policies, scene_policies):
    """Global bans operate on tags; source-specific changes reconcile affected scenes."""
    from tqdm import tqdm
    from stash_cleanup import CleanupClient
    from stash_exclusions import selected_policies, prune_excluded_definitions
    from stash_runtime import RunReport
    selected = {config.source.key for config in settings}
    scoped_tags = selected_policies(scene_policies, selected)
    stats = Counter()
    report = getattr(args, 'report', None) or RunReport(getattr(args, 'verbose', False), tqdm.write)
    wide = not args.scene_id and selected == set(SOURCES)
    with StateStore(args.state_db, scope_for(app.STASH_URL), app.DRY_RUN,
                    cache_bytes=int(getattr(args, 'ram_cache_gb', 0) * 2**30),
                    allow_cleanup=wide or getattr(args, 'abandon_pending', False), cleanup_preview=True,
                    memory_budget_bytes=int(getattr(args, 'ram_budget_gb', 32) * 2**30)) as store:
        if getattr(args, 'abandon_pending', False):
            from stash_recovery import abandon
            stats.update(abandon(store, report.write))
        stash = app.create_stash_client(app.DRY_RUN)
        resolver = None
        try:
            scene_ids, cleanup_stash = args.scene_id or None, stash
            strategy = getattr(args, 'cleanup_strategy', 'auto')
            client = CleanupClient(stash) if strategy in ('auto', 'sql') else None
            sql = None
            if strategy in ('auto', 'sql') and client.sql_available:
                from stash_cleanup_sql import SQLCleanup
                candidate = SQLCleanup(client, args, stats, report)
                try:
                    candidate.probe()
                    sql = candidate
                except Exception as exc:
                    if strategy == 'sql' or store.cleanup_pending:
                        raise
                    report.write('SQL cleanup unavailable (' + app.safe_error(exc, api_key=app.STASH_API_KEY) + '); scanning scenes instead.')
            if sql is not None:
                from stash_cleanup_sql import run_sql_cleanup
                report.write('Cleanup by indexed SELECT: ID cursors, no global marker counts; writes use Stash API.')
                if store.cleanup_pending:
                    stats['cleanup_resumed'] = 1
                scene_ids, cleanup_stash = run_sql_cleanup(app, sql, store, settings, args, core_policies, scene_policies, stats)
            elif strategy == 'sql':
                raise RuntimeError('Stash does not expose the required SQL cleanup API. Use --cleanup-strategy scenes for compatibility.')
            elif store.cleanup_pending:
                raise RuntimeError('Global cleanup is interrupted and the required Stash API is unavailable; '
                                   'restore API access and rerun the same cleanup.')
            elif not args.scene_id:
                report.write('Cleanup scans scenes; global findSceneMarkers filtering is disabled. SQL cleanup is faster for large collections.')
            # An empty candidate list must not be passed to iter_scenes: its empty
            # list convention means the whole library, not zero scenes.
            if scene_ids != []:
                if app.CREATE_MARKERS:
                    stash.detect_marker_capabilities()
                    stash.detect_tag_hierarchy()
                resolver = app.TagResolver(cleanup_stash, source=AI, report=report.detail, create_missing=app.CREATE_MISSING)
                bar = tqdm(cleanup_stash.iter_scenes(scene_ids), desc="Affected scene cleanup", unit="scene")
                for current in bar:
                    scene_id = str(current["id"])
                    try:
                        stats.update(cleanup_scene(app, cleanup_stash, resolver, store, current, settings, selected,
                                                   args, core_policies, scene_policies, scoped_tags, report))
                        stats["processed"] += 1
                    except Exception as exc:
                        stats["errors"] += 1
                        report.issue('cleanup', '', scene_id, exc, app.safe_error(exc, api_key=app.STASH_API_KEY))
                    report.progress(bar, stats)
            if not stats['errors']:
                if getattr(args, 'prune_empty_tags', False):
                    from stash_prune import prune_tags
                    stats.update(sql.prune(prefixes=args.prune_prefixes, skip_marker_groups=app.CREATE_MARKERS)
                                 if sql is not None else prune_tags(stash, args.prune_prefixes, report=report.detail,
                                                                  skip_marker_groups=app.CREATE_MARKERS))
                elif not stats['cleanup_fast_path'] or len({frozenset(v) for v in scene_policies.values()}) > 1:
                    if sql is not None and app.APPLY_SCENE_TAGS and app.CREATE_MARKERS:
                        stats.update(sql.prune(scoped_tags))
                    else:
                        stats.update(prune_excluded_definitions(app, stash, scoped_tags, report.detail))
                if app.CREATE_MARKERS:
                    from stash_marker_groups import for_resolver, prune_groups
                    delta = Counter(getattr(cleanup_stash, 'marker_group_usage_delta', {}))
                    if resolver is not None:
                        delta.update(for_resolver(app, resolver).usage_delta)
                    candidates = sql.marker_group_candidates(delta) if sql is not None else None
                    stats.update(prune_groups(stash, report.detail, candidate_ids=candidates, usage_delta=delta))
        except Exception as exc:
            stats['errors'] += 1
            report.issue('cleanup', '', None, exc, app.safe_error(exc, api_key=app.STASH_API_KEY))
        finally:
            stash.close()
        report.finish('Cleanup' + (' (planned)' if app.DRY_RUN else ''), stats, store.cache)
    return int(bool(stats["errors"]))


def explain(app, args, core_policies, scene_policies):
    label = raw_label(args.explain_tag.removeprefix(AI.prefix))
    # Inspection always opens a read-only snapshot, even without --dry-run.
    with StateStore(args.state_db, scope_for(app.STASH_URL), True,
                    cache_bytes=int(getattr(args, 'ram_cache_gb', 0) * 2**30),
                    memory_budget_bytes=int(getattr(args, 'ram_budget_gb', 32) * 2**30)) as store:
        reports = []
        for scene_id in dict.fromkeys(args.scene_id):
            diagnostics = Counter()
            hits, spans = evidence(app, store, scene_id, core_policies, scene_policies, diagnostics)
            matching = [span for span in spans if span["tag"] == label]
            recorded_sources = {}
            for kind in KINDS:
                sources = {}
                for source, segments in store.records(scene_id, kind).items():
                    matches = []
                    for segment in segments:
                        items = ([dict(start=frame["start"], end=frame["end"], score=score)
                                  for frame in segment["items"] for tag, score in frame["tags"] if tag == label]
                                 if segment["mode"] == "frames" else [entry for entry in segment["items"] if entry["tag"] == label])
                        if items:
                            matches.append({"meta": segment["meta"], "detections": items})
                    if matches:
                        sources[source] = {"excluded": label in scene_policies[source] or (kind == 'markers' and label in core_policies[source]),
                                           "records": matches}
                recorded_sources[kind] = sources
            reports.append({"scene_id": str(scene_id), "tag": AI.prefix + label, 'diagnostics': dict(diagnostics),
                            "scene_contributors": [hit for hit in hits if hit["tag"] == label],
                            "markers": union_intervals(matching), "recorded_sources": recorded_sources,
                            "as_marker_extra": [dict(tag=AI.prefix + marker["tag"], start=marker["start"], end=marker["end"],
                                                     evidence=marker["extra_evidence"][label])
                                                for marker in union_intervals(spans) if label in marker.get("extra_evidence", {})],
                            "pending_sync": any(store.snapshot(scene_id, kind, borrow=True)[1] is not None for kind in KINDS)})
    print(json.dumps(reports, ensure_ascii=False, indent=2, allow_nan=False))
    return 0
