"""Derived indexes and publication checks; raw evidence formats stay compatible."""
from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict
import hashlib
import json
from pathlib import Path


def digest(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
        separators=(',', ':'), allow_nan=False).encode()).hexdigest()


class StoredSegment(dict):
    """Internally borrowed immutable record; private caches never enter JSON.

    Public detached copies are ordinary mutable dicts, without cached digests.
    """
    def __deepcopy__(self, memo):
        result = {}
        memo[id(self)] = result
        result.update((deepcopy(k, memo), deepcopy(v, memo)) for k, v in self.items())
        return result


def segment_digest(segment):
    if isinstance(segment, StoredSegment):
        if not hasattr(segment, '_digest'):
            segment._digest = digest(segment)
        return segment._digest
    return digest(segment)


def frame_group(segment):
    if isinstance(segment, StoredSegment) and hasattr(segment, '_frame_group'):
        return segment._frame_group
    value = digest({'input': segment['meta'].get('frame_input_signature'),
        'times': [[float(row['start']), float(row['end'])] for row in segment['items']]})
    if isinstance(segment, StoredSegment):
        segment._frame_group = value
    return value


def initialize(store):
    """Derived cache invalidated transactionally, including writes by older code."""
    db = store.connection
    db.executescript('''
        CREATE TABLE IF NOT EXISTS merge_checks (
            scope TEXT NOT NULL, scene TEXT NOT NULL, policy TEXT NOT NULL,
            fingerprint TEXT NOT NULL, PRIMARY KEY(scope, scene));
        CREATE INDEX IF NOT EXISTS snapshots_pending_scope
            ON snapshots(scope, scene, kind) WHERE pending IS NOT NULL;
    ''')
    # legacy_imports is only an anti-duplication ledger. Any newly imported
    # legacy evidence also writes results, which already invalidates merge_checks.
    # Drop triggers created by older versions so pruning dead ledger rows does not
    # force unrelated scenes through merge again.
    for event in ('insert', 'update', 'delete'):
        db.execute(f'DROP TRIGGER IF EXISTS perf_legacy_imports_{event}')
    for table in ('results', 'result_thresholds', 'snapshots'):
        for event, row in (('INSERT', 'NEW'), ('UPDATE', 'NEW'), ('DELETE', 'OLD')):
            db.execute(f'''CREATE TRIGGER IF NOT EXISTS perf_{table}_{event.lower()}
                AFTER {event} ON {table} BEGIN
                    DELETE FROM merge_checks WHERE scope={row}.scope AND scene={row}.scene;
                END''')
    for event, row in (('INSERT', 'NEW'), ('UPDATE', 'NEW'), ('DELETE', 'OLD')):
        db.execute(f'''CREATE TRIGGER IF NOT EXISTS perf_cleanup_{event.lower()}
            AFTER {event} ON cleanup_state BEGIN
                DELETE FROM merge_checks WHERE scope={row}.scope;
            END''')
    db.execute('PRAGMA cache_size=-65536')


# Only code that can change merged publication from the same effective policy,
# saved evidence and observed Stash state belongs here. Runtime transport/CLI,
# VTT parsing and maintenance-only cleanup/recovery code have independent
# invalidation signals and must not force a library-wide re-merge.
PUBLICATION_CODE_FILES = (
    'cli.py', 'merge.py', 'merge_run.py', 'merge_parallel.py', 'merge_sync.py', 'separate_sync.py',
    'temporal.py', 'fusion.py', 'scene.py', 'occurrences.py', 'exclusions.py', 'cleanup.py',
    'taggers.py', 'onnx_tagger.py', 'vocabulary.py', 'inference.py', 'tags.py', 'domain.py', 'performance.py',
    'timeline.py', 'merge_sql.py', 'marker_groups.py',
    'thresholds.py', 'retention.py', 'scene_cache.py', 'progress.py', 'state.py', 'state_db.py',
    'state_results.py', 'state_thresholds.py', 'state_metadata.py', 'state_publications.py', 'config.py',
)


def publication_code():
    root = Path(__file__).resolve().parent
    return hashlib.sha256(b''.join((root / name).read_bytes() for name in PUBLICATION_CODE_FILES)).hexdigest()


def publication_key(app, args, hierarchy, cores, tags, stash):
    code = publication_code()
    names = ('APPLY_SCENE_TAGS', 'CREATE_MARKERS', 'CREATE_MISSING', 'MAX_TAGS',
             'MARKER_MIN_HITS', 'MARKER_MERGE_GAP_SECONDS', 'MARKER_MAX_TAGS_PER_FRAME',
             'MAX_MARKERS_PER_SCENE', 'MARKER_USE_END_TIME', 'MARKER_MATCH_TOLERANCE_SECONDS',
             'REPLACE_SCENE_TAGS', 'REPLACE_MARKERS')
    return digest(dict(code=code, policy=asdict(app.TEMPORAL_POLICY),
        settings={name: getattr(app, name) for name in names},
        blocklist=sorted(app.MARKER_TAG_BLOCKLIST),
        cores={s: sorted(v) for s, v in cores.items()}, tags={s: sorted(v) for s, v in tags.items()},
        hierarchy=asdict(hierarchy) if hierarchy else None, end_time=stash.has_end_time))


def scene_fingerprint(scene):
    def tag(value):
        return str(value['id']), value['name']
    markers = sorted((str(m['id']), m.get('title', ''), float(m['seconds']),
        None if m.get('end_seconds') is None else float(m['end_seconds']),
        tag(m['primary_tag']), sorted(tag(t) for t in m.get('tags', [])))
        for m in scene.get('scene_markers', []))
    return digest([str(scene['id']), sorted(tag(t) for t in scene.get('tags', [])), markers])


def checked(store, scene, policy, fingerprint):
    return check_reason(store, scene, policy, fingerprint) == 'reused'


def check_reason(store, scene, policy, fingerprint):
    from stash_ai.preview import check_source
    check_source(store)
    row = store.connection.execute('SELECT policy,fingerprint FROM merge_checks WHERE scope=? AND scene=?',
        (store.scope, str(scene))).fetchone()
    if row is None:
        return 'cache_miss'
    if row[0] != policy:
        return 'policy_changed'
    if row[1] != fingerprint:
        return 'stash_changed'
    return 'reused'


def record_check(store, scene, policy, fingerprint):
    """Caller owns the transaction, including publication promotion if needed."""
    store.connection.execute('INSERT INTO merge_checks VALUES (?,?,?,?) '
        'ON CONFLICT(scope,scene) DO UPDATE SET policy=excluded.policy,fingerprint=excluded.fingerprint',
        (store.scope, str(scene), policy, fingerprint))


def catalog_fingerprint(resolver):
    from stash_ai.marker_groups import PREFIX, ROOT
    return digest(sorted((name, str(t['id']), t.get('description'), t.get('ignore_auto_tag'), sorted(
        (str(p['id']), p['name'], p.get('description')) for p in t.get('parents', [])))
        for name, t in resolver.records.items() if t and (name.startswith(('ai:', PREFIX)) or name == ROOT)))


def projected_scene(scene, resolver, created, updates, deletes, final_tags):
    """Compute the expected fingerprint using native IDs returned on creation.

    Hook-side changes are detected by the next observed fingerprint. A legacy
    adapter that does not return created IDs falls back to a native readback.
    """
    if any(identifier is None for _, identifier in created):
        return None
    names = resolver.names_by_id
    def tag(identifier):
        identifier = str(identifier)
        if names.get(identifier) is None:
            raise KeyError(identifier)
        return {'id': identifier, 'name': names[identifier]}
    def marker(payload, identifier):
        return dict(id=str(identifier), title=payload['title'], seconds=payload['seconds'],
            end_seconds=payload.get('end_seconds'), primary_tag=tag(payload['primary_tag_id']),
            tags=[tag(i) for i in payload['tag_ids']])
    try:
        values = {str(m['id']): m for m in scene.get('scene_markers', []) if str(m['id']) not in deletes}
        for payload in updates:
            values[str(payload['id'])] = marker(payload, payload['id'])
        for payload, identifier in created:
            values[str(identifier)] = marker(payload, identifier)
        return dict(id=scene['id'], tags=[tag(i) for i in final_tags], scene_markers=list(values.values()))
    except KeyError:
        return None


def freeze(value):
    """Hashable legacy identity preserving Python's numeric and list equality."""
    if isinstance(value, dict):
        return tuple((key, freeze(item)) for key, item in sorted(value.items()))
    if isinstance(value, (list, tuple)):
        return tuple(freeze(item) for item in value)
    return value
