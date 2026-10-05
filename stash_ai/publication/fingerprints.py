"""Publication code, policy, catalog and observed-scene fingerprints.

Track only code affecting merged publication from the same effective policy,
saved evidence and observed Stash state. Transport, runtime and maintenance
commands have independent invalidation signals.
"""
from dataclasses import asdict
import hashlib
from pathlib import Path
from stash_ai.signatures import digest


PUBLICATION_CODE_FILES = (
    'storage/metadata_store.py', 'storage/schema.py', 'rethreshold/plans.py',
    'evidence/contracts.py', 'evidence/formats.py', 'evidence/qualification.py', 'evidence/marker_identity.py', 'evidence/interval_geometry.py', 'inference/input_identity.py',
    'runtime/settings.py', 'runtime/context.py', 'runtime/workers.py',
    'cli.py', 'commands/bootstrap.py', 'evidence/pipeline.py', 'evidence/source_markers.py', 'evidence/identity.py', 'evidence/labels.py', 'inference/results.py', 'commands/hierarchy.py', 'commands/namespace_migration.py', 'runs/merged.py', 'runs/merged_model.py', 'publication/publisher.py', 'publication/parallel.py', 'publication/merged.py', 'publication/planning/types.py', 'publication/planning/merged_evidence.py', 'publication/planning/merged_tags.py', 'publication/planning/merged_markers.py', 'publication/index.py', 'publication/markers.py',
    'evidence/temporal/policy.py', 'evidence/temporal/statistics.py', 'evidence/temporal/markers.py', 'evidence/temporal/extras.py', 'evidence/intervals.py', 'evidence/observations.py', 'evidence/frames.py', 'evidence/shared_markers.py', 'evidence/scenes.py', 'evidence/occurrences.py', 'exclusions.py', 'cleanup/state.py', 'cleanup/client.py',
    'models.py', 'backends/loader.py', 'backends/onnx.py', 'backends/onnx_preprocessing.py', 'backends/onnx_session.py', 'backends/vocabulary.py', 'inference/prediction.py', 'inference/scene_predictions.py', 'inference/sprites.py', 'inference/preparation.py', 'stash_api/tags.py', 'domain.py', 'publication/fingerprints.py', 'storage/segments.py', 'storage/merge_checks.py',
    'evidence/timeline.py', 'stash_api/sql.py', 'publication/groups.py',
    'evidence/thresholds.py', 'inference/reuse.py', 'evidence/retention.py', 'evidence/cache_views.py', 'rethreshold/markers.py', 'rethreshold/scenes.py', 'storage/progress.py', 'inference/inputs.py', 'inference/identity.py', 'inference/checkpoints.py', 'signatures.py', 'storage/store.py', 'storage/database.py',
    'storage/archive.py', 'storage/snapshots.py', 'commands/point_migration.py', 'storage/results.py', 'storage/thresholds.py', 'storage/metadata.py', 'storage/publications.py', 'config.py',
)


def publication_code():
    root = Path(__file__).resolve().parents[1]
    return hashlib.sha256(b''.join((root / name).read_bytes() for name in PUBLICATION_CODE_FILES)).hexdigest()


def publication_key(app, args, hierarchy, cores, tags):
    code = publication_code()
    names = ('APPLY_SCENE_TAGS', 'CREATE_MARKERS', 'CREATE_MISSING', 'MAX_TAGS',
             'MARKER_MIN_HITS', 'MARKER_MERGE_GAP_SECONDS', 'MARKER_MAX_TAGS_PER_FRAME',
             'MAX_MARKERS_PER_SCENE', 'MARKER_MATCH_TOLERANCE_SECONDS')
    return digest(dict(code=code, policy=asdict(app.TEMPORAL_POLICY),
        settings={name: getattr(app, name) for name in names},
        cores={s: sorted(v) for s, v in cores.items()}, tags={s: sorted(v) for s, v in tags.items()},
        hierarchy=asdict(hierarchy) if hierarchy else None))


def scene_fingerprint(scene):
    def tag(value):
        return str(value['id']), value['name']
    markers = sorted((str(m['id']), m.get('title', ''), float(m['seconds']),
        None if m.get('end_seconds') is None else float(m['end_seconds']),
        tag(m['primary_tag']), sorted(tag(t) for t in m.get('tags', [])))
        for m in scene.get('scene_markers', []))
    return digest([str(scene['id']), sorted(tag(t) for t in scene.get('tags', [])), markers])


def catalog_fingerprint(resolver):
    from stash_ai.publication.groups import PREFIX, ROOT
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
