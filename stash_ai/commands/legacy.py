"""Capture source-prefixed Stash objects before merged-output migration."""
from stash_ai.publication import markers as marker_ops
from collections import defaultdict

import stash_ai.evidence.labels as merge_labels
import stash_ai.inference.results as merge_results
from stash_ai.models import SOURCES


def legacy_label(name):
    for key, source in SOURCES.items():
        if name.startswith(source.prefix) and name != source.owner_tag:
            label = name[len(source.prefix):]
            if label and not label.startswith(("category:", "marker:")):
                return key, merge_labels.raw_label(label)
    return None


def legacy_objects(scene, kind):
    """Yield only source-owned input objects, preserving unknown scores and ends."""
    for item in scene.get('tags' if kind == 'scene' else 'scene_markers', []):
        name = item['name'] if kind == 'scene' else (item.get('primary_tag') or {}).get('name', '')
        parsed = legacy_label(name)
        if parsed and (kind == 'scene' or marker_ops.owned_marker(item, SOURCES[parsed[0]])):
            yield (*parsed, item)


def publication_state(store, scene_id, kind):
    """Retiring active cache does not retire its durable publication ownership."""
    from stash_ai.storage.recovery import publications
    return any(snapshot is not None for snapshot in store.snapshot(scene_id, kind, borrow=True)) or bool(
        publications(store, scene_id, kind))


def check_sources(app, store, scene):
    """Validate source ownership without importing anything during normal merge."""
    from stash_ai.signatures import freeze
    scene_id = str(scene['id'])
    for kind, enabled in (('scene', app.APPLY_SCENE_TAGS), ('markers', app.CREATE_MARKERS)):
        if not enabled:
            continue
        objects = list(legacy_objects(scene, kind))
        owned_ai = kind == 'markers' and any(
            marker_ops.owned_marker(marker, merge_labels.AI) for marker in scene.get('scene_markers', []))
        if not objects and not owned_ai:
            continue
        existing = store.sources(scene_id, kind)
        if owned_ai and not existing and not publication_state(store, scene_id, kind):
            raise RuntimeError('Merged markers exist without their source state. Restore the original --state-db before synchronization.')
        if objects:
            imported = {freeze(item) for item in store.imported_legacy(scene_id, kind, borrow=True)}
            if any(model not in existing or freeze(item) not in imported for model, _, item in objects):
                raise RuntimeError(f'{scene_id}: new source-prefixed {kind} objects require --import-legacy-only before merge.')


def import_scene(app, store, scene):
    """Commit evidence and its anti-duplication ledger atomically for one scene."""
    from collections import Counter
    from stash_ai.signatures import freeze
    scene_id, counts = str(scene['id']), Counter()
    with store.connection:
        if not store.connection.in_transaction:
            store.connection.execute('BEGIN')
        for kind, enabled in (('scene', app.APPLY_SCENE_TAGS), ('markers', app.CREATE_MARKERS)):
            if not enabled:
                continue
            existing = store.sources(scene_id, kind)
            if kind == 'markers' and not existing and not publication_state(store, scene_id, kind) and any(
                    marker_ops.owned_marker(marker, merge_labels.AI) for marker in scene.get('scene_markers', [])):
                raise RuntimeError('Merged markers exist without their source state. Restore the original --state-db before synchronization.')
            imported = {freeze(item) for item in store.imported_legacy(scene_id, kind, borrow=True)}
            originals, by_model = [], defaultdict(list)
            for model, label, item in legacy_objects(scene, kind):
                originals.append(item)
                if model in existing and freeze(item) in imported:
                    continue
                if kind == 'scene':
                    value = {'tag': label, 'score': None, 'legacy_tag_id': str(item['id'])}
                else:
                    extras, manual_ids = set(), set()
                    for tag in item.get('tags', []):
                        extra = legacy_label(tag['name'])
                        if extra:
                            extras.add(extra[1])
                        elif tag['name'] not in {source.owner_tag for source in SOURCES.values()}:
                            manual_ids.add(str(tag['id']))
                    value = dict(tag=label, start=item['seconds'], end=item.get('end_seconds'),
                        score=None, hits=None, extra_tags=sorted(extras - {label}), extra_ids=sorted(manual_ids),
                        legacy_marker_id=str(item['id']))
                by_model[model].append(value)
            for model, items in by_model.items():
                segment = {'mode': 'scene' if kind == 'scene' else 'spans', 'items': items,
                           'meta': {'origin': 'legacy', 'captured_at': merge_results.now(), 'revision': None}}
                store.results.append_legacy_in_transaction(scene_id, model, kind, segment)
                counts['legacy_scene_tags_imported' if kind == 'scene' else 'legacy_markers_imported'] += len(items)
            store.legacy.remember_in_transaction(scene_id, kind, originals)
    return counts


def run_only(app, args):
    """Read Stash objects into local evidence; never publish or load model assets."""
    from collections import Counter
    from tqdm import tqdm
    from stash_ai.storage.store import StateStore, scope_for
    stats, report = Counter(), args.report
    with StateStore(args.state_db, scope_for(app.STASH_URL), app.DRY_RUN,
                    cache_bytes=int(args.ram_cache_gb * 2**30),
                    memory_budget_bytes=int(args.ram_budget_gb * 2**30)) as store:
        store.report = report
        if store.has_pending_publications():
            raise RuntimeError('Pending publication plans exist; finish or abandon them before --import-legacy-only.')
        stash = app.services.create_stash_client(app.DRY_RUN)
        stash.report = report
        try:
            stash.create_markers = app.CREATE_MARKERS
            stash.configure_merge_reads(report, merge_only=True)
            for scene in tqdm(stash.iter_scenes(args.scene_id), desc='Import legacy', unit='scene'):
                try:
                    with report.measure('legacy_import'):
                        stats.update(import_scene(app, store, scene))
                    stats['processed'] += 1
                except Exception as exc:
                    stats['errors'] += 1
                    report.issue('legacy import', '', scene['id'], exc,
                                 app.services.safe_error(exc, api_key=app.STASH_API_KEY))
        finally:
            stash.close()
        report.finish('Legacy import' + (' (planned)' if app.DRY_RUN else ''), stats, store.cache)
    return int(bool(stats['errors']))
