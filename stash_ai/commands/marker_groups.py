"""Repair marker group membership without replaying model evidence."""
from collections import Counter
from stash_ai.publication import markers as marker_ops
from stash_ai.publication.groups import for_resolver
from stash_ai.cleanup.groups import prune_groups


def run_backfill(app, args):
    """Repair group membership on existing owned markers, without source replay."""
    from tqdm import tqdm
    from stash_ai.evidence.labels import AI
    from stash_ai.storage.store import StateStore, scope_for
    from stash_ai.evidence.occurrences import OCCURRENCE_TAG
    stats, report = Counter(), args.report
    with StateStore(args.state_db, scope_for(app.STASH_URL), app.DRY_RUN) as store:
        if store.has_pending_publications():
            raise RuntimeError('An interrupted merge is pending; run --merge-only before synchronizing marker groups.')
        stash = app.create_stash_client(app.DRY_RUN)
        try:
            resolver = app.TagResolver(stash, source=AI, report=report.detail, create_missing=app.CREATE_MISSING)
            resolver.preload()
            groups = for_resolver(app, resolver)
            for row in tqdm(stash.iter_scenes(args.scene_id), desc='Marker groups', unit='scene'):
                try:
                    scene = stash.find_scene(str(row['id']))
                    updates = []
                    for marker in scene.get('scene_markers', []):
                        core = marker['primary_tag']['name']
                        if not marker_ops.owned_marker(marker, AI) or not core.startswith(AI.prefix):
                            continue
                        occurrence = any(t['name'] == OCCURRENCE_TAG for t in marker.get('tags', []))
                        desired = groups.resolve(core[len(AI.prefix):], occurrence)
                        tags = {str(t['id']) for t in marker.get('tags', []) if not groups.owns(t)} | {desired}
                        if tags == {str(t['id']) for t in marker.get('tags', [])}:
                            stats['markers_unchanged'] += 1
                            continue
                        payload = dict(id=str(marker['id']), scene_id=str(scene['id']), title=marker['title'],
                            seconds=marker['seconds'], primary_tag_id=str(marker['primary_tag']['id']), tag_ids=sorted(tags))
                        payload['end_seconds'] = marker.get('end_seconds')
                        updates.append(payload)
                    groups.record_preview(scene, [], updates, [])
                    for payload in updates:
                        if stash.dry_run:
                            report.detail(f"[DRY] Update marker group for marker {payload['id']} in scene {scene['id']}.")
                        else:
                            stash.update_marker(payload)
                    stats['markers_updated'] += len(updates)
                    stats['processed'] += 1
                except Exception as exc:
                    stats['errors'] += 1
                    report.issue('marker groups', '', row['id'], exc, app.safe_error(exc, api_key=app.STASH_API_KEY))
            if not stats['errors']:
                stats.update(prune_groups(stash, report.detail, usage_delta=groups.usage_delta))
        finally:
            stash.close()
    report.finish('Marker groups' + (' (planned)' if app.DRY_RUN else ''), stats)
    return int(bool(stats['errors']))
