"""One-time Stash migration of ordinary AI points to native marker ranges."""
from collections import Counter
import math

from stash_ai.evidence.labels import AI
from stash_ai.evidence.marker_identity import OCCURRENCE_TAG
from stash_ai.publication.markers import owned_marker


def ordinary_points(scene):
    return [marker for marker in scene.get('scene_markers', [])
            if owned_marker(marker, AI) and marker.get('end_seconds') is None
            and not any(tag['name'] == OCCURRENCE_TAG for tag in marker.get('tags', []))]


def require_native_ranges(scene, spans):
    if ordinary_points(scene):
        raise RuntimeError(f"Scene {scene['id']} has historical AI point markers; "
                           'run --migrate-point-markers-only before merging or cleaning ranges.')


def plan(store, scene):
    """Resolve all extents before writing; snapshots remain the durable retry proof."""
    from stash_ai.evidence.interval_geometry import union_intervals
    from stash_ai.storage.recovery import publications
    points = ordinary_points(scene)
    if not points:
        return []
    candidates = [snapshot for snapshot in (*store.snapshot(scene['id'], 'markers', borrow=True),
                  *publications(store, scene['id'], 'markers')) if snapshot is not None]
    components = [component for snapshot in candidates
                  for component in union_intervals(snapshot['contributions'], borrow=True)]
    ranges = [component for component in components if component['end'] is not None]
    updates = []
    for marker in points:
        ends = [component['end'] for component in ranges
                if AI.prefix + component['tag'] == marker['primary_tag']['name']
                and abs(round(component['start'], 3) - marker['seconds']) <= .0005]
        if not ends:
            raise RuntimeError(f"No saved extent for marker {marker['id']} in scene {scene['id']}; "
                               'restore its publication snapshot before migrating.')
        end = round(max(ends), 3)
        if not math.isfinite(end) or end <= marker['seconds']:
            raise ValueError(f"Invalid saved extent for marker {marker['id']}.")
        updates.append(dict(id=str(marker['id']), scene_id=str(scene['id']), title=marker['title'],
                            seconds=marker['seconds'], end_seconds=end,
                            primary_tag_id=str(marker['primary_tag']['id']),
                            tag_ids=[str(tag['id']) for tag in marker.get('tags', [])]))
    return updates


def run(app, args):
    from tqdm import tqdm
    from stash_ai.storage.store import StateStore, scope_for
    stats, report = Counter(), args.report
    with StateStore(args.state_db, scope_for(app.STASH_URL), app.DRY_RUN, allow_cleanup=True) as store:
        stash = app.services.create_stash_client(app.DRY_RUN)
        try:
            for row in tqdm(stash.iter_scenes(args.scene_id), desc='Migrate AI point markers', unit='scene'):
                try:
                    scene = stash.find_scene(str(row['id']))
                    updates = plan(store, scene)
                    for payload in updates:
                        if app.DRY_RUN:
                            report.detail(f"[DRY] Marker {payload['id']}: end_seconds={payload['end_seconds']}.")
                        else:
                            stash.update_marker(payload)
                        stats['markers_migrated'] += 1
                    stats['processed'] += 1
                except Exception as exc:
                    stats['errors'] += 1
                    report.issue('point migration', '', row['id'], exc,
                                 app.services.safe_error(exc, api_key=app.STASH_API_KEY))
        finally:
            stash.close()
    report.finish('Point marker migration' + (' (planned)' if app.DRY_RUN else ''), stats)
    return int(bool(stats['errors']))
