"""Reconcile archived proof against observed outputs before durable promotion."""
from copy import deepcopy
from stash_ai.publication import markers as marker_ops
from stash_ai.storage.recovery import publications
from stash_ai.evidence.labels import AI, source_codes
from stash_ai.evidence.intervals import union_intervals
from stash_ai.evidence.marker_identity import OCCURRENCE_TAG, is_occurrence
from stash_ai.evidence.shared_markers import prune_blocked_extras
from stash_ai.signatures import freeze


class ObservedContributions:
    def __init__(self):
        self.values, self.seen = [], set()

    def append(self, item):
        identity = freeze(item)
        if identity not in self.seen:
            self.seen.add(identity)
            self.values.append(item)


def reconcile_scene_snapshot(scene, candidates):
    """Keep observed associations and their last confirmed compatible proofs."""
    result, proofs = {}, ObservedContributions()
    current_ids = {str(t['id']) for t in scene.get('tags', [])}
    current_names = {t['name'] for t in scene.get('tags', [])}
    result['owned_ids'] = sorted(current_ids & {str(i) for s in candidates for i in s.get('owned_ids', [])})
    # Existing labels retain their last confirmed proofs. Interrupted plans
    # supply newly attached labels; never combine different inference runs.
    labels = set()
    for snapshot in candidates:
        selected = [i for i in snapshot.get('contributions', [])
                    if AI.prefix + i['tag'] in current_names and i['tag'] not in labels]
        for item in selected:
            proofs.append(item)
        labels.update(i['tag'] for i in selected)
    result['contributions'] = proofs.values
    return result


def reconcile_marker_snapshot(scene, candidates):
    """Reconstruct surviving marker proof without writes or inference."""
    proofs = ObservedContributions()
    versions = [union_intervals(s.get('contributions', []), borrow=True) for s in candidates]
    for marker in scene.get('scene_markers', []):
        if not marker_ops.owned_marker(marker, AI):
            continue
        occurrence = any(t['name'] == OCCURRENCE_TAG for t in marker.get('tags', []))
        present = {t['name'][len(AI.prefix):] for t in marker.get('tags', [])
                   if t['name'].startswith(AI.prefix) and t['name'] not in (AI.owner_tag, OCCURRENCE_TAG)}
        matches = [component for version in versions for component in version
            if marker['primary_tag']['name'] == AI.prefix + component['tag']
            and abs(round(component['start'], 3) - marker['seconds']) <= .0005
            and (marker.get('end_seconds') is None or (component['end'] is not None
                 and abs(round(component['end'], 3) - marker['end_seconds']) <= .0005))
            and is_occurrence(component) == occurrence]
        if not matches:
            raise RuntimeError(f"Cannot identify surviving marker {marker['id']} from saved publications. "
                               'Run --merge-only with the current YAML policies to reconcile this scene.')
        # Same time/core can occur in both plans with different extras.
        # Prefer the version matching what Stash actually received.
        def rank(component):
            title = AI.title_prefix + component['tag'] + ' [' + '+'.join(source_codes(component['sources'])) + ']'
            if is_occurrence(component):
                title += ' · wykrycie'
            return (title != marker['title'], len(present - set(component['extra_tags'])),
                    len(set(component['extra_tags']) - present))
        match = min(matches, key=rank)
        present_ids = {str(t['id']) for t in marker.get('tags', [])}
        for item in match['contributors']:
            extras = set(item.get('extra_tags', [])) | set(item.get('extra_support', {}))
            observed = prune_blocked_extras(deepcopy(item), extras - present)
            if 'extra_ids' in observed:
                observed['extra_ids'] = [i for i in observed['extra_ids'] if str(i) in present_ids]
            proofs.append(observed)

    return {'contributions': proofs.values}


def reconcile(store, scene, kind, published):
    """Observe survivors, then save the baseline even when cleanup is a no-op."""
    archived = publications(store, scene['id'], kind)
    if not archived and not (published or {}).get('recovery_required'):
        return published, False
    candidates = [*([published] if published is not None else []), *archived]
    if kind == 'scene':
        result = reconcile_scene_snapshot(scene, candidates)
    elif kind == 'markers':
        result = reconcile_marker_snapshot(scene, candidates)
    else:
        raise ValueError('Unknown recovery output kind.')
    # Dry-run writes only its temporary overlay; no Stash mutations happen here.
    store.publish(str(scene['id']), {kind: result})
    return result, True
