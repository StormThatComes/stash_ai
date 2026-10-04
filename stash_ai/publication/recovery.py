"""Reconcile archived proof against surviving Stash outputs before cleanup."""
from copy import deepcopy
from stash_ai.publication import markers as marker_ops
from stash_ai.storage.recovery import publications


def reconcile(app, stash, store, scene, kind, published):
    """Observe surviving outputs; never replay an abandoned plan or infer votes."""
    from stash_ai.evidence.labels import AI, source_codes
    from stash_ai.evidence.intervals import union_intervals
    from stash_ai.evidence.marker_identity import OCCURRENCE_TAG, is_occurrence
    from stash_ai.evidence.shared_markers import prune_blocked_extras
    from stash_ai.signatures import freeze
    archived = publications(store, scene['id'], kind)
    if not archived and not (published or {}).get('recovery_required'):
        return published, False
    candidates = [*([published] if published is not None else []), *archived]
    result, contributions, seen = {}, [], set()

    def append(item):
        identity = freeze(item)
        if identity not in seen:
            seen.add(identity)
            contributions.append(item)

    if kind == 'scene':
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
                append(item)
            labels.update(i['tag'] for i in selected)
    else:
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
                append(observed)
    result['contributions'] = contributions
    # Save this observed baseline even if the new cleanup has nothing to remove.
    # The dry-run StateStore writes only its temporary overlay.
    store.publish(str(scene['id']), {kind: result})
    return result, True

