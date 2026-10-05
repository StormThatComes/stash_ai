"""One-time conversion of source-owned Stash outputs using fresh inference."""
from stash_ai.models import SOURCES
from stash_ai.evidence.labels import AI, raw_label
from stash_ai.publication.markers import owned_marker, marker_differs
from stash_ai.publication.index import MarkerIntervalIndex


def source_label(name):
    for model, source in SOURCES.items():
        if name.startswith(source.prefix) and name != source.owner_tag:
            label = name[len(source.prefix):]
            if label and not label.startswith(('category:', 'marker:')):
                return model, raw_label(label)
    return None


def source_objects(scene, kind):
    for item in scene.get('tags' if kind == 'scene' else 'scene_markers', []):
        parsed = source_label(item['name'] if kind == 'scene' else item['primary_tag']['name'])
        if parsed and (kind == 'scene' or owned_marker(item, SOURCES[parsed[0]])):
            yield *parsed, item


def check_sources(app, store, scene, *, transition=False):
    from stash_ai.storage.recovery import publications
    for kind, enabled in (('scene', app.APPLY_SCENE_TAGS), ('markers', app.CREATE_MARKERS)):
        if not enabled:
            continue
        if not transition and any(source_objects(scene, kind)):
            raise RuntimeError(f"{scene['id']}: source-prefixed {kind} outputs require --migrate-legacy-only with fresh inference.")
        if kind == 'markers' and any(owned_marker(marker, AI) for marker in scene.get('scene_markers', [])):
            known = (store.sources(scene['id'], kind) or
                     any(value is not None for value in store.snapshot(scene['id'], kind, borrow=True)) or
                     publications(store, scene['id'], kind))
            if not known:
                raise RuntimeError('Merged markers exist without their source state. Restore the original --state-db before synchronization.')


def extend_plan(app, scene, resolved, catalog, plan):
    """Keep manual objects/extras and reuse source marker IDs where ranges overlap."""
    scene_labels = {hit['tag'] for hit in plan.snapshots.get('scene', {}).get('contributions', [])}
    removed = {str(item['id']) for _, label, item in source_objects(scene, 'scene')
               if label not in scene_labels or catalog.ids.get(AI.prefix + label)} if app.APPLY_SCENE_TAGS else set()
    plan.final_scene_tag_ids.difference_update(removed)
    old = [(label, marker) for _, label, marker in source_objects(scene, 'markers')] if app.CREATE_MARKERS else []
    index = MarkerIntervalIndex([dict(marker, primary_tag={'name': label, 'id': label}) for label, marker in old], by_name=True)
    available = {str(marker['id']) for _, marker in old}
    owners = {source.owner_tag for source in SOURCES.values()}
    labels = {str(identifier): name[len(AI.prefix):] for name, identifier in catalog.ids.items()
              if identifier and name.startswith(AI.prefix)}

    def key(payload):
        return payload['primary_tag_id'], payload['seconds'], payload.get('end_seconds')

    for wanted, interval in plan.wanted_markers:
        label = labels[wanted['primary_tag_id']]
        related = index.overlapping(label, interval, app.MARKER_MATCH_TOLERANCE_SECONDS)
        extras = {str(tag['id']) for marker in related for tag in marker.get('tags', [])
                  if tag['name'] not in owners and not source_label(tag['name'])
                  and not tag['name'].startswith(AI.prefix) and tag['name'] not in resolved.blocked_names
                  and str(tag['id']) not in catalog.managed_group_ids}
        created = next((payload for payload in plan.creates if key(payload) == key(wanted)), None)
        updated = next((payload for payload in plan.updates if key(payload) == key(wanted)), None)
        if created is not None:
            matches = [marker for marker in related if str(marker['id']) in available]
            if matches:
                current = min(matches, key=lambda marker: (abs(marker['seconds'] - wanted['seconds']), str(marker['id'])))
                available.remove(str(current['id']))
                plan.creates.remove(created)
                updated = {**created, 'id': str(current['id'])}
                plan.updates.append(updated)
            else:
                created['tag_ids'] = sorted(set(created['tag_ids']) | extras)
        if updated is not None:
            updated['tag_ids'] = sorted(set(updated['tag_ids']) | extras)
        elif created is None and extras:
            current = next(marker for marker in scene.get('scene_markers', [])
                           if owned_marker(marker, AI) and not marker_differs(marker, wanted))
            changed = {**wanted, 'id': str(current['id']), 'tag_ids': sorted(set(wanted['tag_ids']) | extras)}
            if marker_differs(current, changed):
                plan.updates.append(changed)
                plan.unchanged -= 1
    predicted = {marker['tag'] for marker in resolved.markers}
    plan.deletes.update(str(marker['id']) for label, marker in old
                        if str(marker['id']) in available and (label not in predicted or catalog.ids.get(AI.prefix + label)))
    return {'source_scene_tags_migrated': len(removed), 'source_markers_migrated': len(old) - len(available - plan.deletes)}
