"""One-time conversion of source-owned Stash outputs using fresh inference."""
from stash_ai.tools.legacy_namespaces import NAMESPACES
from stash_ai.evidence.labels import AI, raw_label
from stash_ai.publication.markers import owned_marker, marker_differs
from stash_ai.publication.index import MarkerIntervalIndex
from stash_ai.publication.publisher import MergePublisher
from stash_ai.runs.merged import MergedRun


def source_label(name):
    for model, source in NAMESPACES.items():
        if name.startswith(source.prefix) and name != source.owner_tag:
            label = name[len(source.prefix):]
            if label and not label.startswith(('category:', 'marker:')):
                return model, raw_label(label)
    return None


def source_objects(scene, kind):
    for item in scene.get('tags' if kind == 'scene' else 'scene_markers', []):
        parsed = source_label(item['name'] if kind == 'scene' else item['primary_tag']['name'])
        if parsed and (kind == 'scene' or owned_marker(item, NAMESPACES[parsed[0]])):
            yield *parsed, item


def extend_plan(app, scene, resolved, catalog, plan):
    """Keep manual objects/extras and reuse source marker IDs where ranges overlap."""
    scene_labels = {hit['tag'] for hit in plan.snapshots.get('scene', {}).get('contributions', [])}
    removed = {str(item['id']) for _, label, item in source_objects(scene, 'scene')
               if label not in scene_labels or catalog.ids.get(AI.prefix + label)} if app.APPLY_SCENE_TAGS else set()
    plan.final_scene_tag_ids.difference_update(removed)
    old = [(label, marker) for _, label, marker in source_objects(scene, 'markers')] if app.CREATE_MARKERS else []
    index = MarkerIntervalIndex([dict(marker, primary_tag={'name': label, 'id': label}) for label, marker in old], by_name=True)
    available = {str(marker['id']) for _, marker in old}
    owners = {source.owner_tag for source in NAMESPACES.values()}
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


def sync_migrated(app, stash, resolver, store, scene, scene_hits, spans, *, report, tag_policies, publication_key):
    from stash_ai.publication.merged import (prepare_sync_evidence, resolve_sync_tags, resolve_marker_catalog,
        plan_scene_tags, plan_markers, sync_statistics, publish_sync_plan)
    scene_id = str(scene['id'])
    evidence = prepare_sync_evidence(app, stash, store, scene, scene_hits, spans, cleanup=False, report=report)
    resolved = resolve_sync_tags(app, stash, resolver, scene_id, evidence, tag_policies, report)
    catalog = resolve_marker_catalog(resolver, resolved)
    scene = resolved.scene
    plan = plan_scene_tags(app, store, scene_id, scene, evidence.scene_hits, evidence.scene_labels,
                           resolved.ids, resolved.blocked_names)
    plan_markers(app, store, resolved, catalog, plan, report=report)
    migrated = extend_plan(app, scene, resolved, catalog, plan)
    if app.CREATE_MARKERS:
        plan.snapshots['markers'] = {'contributions': resolved.spans}
    stats = sync_statistics(app, plan, resolved.markers, resolved.suppressed)
    stats.update(migrated)
    return publish_sync_plan(app, stash, resolver, store, scene_id, scene, resolved.markers,
        evidence.scene_labels, plan, stats, report=report, publication_key=publication_key)


class MigrationPublisher(MergePublisher):
    def _sync(self, current, scene_hits, spans):
        return sync_migrated(self.app, self.stash, self.resolver, self.store, current, scene_hits, spans,
            report=self.report, tag_policies=self.scene_policies, publication_key=self.publication_key)


class MigrationRun(MergedRun):
    def _scan_scenes(self):
        from tqdm import tqdm
        from stash_ai.publication.fingerprints import scene_fingerprint
        from stash_ai.publication.preconditions import require_source_state
        from stash_ai.storage.progress import output_kinds
        for current in tqdm(self.stash.iter_scenes(self.args.scene_id), desc='Read legacy outputs', unit='scene'):
            scene_id = str(current['id'])
            try:
                if not (any(any(source_objects(current, kind)) for kind in output_kinds(self.app))
                        or self.store.has_pending_publication(scene_id)):
                    continue
                require_source_state(self.app, self.store, current)
                self.scenes[scene_id] = dict(id=scene_id, paths=current.get('paths') or {},
                                            fingerprint=scene_fingerprint(current), publication_reason='forced')
            except Exception as exc:
                self.stats['errors'] += 1
                self.report.issue('read legacy outputs', '', scene_id, exc,
                    self.app.services.safe_error(exc, api_key=self.app.STASH_API_KEY))

    def _merge_scenes(self):
        MigrationPublisher(self.app, self.args, self.store, self.stash, self.resolver,
            self.core_policies, self.scene_policies, publication_key=self.merge_key,
            stats=self.stats, report=self.report).run(self.scenes)


def main(argv=None, *, runtime=None):
    from stash_ai.runtime.context import RuntimeContext
    from stash_ai.runtime.diagnostics import MergeDiagnostics
    from stash_ai.commands.arguments import build_parser
    from stash_ai.commands.validation import validate_args
    from stash_ai.commands.modes import Command
    from stash_ai.commands import bootstrap
    app = runtime if runtime is not None else RuntimeContext()
    parser = build_parser(app)
    parser.description = 'Convert source-owned jt:/wd:/dtq:/op: outputs to AI using fresh inference.'
    parser.set_defaults(resume=False, restart=True)
    args = validate_args(app, parser, parser.parse_args(argv))
    if args.command is not Command.INFERENCE or args.resume or args.abandon_pending:
        parser.error('Namespace migration requires fresh inference; command modes, resume and abandonment do not apply.')
    bootstrap.runtime_options(app, args)
    args.force_merge = True
    settings = bootstrap.selected_settings(app, args)
    cores, tags = bootstrap.policies(app, args, hierarchy_only=False, cleanup_only=False)
    catalogs, hierarchy = bootstrap.catalogs(app, settings, args, tags)
    with MergeDiagnostics(args.report, args.profile_merge, dry_run=app.DRY_RUN, merge_only=False, force=True):
        return MigrationRun(app, settings, args, catalogs, hierarchy, cores, tags).run()


if __name__ == '__main__':
    from ._entrypoint import run as entrypoint
    raise SystemExit(entrypoint(main))
