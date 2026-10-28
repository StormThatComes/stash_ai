"""Offline two-identity merge diagnostics and verified frozen-catalogue tuning."""
from contextlib import closing
import json
from pathlib import Path

from .evidence_pixels import alias_scope
from .clustering import _groups, matching_policy
from .config import plain
from .database import encoded
from .fragmentation_report import _open_readonly, _quality, _quality_summary
from .global_views import catalogue
from .identity_report import _select
from .merge_report_policy import analyse
from .models import namespace
from .pose_policy import anchor_sql
from .reference_compatibility import fallback_sql
from .reference_roles import strict_sql
from .reference_usage import ReferenceUsage
from .scanning import scan_identity
from .strict_partition import summary_views


def _identity(db, requested, person, group, config):
    invalid = db.execute('SELECT 1 FROM members m JOIN observations o ON o.id=m.observation '
        'JOIN scans s ON s.id=o.scan WHERE m.person=? AND (s.scope<>? OR s.namespace<>?) LIMIT 1',
        (person['id'], person['scope'], person['namespace'])).fetchone()
    if invalid:
        raise ValueError('Identity has incompatible retained membership evidence.')
    counts = dict(db.execute('SELECT COUNT(*) AS members,COUNT(*) FILTER (WHERE '+strict_sql()+') AS strict_authority,'
        'COUNT(*) FILTER (WHERE '+anchor_sql('o', config.data['clustering'])+') AS anchor_quality '
        'FROM members m JOIN observations o ON o.id=m.observation WHERE m.person=?', (person['id'],)).fetchone())
    scenes = [r[0] for r in db.execute('SELECT DISTINCT s.scene FROM members m '
        'JOIN observations o ON o.id=m.observation JOIN scans s ON s.id=o.scan '
        'WHERE m.person=? ORDER BY CAST(s.scene AS INTEGER),s.scene', (person['id'],))]
    from .body_reports import goal_rows
    bodies=goal_rows(db,config,person['id'])
    counts['person_members']=len(bodies)
    counts['person_anchors']=sum(r['sample_role']=='anchor' for r in bodies)
    scenes=sorted(set(scenes)|{r['scene'] for r in bodies},key=int)
    samples = []
    if group is not None:
        rows = {r['id']: r for r in db.execute('SELECT o.id,s.scene,o.cue,o.seconds,o.quality FROM observations o '
            'JOIN scans s ON s.id=o.scan WHERE o.id IN (SELECT value FROM json_each(?))',
            (encoded(group.sample_ids),))}
        samples = [dict(observation_id=i, scene_id=rows[i]['scene'], cue_index=rows[i]['cue'],
                        seconds=rows[i]['seconds'], quality=_quality(rows[i]['quality'])) for i in group.sample_ids]
    return dict(requested_identity_id=requested['id'], identity_id=person['id'],
        requested_performer_id=requested['performer'], performer_id=person['performer'], scenes=scenes, **counts,
        selected_strict_references=len(samples), reference_limit=config.data['clustering']['max_samples_per_person'],
        reference_samples=samples, reference_quality=_quality_summary([s['quality'] for s in samples]))


def _usage(config, groups):
    usage = ReferenceUsage(config.data['clustering']['max_reference_bytes'])
    for group in groups:
        usage.add(group.samples, people=1, kind='Report strict')
    return usage


def build(config, left_id, right_id, *, scope=None, by_performer=False, simulate=False):
    """Compare UUIDs or Stash IDs in one read-only snapshot without changing assignments."""
    model_namespace = namespace(config)
    with closing(_open_readonly(config.database)) as db, alias_scope(db):
        requested_left, left = _select(db, model_namespace,
            None if by_performer else left_id, left_id if by_performer else None, scope)
        requested_right, right = _select(db, model_namespace,
            None if by_performer else right_id, right_id if by_performer else None, scope)
        if left['scope'] != right['scope']:
            raise ValueError('Identity comparison requires the same Stash scope.')
        identities = (left['id'], right['id'])
        from .hybrid_evidence import report as hybrid_report
        hybrid = hybrid_report(db,config,left['scope'],model_namespace,focus=set(identities))
        from .body_global import report as person_report
        independent_person=person_report(db,config,left['scope'],model_namespace,focus=set(identities))
        result = dict(format=1, kind='recognition_identity_merge', scope=left['scope'],
            namespace=model_namespace, matching_policy=matching_policy(config),
            requested_identity_ids=[requested_left['id'], requested_right['id']], identity_ids=list(identities),
            requested_performer_ids=[requested_left['performer'], requested_right['performer']],
            performer_ids=[left['performer'], right['performer']],
            current_configuration={key: plain(config.data[key]) for key in ('detection', 'recognition', 'clustering')},
            paths=[], recommendations=[], hybrid=hybrid, independent_person=independent_person,
            analysis_basis='production_frozen_catalogue_counterfactual',
            guarantees_full_rebuild_merge=False,
            limitations=[
                'A verified proposal makes the production consolidation planner accept these current frozen '
                'references. It does not replay scene order, identity births, attachments or label reconciliation.',
                'A cold rebuild can change references, identities and competitors. No exact historic decision is reconstructed.',
                'All compatible strict identities in this scope remain competitors. Complete connected components '
                'are checked; recommendations can include additional identities, listed in merged_identity_ids.',
                'Proposals change global YAML settings, not just the requested pair. Scores are cosine similarities, not probabilities.',
                'At most four measured view pairs and two threshold strategies per route are searched. No verified '
                'proposal means this bounded search found none, not that all configurations are impossible.',
                'Rejected, superseded, incompatible and attachment-only members do not become strict references. '
                'No proposed detection/portrait change can manufacture missing evidence offline.',
            ])
        if left['id'] == right['id']:
            result.update(status='already_merged', message='Both IDs resolve to the same canonical identity; no YAML change is needed.')
            if simulate:
                from .replay_simulation import merge_impact
                result['simulation'] = merge_impact(config, result, db)
            return result
        grouping = config.data['clustering']
        groups = _groups(db, left['scope'], model_namespace, grouping,
            scan_policy=scan_identity(config, model_namespace),
            compatibility=fallback_sql(db, config, model_namespace))
        by_person = {g.identifier: g for g in groups}
        result['identities'] = [_identity(db, requested, person, by_person.get(person['id']), config)
                                for requested, person in ((requested_left, left), (requested_right, right))]
        result['catalogue'] = dict(strict_identities=len(groups), strict_references=sum(len(g.samples) for g in groups),
                                   reference_selection='production_quality_diversity_v1')
        summaries = summary_views(groups, _usage(config, groups))
        result['paths'].append(analyse(db, config, identities, 'strict_core', lambda _: summaries))
        # Catalogue formation depends on the thresholds being proposed, so each
        # trial must build fresh cohorts while retaining the same DB snapshot.
        def views(candidate):
            return catalogue(db, candidate, left['scope'], model_namespace,
                usage=_usage(candidate, groups), scan_policy=scan_identity(candidate, model_namespace), complete=True)
        result['paths'].append(analyse(db, config, identities, 'confirmed_views', views))
        if config.data['sapiensid']['enabled']:
            from .hybrid_report import analyse as hybrid_analyse
            result['paths'].append(hybrid_analyse(db,config,identities,left['scope'],model_namespace))
        from .body_identity import enabled as body_enabled
        if body_enabled(config):
            from .body_merge_report import analyse as body_analyse
            result['paths'].append(body_analyse(db,config,identities,left['scope'],model_namespace))
        for path in result['paths']:
            for item in path['recommendations']:
                component = item['verification']['shared_component']
                result['recommendations'].append(dict(item, route=path['route'],
                    merged_identity_ids=list(component['people']),
                    additional_identity_ids=[p for p in component['people'] if p not in identities],
                    next_step='Apply this YAML patch, then rebuild the retained scenes for both identities together.'))
        result['recommendations'].sort(key=lambda item: (
            len(item['additional_identity_ids']), len(item['changes']), item['route'], item['yaml']))
        unchanged = any(not item['changes'] for item in result['recommendations'])
        result['status'] = ('eligible_under_current_parameters' if unchanged else
                            'verified_parameter_changes' if result['recommendations'] else 'no_verified_proposal')
        result['next_action'] = dict(
            scene_ids=sorted({s for person in result['identities'] for s in person['scenes']}, key=lambda s: (int(s), s)),
            requires_identity_rebuild=True, portraits_only_is_sufficient=False)
        result['evidence_actions'] = []
        for person in result['identities']:
            if not person['selected_strict_references'] and not person['person_anchors']:
                result['evidence_actions'].append(dict(identity_id=person['identity_id'],
                    reason='no_current_strict_references',
                    action='Inspect anchor quality, latest scan compatibility and recorded assignment roles. '
                           'Re-run recognition if measurements are missing; auxiliary attachments cannot train strict references.'))
        if not grouping['anchor_view_matching']:
            result['evidence_actions'].append(dict(reason='consolidation_feature_disabled',
                action='Both face consolidation routes require clustering.anchor_view_matching=true '
                       'and an existing anchor pose or visibility restriction. Ordinary nomination merges are not simulated.'))
        if simulate:
            from .replay_simulation import merge_impact
            result['simulation'] = merge_impact(config, result, db)
        return result


def execute(args, config):
    by_performer = args.performer_id is not None
    identifiers = args.performer_id if by_performer else args.identity_id
    result = build(config, *identifiers, scope=args.scope, by_performer=by_performer,
                   simulate=getattr(args, 'simulate', False))
    output = json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + '\n'
    if args.output is None:
        print(output, end='')
    else:
        target = Path(args.output).expanduser().resolve()
        if target == config.database.resolve():
            raise ValueError('Report output must not overwrite the face database.')
        with target.open('x', encoding='utf-8') as file:
            file.write(output)
        print(f'Identity merge report written: {target} ({result["status"]})')
    return 0
