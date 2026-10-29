"""Explicit merge goals resolved against one retained read-only snapshot."""
from .identity_report import _select
from .models import namespace
from .merge_goal_index import evaluation_scope


def resolve(db, config, sets, *, scope=None, by_performer=True, policy='identity', scene_exclusions=()):
    from .merge_goal_identity import POLICY, freeze
    from .merge_goal_sets import freeze_set
    from .reference_roles import strict_sql
    from .pose_policy import is_anchor
    from .merge_goal_samples import freeze as freeze_samples
    from .person_signal import contract
    if policy not in ('identity', 'strict'):
        raise ValueError('Goal policy must be identity or strict.')
    if not sets or any(not group for group in sets):
        raise ValueError('Merge goals require nonempty sets.')
    groups, owners = [], {}
    diagnostics = dict(format=1, roles='input_reference_eligibility',
                       sapiensid_enabled=config.data['sapiensid']['enabled'], sapiensid_contract=contract(config))
    for index, identifiers in enumerate(sets):
        members = {}
        for identifier in identifiers:
            requested, person = _select(db, namespace(config),
                None if by_performer else identifier, identifier if by_performer else None, scope)
            if scope is None:
                scope = person['scope']
            if person['scope'] != scope:
                raise ValueError('Merge goals require the same scope.')
            identity = person['id']
            if identity in owners and owners[identity] != index:
                raise ValueError(f'Overlapping sets after alias resolution: {identifier} -> {identity}')
            owners[identity] = index
            item = members.setdefault(identity, dict(identity_id=identity, performer_id=person['performer'],
                requested_ids=[], observations=[]))
            item['requested_ids'].append(identifier)
        groups.append(dict(index=index, members=list(members.values())))
    from .merge_scene_exclusions import resolve as resolve_exclusions, freeze_targets
    exclusions = resolve_exclusions(db, config, scene_exclusions, scope, owners)
    for group in groups:
        forbidden = {e['scene'] for e in exclusions if e['set'] == group['index']}
        set_rows = []
        kept, excluded = [], []
        for item in group['members']:
            identity = item['identity_id']
            rows = [dict(r) for r in db.execute(
                'SELECT o.id,s.id AS scan,s.scene,o.cue,o.seconds,o.quality,o.rejected,s.status, '
                +strict_sql()+' AS reference_authority FROM members m JOIN observations o ON o.id=m.observation '
                'JOIN scans s ON s.id=o.scan WHERE m.person=? ORDER BY o.id', (identity,))]
            for row in rows:
                row['goal_role'] = ('reference' if row['reference_authority'] and row['status'] == 'complete'
                    and row['rejected'] is None and is_anchor(row['quality'], config.data['clustering']) else 'uncertain')
            freeze_samples(db, config, rows, diagnostics)
            for row in rows:row['evidence_kind']='face'
            from .body_reports import goal_rows
            rows.extend(goal_rows(db,config,identity))
            removed = [r for r in rows if r['scene'] in forbidden]
            if removed:
                item['excluded_observations'] = [dict(id=r['id'], scene=r['scene']) for r in removed]
                rows = [r for r in rows if r['scene'] not in forbidden]
                if not rows:
                    excluded.append(item)
                    continue
            kept.append(item)
            item['observations'] = [{k:r[k] for k in ('id','scan','scene','cue','goal_role','sample_role','body_state','evidence_kind')} for r in rows]
            if policy == 'identity':
                item['identity_evidence'] = freeze(rows)
            set_rows.extend(rows)
        group['members'] = kept
        if excluded:
            group['excluded_members'] = excluded
        if policy == 'identity':
            group['identity_evidence'] = freeze_set(set_rows)
        else:
            # Diagnostic physical continuity, independent of strict all-sample acceptance.
            group['construction_evidence'] = freeze_set(set_rows)
    result = dict(format=3, kind='recognition_merge_goals', scope=scope, namespace=namespace(config),
                selector='performer' if by_performer else 'identity', sample_diagnostics=diagnostics,
                goal_policy=dict(POLICY) if policy == 'identity' else dict(name='strict', version=1),
                groups=groups, status='goals_resolved', recommendations=[])
    if exclusions:
        result['scene_exclusions'] = exclusions
        freeze_targets(db, config, result)
    return result


def execute(args, config):
    from contextlib import closing
    from .fragmentation_report import _open_readonly
    checkpoint=getattr(args,'checkpoint',None)
    resume=getattr(args,'resume',False)
    yaml_mode=getattr(args,'yaml_mode','full')
    if checkpoint and args.yaml_output is None:
        args.yaml_output=checkpoint.with_name(checkpoint.name+'.best.yml')
    targets = [p.expanduser().resolve() for p in (args.output, args.yaml_output) if p]
    protected={config.source_path.resolve(),config.database.resolve()}
    if checkpoint:targets.append(checkpoint.expanduser().resolve())
    allowed={p.expanduser().resolve() for p in (args.yaml_output,checkpoint) if p} if resume else set()
    if len(set(targets)) != len(targets) or any(p in protected or (p.exists() and p not in allowed) for p in targets):
        raise FileExistsError('Outputs must be distinct new files.')
    from .merge_solver_checkpoint import winner_file
    writer=winner_file(args.yaml_output,resume=resume) if args.solve else None
    sets = args.performer_set or [args.performer_id or args.identity_id]
    with closing(_open_readonly(config.database)) as db:
        result = resolve(db, config, sets, scope=args.scope, by_performer=args.identity_id is None,
                         policy=getattr(args, 'goal_policy', 'identity'),
                         scene_exclusions=getattr(args, 'exclude_performer_scene', None) or ())
        result['snapshot_sha256'] = fingerprint(db, config, result['scope'])
        if args.solve:
            from .merge_solver import solve
            from .merge_goal_output import group_summary_lines
            import json
            import sys
            def progress(name, stage, item):
                if args.progress_style != 'off':
                    detail = ''
                    if stage == 'start':
                        detail = (' (input configuration)' if item['previous_trial'] is None else
                                  f" changes_vs={item['previous_trial']} parent={item['search_parent']}")
                    elif stage == 'complete':
                        detail = (f" disjoint={item['goal_result']['disjoint']} "
                                  f"best={item['best_trial']} best_number={item['best_trial_number']}")
                        if item['goal_result'].get('goal_policy', {}).get('name') == 'identity':
                            detail += f" goal_policy=identity strict_disjoint={item['goal_result']['strict_disjoint']}"
                    print(f'Merge solver {name}: {stage}{detail}', file=sys.stderr, flush=True)
                    if stage == 'start':
                        for change in item['changes_from_previous']:
                            print(f"  {change['parameter']}: {json.dumps(change['previous'])} -> "
                                  f"{json.dumps(change['current'])}", file=sys.stderr, flush=True)
                        if item.get('search_phase') == 'bidirectional':
                            print(f"  Search: phase=bidirectional strategy={item['search_strategy']}",
                                  file=sys.stderr, flush=True)
                        if item.get('search_phase') == 'feasibility':
                            print(f"  Search: phase=feasibility profile={item['search_strategy']}", file=sys.stderr, flush=True)
                    elif stage == 'complete':
                        for group in item['goal_result']['groups']:
                            print('\n'.join(group_summary_lines(group)), file=sys.stderr, flush=True)
                        from .merge_scene_exclusions import summary_lines
                        for line in summary_lines(item['goal_result']):
                            print(line, file=sys.stderr, flush=True)
                        construction = item['goal_result'].get('construction')
                        if construction:
                            merged = [g['index']+1 for g in construction['groups'] if g['merged']]
                            print(f"  Construction: {construction['status']} merged_sets={merged} "
                                  f"goals_satisfied={construction['goals_satisfied']}", file=sys.stderr, flush=True)
                    elif stage == 'failed':
                        print(f"  Feasibility probe incomplete: {item['reason']}; {item['error']}", file=sys.stderr, flush=True)
            result.update(solve(config, result, db, maximum=args.max_trials, progress=progress,
                                checkpoint=checkpoint,resume=resume,winner_writer=writer,
                                sampler=getattr(args,'solver_sampler','heuristic'),seed=getattr(args,'solver_seed',0),
                                yaml_mode=yaml_mode, feasibility_trials=getattr(args, 'feasibility_trials', 6), outside_limits={
                                    key: getattr(args, attr, None) for key, attr in (
                                        ('coverage_lost', 'max_outside_coverage_loss'),
                                        ('split_pairs', 'max_outside_split_pairs'),
                                        ('newly_joined_pairs', 'max_outside_joined_pairs'),
                                        ('new_outsider_observations', 'max_new_outsider_observations'))}))
        elif args.simulate:
            result['simulation'] = evaluate(config, result, db)
            result['status'] = 'baseline_evaluated'
    from .merge_goal_output import export
    export(config, result, args.output, args.yaml_output,yaml_owner=writer,yaml_mode=yaml_mode)
    if args.solve and args.progress_style != 'off':
        from .merge_goal_output import summary
        print(summary(result), file=sys.stderr, flush=True)
    return 130 if result.get('interrupted') else 2 if result.get('status') == 'no_compatible_scans' else 0


@evaluation_scope
def assess(goals, observations, retained, predicted):
    """Evaluate final memberships, including partial losses and outsider bridges."""
    from collections import Counter
    groups, owners = [], {}
    goal_ids = {r['id'] for g in goals['groups'] for m in g['members'] for r in m['observations']}
    for group in goals['groups']:
        members, counts, complete_owners = [], Counter(), Counter()
        destination_fractions = Counter()
        destination_squares = Counter()
        missing = lost = 0
        for member in group['members']:
            ids = {r['id'] for r in member['observations']}
            selected = ids & observations.keys()
            absent = len(ids - selected)
            withheld = sum(predicted.get(k) is None for k in selected)
            destinations = Counter(predicted[k] for k in selected if predicted.get(k) is not None)
            counts.update(destinations)
            assigned = sum(destinations.values())
            if ids:
                for destination, count in destinations.items():
                    fraction = count / len(ids)
                    destination_fractions[destination] += fraction
                    destination_squares[destination] += fraction*fraction
            complete = bool(ids) and not absent and not withheld and len(destinations) == 1
            if complete:
                complete_owners.update(destinations.keys())
            missing += absent
            lost += withheld
            members.append(dict(identity_id=member['identity_id'], performer_id=member['performer_id'],
                observations=len(ids), missing=absent, withheld=withheld,
                assigned=assigned, assigned_fraction=assigned / len(ids) if ids else 0.,
                resulting_identities=dict(destinations), fully_assigned_to_one=complete))
        for destination in counts:
            owners.setdefault(destination, []).append(group['index'])
        outsiders = {k for k,v in predicted.items() if v is not None and v in counts and k not in goal_ids}
        largest = max(complete_owners.values(), default=0)
        groups.append(dict(index=group['index'], members=members, missing=missing, withheld=lost,
            outsider_observations=len(outsiders), additional_retained_identities=sorted({retained[k] for k in outsiders if retained.get(k) is not None}),
            resulting_identities=dict(counts), fragments=len(counts),
            largest_complete_performers=largest, total_performers=len(members),
            complete_fraction=largest/len(members),
            observations=sum(m['observations'] for m in members), assigned=sum(m['assigned'] for m in members),
            assigned_fraction=sum(m['assigned_fraction'] for m in members)/len(members),
            consolidation_fraction=max(destination_fractions.values(), default=0.)/len(members),
            pair_consolidation_fraction=(sum(v*v-destination_squares[k] for k,v in destination_fractions.items())
                /(len(members)*(len(members)-1)) if len(members)>1 else (1. if largest else 0.)),
            complete_components=sorted(complete_owners.values(),reverse=True),
            fully_merged=largest == len(members) and not missing and not lost))
    by_index = {g['index']: g for g in groups}
    conflicts = []
    for identity, sets in sorted(owners.items()):
        if len(sets) < 2:
            continue
        counts = [by_index[index]['resulting_identities'][identity] for index in sets]
        pairs = (sum(counts)**2 - sum(n*n for n in counts))//2
        conflicts.append(dict(identity=identity, sets=sets, observation_pairs=pairs))
    covered = all(g['members'] and all(m['observations'] for m in g['members']) and
                  not g['missing'] and not g['withheld'] for g in groups)
    result = dict(groups=groups, conflicts=conflicts,
                cross_set_observation_pairs=sum(c['observation_pairs'] for c in conflicts),
                disjoint=not conflicts, coverage_preserved=covered,
                admissible=covered and not conflicts,
                fully_solved=covered and not conflicts and all(g['fully_merged'] for g in groups))
    if goals.get('goal_policy', {}).get('name') == 'identity':
        from .merge_goal_identity import apply
        apply(goals, result, observations, predicted)
    from .merge_goal_samples import attach
    from .merge_scene_exclusions import apply as apply_exclusions
    return apply_exclusions(goals, attach(goals, result, observations, predicted), observations, predicted)


def evaluate(config, goals, db, variants=None):
    from .replay_simulation import replay
    return replay(config, variants or {}, scope=goals['scope'], source=db,
        consume=lambda name, observations, retained, predicted:
            dict(goal_result=assess(goals, observations, retained, predicted)))


def objective(status):
    """Fair lexicographic objective; larger sets cannot hide an untouched smaller set."""
    if 'outside_preserved' in status:
        unguarded = {k: v for k, v in status.items() if k != 'outside_preserved'}
        return (status['outside_preserved'],) + objective(unguarded)
    if 'scene_exclusions_satisfied' in status:
        unguarded = {k: v for k, v in status.items() if k != 'scene_exclusions_satisfied'}
        # Positive coverage is still required for admissibility. Among failed
        # trials reward repairing more explicit negatives before consolidation.
        return (status['admissible'], status['scene_exclusions_satisfied'],
                -sum(not r['satisfied'] for r in status['scene_exclusions']),
                -sum(r['matching_count'] for r in status['scene_exclusions'])) + objective(unguarded)
    if status.get('goal_policy', {}).get('name') == 'identity':
        from .merge_goal_identity import objective as identity_objective
        return identity_objective(status)
    groups = status['groups']
    return (status['admissible'], status['fully_solved'],
            status['disjoint'],
            min(g['assigned_fraction'] for g in groups),
            sum(g['assigned_fraction'] for g in groups)/len(groups),
            -status['cross_set_observation_pairs'],
            min(g['complete_fraction'] for g in groups),
            sum(g['largest_complete_performers'] for g in groups),
            min(g.get('pair_consolidation_fraction',0.) for g in groups),
            sum(g.get('pair_consolidation_fraction',0.) for g in groups)/len(groups),
            min(g['consolidation_fraction'] for g in groups),
            sum(g['consolidation_fraction'] for g in groups)/len(groups),
            -sum(g['fragments'] for g in groups))


def unmet_conditions(status):
    """Every unsatisfied goal, including empty members and cross-set merges."""
    if status.get('goal_policy', {}).get('name') == 'identity':
        from .merge_goal_identity import unmet
        return unmet(status) + status.get('outside_violations', []) + status.get('scene_exclusion_violations', [])
    reasons = list(status.get('outside_violations', [])) + status.get('scene_exclusion_violations', [])
    for group in status.get('groups', []):
        for member in group['members']:
            base = dict(set=group['index'], identity=member['identity_id'])
            if not member['observations']:
                reasons.append(dict(base, reason='empty_identity'))
            for key, reason in (('missing', 'missing_latest_observations'), ('withheld', 'withheld_observations')):
                if member[key]:
                    reasons.append(dict(base, reason=reason, count=member[key]))
            if len(member['resulting_identities']) > 1:
                reasons.append(dict(base, reason='performer_split', fragments=len(member['resulting_identities'])))
        if not group['fully_merged']:
            reasons.append(dict(set=group['index'], reason='incomplete_set', fragments=group['fragments'],
                complete=group['largest_complete_performers'], total=group['total_performers']))
    reasons.extend(dict(reason='cross_set_merge', **conflict) for conflict in status.get('conflicts', []))
    return reasons


def fingerprint(db, config, scope):
    """Hash retained numerical inputs and bindings without materializing a second corpus."""
    import hashlib
    from .config import digest
    from .scan_heads import predicate as current_scan
    value = hashlib.sha256()
    for sql in (
        'SELECT s.* FROM scans s WHERE s.scope=? AND s.namespace=? ORDER BY s.id',
        'SELECT s.id FROM scans s WHERE s.scope=? AND s.namespace=? AND '+current_scan()+' ORDER BY s.id',
        'SELECT o.id,o.scan,o.quality,o.rejected,o.embedding,m.person FROM observations o JOIN scans s ON s.id=o.scan LEFT JOIN members m ON m.observation=o.id WHERE s.scope=? AND s.namespace=? ORDER BY o.id',
        'SELECT id,created,performer FROM people WHERE scope=? AND namespace=? ORDER BY id'):
        for row in db.execute(sql, (scope, namespace(config))):
            value.update(digest([v.hex() if isinstance(v, bytes) else v for v in row]).encode())
    from .person_signal import available
    if available(db):
        for row in db.execute('SELECT e.observation,e.contract,e.status,e.metadata,e.embedding FROM person_evidence e '
                'JOIN observations o ON o.id=e.observation JOIN scans s ON s.id=o.scan '
                'WHERE s.scope=? AND s.namespace=? ORDER BY e.observation,e.contract',(scope,namespace(config))):
            value.update(digest([v.hex() if isinstance(v,bytes) else v for v in row]).encode())
    from .body_identity import available as bodies_available
    if bodies_available(db):
        for sql in (
            'SELECT b.id,b.scan,b.cue,b.seconds,b.contract,b.face,b.quality,b.rejected,b.embedding,m.person,m.role,m.status,m.policy,m.proof FROM body_observations b JOIN scans s ON s.id=b.scan LEFT JOIN body_members m ON m.observation=b.id WHERE s.scope=? AND s.namespace=? ORDER BY b.id',
            'SELECT c.* FROM body_cues c JOIN scans s ON s.id=c.scan WHERE s.scope=? AND s.namespace=? ORDER BY c.scan,c.contract,c.cue'):
            for row in db.execute(sql,(scope,namespace(config))):
                value.update(digest([v.hex() if isinstance(v,bytes) else v for v in row]).encode())
    return value.hexdigest()


def goal_diagnostics(status):
    """Small provenance summary shared by live winner YAML and final export."""
    return dict(policy=status.get('goal_policy', dict(name='strict', version=1)),
                strict_fully_solved=status.get('strict_fully_solved', status['fully_solved']),
                strict_disjoint=status.get('strict_disjoint', status['disjoint']),
                soft_lost_units=sum(g.get('soft_lost_units', 0) for g in status['groups']),
                soft_budget_excess=sum(g.get('soft_budget_excess', 0) for g in status['groups']),
                **({'scene_exclusions': status['scene_exclusions']} if 'scene_exclusions' in status else {}))
