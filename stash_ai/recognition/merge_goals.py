"""Explicit merge goals resolved against one retained read-only snapshot."""
from .identity_report import _select
from .models import namespace


def resolve(db, config, sets, *, scope=None, by_performer=True):
    if not sets or any(not group for group in sets):
        raise ValueError('Merge goals require nonempty sets.')
    groups, owners = [], {}
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
        for identity, item in members.items():
            item['observations'] = [dict(r) for r in db.execute(
                'SELECT o.id,s.id AS scan,s.scene,o.cue FROM members m JOIN observations o ON o.id=m.observation '
                'JOIN scans s ON s.id=o.scan WHERE m.person=? ORDER BY o.id', (identity,))]
        groups.append(dict(index=index, members=list(members.values())))
    return dict(format=2, kind='recognition_merge_goals', scope=scope, namespace=namespace(config),
                groups=groups, status='goals_resolved', recommendations=[])


def execute(args, config):
    from contextlib import closing
    from .fragmentation_report import _open_readonly
    targets = [p.expanduser().resolve() for p in (args.output, args.yaml_output) if p]
    if len(set(targets)) != len(targets) or any(p.exists() for p in targets):
        raise FileExistsError('Outputs must be distinct new files.')
    sets = args.performer_set or [args.performer_id or args.identity_id]
    with closing(_open_readonly(config.database)) as db:
        result = resolve(db, config, sets, scope=args.scope, by_performer=args.identity_id is None)
        result['snapshot_sha256'] = fingerprint(db, config, result['scope'])
        if args.solve:
            from .merge_solver import solve
            import json
            import sys
            def progress(name, stage, item):
                if args.progress_style != 'off':
                    detail = ''
                    if stage == 'start':
                        detail = (' (input configuration)' if item['previous_trial'] is None else
                                  f" changes_vs={item['previous_trial']} parent={item['search_parent']}")
                    elif item is not None:
                        counts = [(g['largest_complete_performers'], g['total_performers'])
                                  for g in item['goal_result']['groups']]
                        detail = (f" {counts} disjoint={item['goal_result']['disjoint']} "
                                  f"best={item['best_trial']} best_number={item['best_trial_number']}")
                    print(f'Merge solver {name}: {stage}{detail}', file=sys.stderr, flush=True)
                    if stage == 'start':
                        for change in item['changes_from_previous']:
                            print(f"  {change['parameter']}: {json.dumps(change['previous'])} -> "
                                  f"{json.dumps(change['current'])}", file=sys.stderr, flush=True)
            result.update(solve(config, result, db, maximum=args.max_trials, progress=progress))
        elif args.simulate:
            result['simulation'] = evaluate(config, result, db)
            result['status'] = 'baseline_evaluated'
    from .merge_goal_output import export
    export(config, result, args.output, args.yaml_output)
    if args.solve and args.progress_style != 'off':
        from .merge_goal_output import summary
        print(summary(result), file=sys.stderr, flush=True)
    return 130 if result.get('interrupted') else 2 if result.get('status') == 'no_compatible_scans' else 0


def assess(goals, observations, retained, predicted):
    """Evaluate final memberships, including partial losses and outsider bridges."""
    from collections import Counter
    groups, owners = [], {}
    goal_ids = {r['id'] for g in goals['groups'] for m in g['members'] for r in m['observations']}
    for group in goals['groups']:
        members, counts, complete_owners = [], Counter(), Counter()
        destination_fractions = Counter()
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
                    destination_fractions[destination] += count / len(ids)
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
    return dict(groups=groups, conflicts=conflicts,
                cross_set_observation_pairs=sum(c['observation_pairs'] for c in conflicts),
                disjoint=not conflicts, coverage_preserved=covered,
                admissible=covered and not conflicts,
                fully_solved=covered and not conflicts and all(g['fully_merged'] for g in groups))


def evaluate(config, goals, db, variants=None):
    from .replay_simulation import replay
    return replay(config, variants or {}, scope=goals['scope'], source=db,
        consume=lambda name, observations, retained, predicted:
            dict(goal_result=assess(goals, observations, retained, predicted)))


def objective(status):
    """Fair lexicographic objective; larger sets cannot hide an untouched smaller set."""
    groups = status['groups']
    return (status['admissible'], status['fully_solved'],
            status['disjoint'],
            min(g['assigned_fraction'] for g in groups),
            sum(g['assigned_fraction'] for g in groups)/len(groups),
            -status['cross_set_observation_pairs'],
            min(g['complete_fraction'] for g in groups),
            sum(g['largest_complete_performers'] for g in groups),
            min(g['consolidation_fraction'] for g in groups),
            sum(g['consolidation_fraction'] for g in groups)/len(groups),
            -sum(g['fragments'] for g in groups))


def unmet_conditions(status):
    """Every unsatisfied goal, including empty members and cross-set merges."""
    reasons = []
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
    return value.hexdigest()
