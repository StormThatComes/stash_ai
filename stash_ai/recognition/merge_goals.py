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
    import json
    from .fragmentation_report import _open_readonly
    sets = args.performer_set or [args.performer_id or args.identity_id]
    with closing(_open_readonly(config.database)) as db:
        result = resolve(db, config, sets, scope=args.scope, by_performer=args.identity_id is None)
        if args.solve:
            from .merge_solver import solve
            result.update(solve(config, result, db, maximum=args.max_trials))
        elif args.simulate:
            result['simulation'] = evaluate(config, result, db)
            result['status'] = 'baseline_evaluated'
    from .merge_goal_output import export
    export(config, result, args.output, args.yaml_output)
    return 0


def assess(goals, observations, retained, predicted):
    """Evaluate final memberships, including partial losses and outsider bridges."""
    from collections import Counter
    groups, owners = [], {}
    for group in goals['groups']:
        members, counts, complete_owners = [], Counter(), Counter()
        missing = lost = 0
        for member in group['members']:
            ids = {r['id'] for r in member['observations']}
            selected = ids & observations.keys()
            absent = len(ids - selected)
            withheld = sum(predicted.get(k) is None for k in selected)
            destinations = Counter(predicted[k] for k in selected if predicted.get(k) is not None)
            counts.update(destinations)
            complete = bool(ids) and not absent and not withheld and len(destinations) == 1
            if complete:
                complete_owners.update(destinations.keys())
            missing += absent
            lost += withheld
            members.append(dict(identity_id=member['identity_id'], performer_id=member['performer_id'],
                observations=len(ids), missing=absent, withheld=withheld,
                resulting_identities=dict(destinations), fully_assigned_to_one=complete))
        for destination in counts:
            owners.setdefault(destination, []).append(group['index'])
        largest = max(complete_owners.values(), default=0)
        groups.append(dict(index=group['index'], members=members, missing=missing, withheld=lost,
            resulting_identities=dict(counts), fragments=len(counts),
            largest_complete_performers=largest, total_performers=len(members),
            complete_fraction=largest/len(members),
            fully_merged=largest == len(members) and not missing and not lost))
    conflicts = [dict(identity=k, sets=v) for k, v in sorted(owners.items()) if len(v)>1]
    covered = all(g['members'] and all(m['observations'] for m in g['members']) and
                  not g['missing'] and not g['withheld'] for g in groups)
    return dict(groups=groups, conflicts=conflicts, disjoint=not conflicts, coverage_preserved=covered,
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
            min(g['complete_fraction'] for g in groups),
            sum(g['largest_complete_performers'] for g in groups),
            -sum(g['fragments'] for g in groups))
