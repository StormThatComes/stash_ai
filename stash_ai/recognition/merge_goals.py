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
    output = json.dumps(result, indent=2) + '\n'
    if args.output:
        with args.output.open('x', encoding='utf-8') as stream:
            stream.write(output)
    else:
        print(output, end='')
    return 0
