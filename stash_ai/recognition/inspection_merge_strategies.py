"""Assemble measured pair scenarios and select independent controls for replay."""

from . import inspection_track_strategies as tracks
from .body_identity import policy as body_policy
from .clustering import matching_policy
from .inspection_advice import _path
from .inspection_pair_variants import patch_values
from .pose_policy import restricted
from .reference_roles import observation_sql
from .resource_errors import ResourceLimit


def _plans(routes, family):
    return [p for r in routes if r.get('family') == family for p in r['patches']]


def _seed(db, scan, config, source, kind):
    if kind == 'face':
        return db.execute(
            'SELECT o.cue,o.id FROM candidates c JOIN observations o ON o.id='
            + observation_sql('c')
            + " AND o.scan=c.scan WHERE c.scan=? AND c.policy=? AND c.person=? AND c.status='accepted' "
            "AND json_extract(c.payload,'$.face_comparison.stage')='seed' ORDER BY o.cue,o.id LIMIT 1",
            (scan['id'], matching_policy(config), source),
        ).fetchone()
    return db.execute(
        'SELECT b.cue,b.id FROM body_members m JOIN body_observations b ON b.id=m.observation '
        "WHERE b.scan=? AND m.policy=? AND m.person=? AND m.status='accepted' AND m.role='anchor' "
        "AND json_extract(m.proof,'$.reason')='body_scene_core' ORDER BY b.cue,b.id LIMIT 1",
        (scan['id'], body_policy(config), source),
    ).fetchone()


def birth_and_recovery(db, config, scan, routes):
    result = dict(
        title='Prevent a birth and attach its observations to the surviving core',
        strategy='birth_recovery',
        status='Insufficient evidence',
        notes=[],
        patches=[],
        witnesses=[],
    )
    # Every option retains the direction of its measured demotion.
    seen = set()
    for route in routes:
        if not route.get('birth_source') or not route['patches']:
            continue
        source, kind = route['birth_source'], route['birth_kind']
        kinds = ('face', 'body') if kind == 'both' else (kind,)
        for signal in kinds:
            seed = _seed(db, scan, config, source, signal)
            if seed is not None:
                info = f'First recorded {signal} seed/core member for {source}: scene {scan["scene"]}, cue {seed[0]}.'
                if info not in result['notes']:
                    result['notes'].append(info)
                    result['witnesses'].append(dict(label=info, refs=[(signal, seed[1])]))
        for family in ('nominations', 'confirmed_views', 'local_body'):
            for attach in _plans(routes, family):
                if not attach['changes']:
                    continue
                for demote in route['patches']:
                    changes = patch_values(demote)
                    attach_values = patch_values(attach)
                    if any(k in changes and changes[k] != v for k, v in attach_values.items()):
                        continue
                    changes.update(attach_values)
                    if family in ('nominations', 'confirmed_views'):
                        changes['clustering.auxiliary_matching'] = True
                    if family == 'confirmed_views':
                        changes['clustering.view_prototypes_enabled'] = True
                    patch = _path(
                        config,
                        f'{route["title"]} · {demote["title"]} + {family}',
                        changes,
                        evidence=[*demote.get('evidence', ()), *attach.get('evidence', ())],
                    )
                    key = source, patch['yaml']
                    if patch['changes'] and key not in seen:
                        patch.update(witnesses=demote.get('witnesses', []), birth_source=source)
                        result['patches'].append(patch)
                        seen.add(key)
    if result['patches']:
        result['status'] = 'Rebuild hypothesis'
    result['notes'].append('Replay checks the fate of every source observation, including its former core.')
    return result


def local_global(config, report, routes):
    result = dict(
        title='Compare local body then global face/View with local face then global body',
        strategy='local_global',
        status='Insufficient evidence',
        notes=[],
        patches=[],
    )
    if not report or not tracks.enabled(config):
        return result
    grouping = config.data['clustering']
    global_body = config.data['sapiensid']['global_matching']
    if not all(p['reference_samples'] for p in report['identities']):
        result['notes'].append('Both sides need current face references to compare modality order.')
        return result
    if not any(p['person_members'] for p in report['identities']):
        result['notes'].append('No retained body members for the pair.')
        return result
    # In both directions keep the global thresholds at the loaded policy:
    # the local formation/attachment gates are the experiment.
    for family, settings, title in (
        (
            'local_body',
            {
                tracks.LOCAL + 'body_first_matching': True,
                tracks.LOCAL + 'tracklet_matching': True,
                'sapiensid.global_matching.enabled': False,
            },
            'Local body paths, global strict face / View',
        ),
        (
            'local_face',
            {tracks.LOCAL + 'body_first_matching': False, 'sapiensid.global_matching.enabled': True},
            'Local same-scene face support, global body consolidation',
        ),
    ):
        if family == 'local_body' and restricted(grouping):
            settings['clustering.anchor_view_matching'] = True
        for measured in _plans(routes, family) or [dict(changes=[])]:
            changes = {**patch_values(measured), **settings}
            patch = _path(config, title, changes)
            if not patch['notes']:
                patch['mechanism'] = (
                    'global_face_consolidation' if family == 'local_body' else 'global_body_merge'
                )
                result['patches'].append(patch)
    if result['patches']:
        result['status'] = 'Rebuild hypothesis'
    result['notes'].append(
        f'Global face match floor {config.data["recognition"]["match_threshold"]:g}; '
        f'global body match floor {global_body["match_threshold"]:g}. '
        'Inspect the recorded route after replay; other enabled mechanisms may also contribute.'
    )
    return result


def strategies(db, config, scan, ids, routes, catalogue_routes, controls, report):
    extra = list(catalogue_routes)
    for key, title in [
        ('anchor_influence', 'Isolate an influential strict anchor'),
        ('view_bridge', 'Bridge appearances through an intermediate View'),
    ]:
        if not any(r['strategy'] == key for r in extra):
            extra.append(
                dict(
                    strategy=key,
                    title=title,
                    status='Incomplete',
                    notes=['Catalogue measurement unavailable.'],
                    patches=[],
                )
            )
    local_body = next((r for r in routes if r.get('family') == 'local_body'), None)
    if local_body:
        from .inspection_catalogue_strategies import body_anchor_options

        anchor_route = next(r for r in extra if r['strategy'] == 'anchor_influence')
        body_anchor_options(db, config, ids, scan, anchor_route, local_body)
    try:
        extra.extend(tracks.track_strategies(db, config, scan, ids))
    except ResourceLimit as exc:
        extra.extend(
            dict(strategy=key, title=title, status='Incomplete', notes=[str(exc)], patches=[])
            for key, title in tracks.TITLES.items()
        )
    extra.extend([birth_and_recovery(db, config, scan, routes), local_global(config, report, routes)])
    if report:
        rivals = report['independent_person'].get('nearest_competitors', ())
        ranked = [r for r in rivals if r['identity'] in ids and r['rival'] not in ids]
        if ranked:
            controls['body'] = [max(ranked, key=lambda r: (r['score'], r['rival']))['rival']]
        hybrid = [
            r
            for r in report['hybrid'].get('pairs', ())
            if len({r['left'], r['right']}.intersection(ids)) == 1
            and r.get('person')
            and 'face_contradiction' not in r.get('blockers', ())
        ]
        if hybrid:
            rival = max(hybrid, key=lambda r: r['person']['mean'])
            controls['hybrid'] = sorted({rival['left'], rival['right']} - set(ids))
    channels = {
        'strict_core': ('face',),
        'local_face': ('face',),
        'nominations': ('face',),
        'confirmed_views': ('view',),
        'sapiensid_hybrid': ('hybrid', 'body'),
        'independent_person': ('body',),
        'local_body': ('body',),
        'view_bridge': ('view',),
        'anchor_influence': ('face', 'body'),
        'turn': ('body', 'face'),
        'gap': ('body',),
        'crossing': ('body', 'face'),
    }
    evidence = [f'{signal}: {", ".join(people)}' for signal, people in controls.items() if people]
    control_route = dict(
        title='Challenge each strategy with its closest measured rivals',
        strategy='rival_control',
        status='Measured controls' if evidence else 'Insufficient evidence',
        notes=evidence or ['No third identity with comparable retained references was found.'],
        patches=[],
    )
    for route in [*routes, *extra]:
        signals = channels.get(route.get('family') or route.get('strategy'), ('face', 'body', 'view'))
        selected = [p for s in signals for p in controls.get(s, ())]
        for patch in route['patches']:
            patch['controls'] = list(dict.fromkeys([*patch.get('controls', ()), *selected]))
            if patch['controls']:
                patch.setdefault('evidence', []).append(
                    'Protected replay controls: ' + ', '.join(patch['controls'])
                )
    return [*extra, control_route]
