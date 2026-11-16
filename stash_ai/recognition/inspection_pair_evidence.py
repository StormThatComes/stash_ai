"""Scoped identities, measured gates and addressable evidence for the pair inspector."""
from contextlib import closing

from .body_identity import available as bodies_available
from .identity_inspection import TABLES
from .inspection_policy import selection
from .inspection_runtime import check_cancelled, open_readonly


def find_identity(reader, scope, value):
    """Resolve an exact Stash ID or UUID with current accepted observations in this scope."""
    with closing(open_readonly(reader.config.database)) as db:
        queries, args = [], [scope, reader.namespace, value, value]
        for kind, (table, members) in TABLES.items():
            if kind == 'body' and not bodies_available(db):
                continue
            current, parameters = selection(reader.config, kind)
            accepted = " AND m.status='accepted'" if kind == 'body' else ''
            queries.append(f'EXISTS (SELECT 1 FROM {members} m JOIN {table} o ON o.id=m.observation '
                           f'JOIN scans s ON s.id=o.scan WHERE m.person=p.id AND s.scope=p.scope '
                           f'AND s.namespace=p.namespace AND {current}{accepted})')
            args.extend(parameters)
        rows = db.execute('SELECT p.id AS identity,p.performer FROM people p WHERE p.scope=? AND p.namespace=? '
                          'AND (p.id=? OR p.performer=?) AND NOT EXISTS (SELECT 1 FROM identity_merges WHERE source=p.id) '
                          'AND ('+' OR '.join(queries)+') LIMIT 2', args).fetchall()
    if len(rows) != 1:
        raise ValueError('No unique current identity matches that ID in this scope. Refresh if memberships changed.')
    return dict(rows[0])


def gate(label, passed, detail='', *, reached=True):
    status = ('Pass' if passed else 'Blocked') if passed is not None else 'Unknown' if reached else 'Not reached'
    return dict(label=label, status=status, detail=detail)


def threshold(label, value, floor):
    if value is None or floor is None:
        return gate(label, None, 'Measurement unavailable')
    return gate(label, value >= floor, f'{value:.5g} / required {floor:.5g}')


def witness(label, kind, left, right, right_kind=None):
    return dict(label=label, refs=[(kind,left), (right_kind or kind,right)])


def frozen_steps(config, route, proof, current, blockers, component):
    """Independent measurements remain visible even if an earlier gate fails."""
    face = route in ('strict_core','confirmed_views')
    face_cfg = config.data['recognition']
    steps = [gate('Eligible pair of references', True if proof else None)]
    if face and proof:
        effective = current['effective_thresholds']
        required = 1 if route=='strict_core' else face_cfg['min_merge_support']
        steps.extend((threshold('Similarity', proof['match_score'], effective['match']),
                      threshold('Independent support', min(proof['support']), required),
                      threshold('Weakest cross comparison', proof['minimum_cross_similarity'], effective['cohesion']),
                      threshold('Cross mean', proof['mean_cross_similarity'], effective['match'])))
        if proof['cooccurring']:
            steps.append(threshold('Co-presence reflection gate', proof['minimum_cross_similarity'],
                                   max(effective['match'],face_cfg['cooccurrence_merge_threshold'])))
    elif proof and route == 'independent_person':
        cfg = config.data['sapiensid']['global_matching']
        steps.extend((threshold('Body similarity',proof.get('score'),cfg['match_threshold']),
                      threshold('Body cohesion',proof.get('minimum'),cfg['cohesion_threshold'])))
    elif proof:
        cfg = config.data['sapiensid']['matching']
        for signal, key in (('face','face_floor'),('person','person_threshold')):
            steps.append(threshold(signal.capitalize()+' cohesion',(proof.get(signal) or {}).get('minimum'),cfg[key]))
    else:
        steps.append(gate('Score the pair',None,reached=False))
    structural = [b for b in blockers if b not in ('no_shared_strong_component','no_shared_supported_component')]
    if structural:
        steps.append(gate('Planner constraints',False,', '.join(dict.fromkeys(structural))))
    rival = (component or {}).get('nearest_rival')
    if isinstance(rival,dict) and 'gap' in rival:
        step = threshold('Lead over nearest rival',rival['gap'],(component or {}).get('ambiguity_margin'))
        step['detail'] += f" · rival {rival.get('rival')} · score {rival.get('score')}"
        steps.append(step)
    steps.append(gate('Accept component', None if component is None else component.get('outcome')=='merge',
                      reached=not any(s['status']=='Blocked' for s in steps)))
    return steps


def face_witnesses(db, config, ids, proof, component):
    import numpy as np
    from .clustering import vector
    from .database import encoded
    from .physical_faces import duplicate
    refs = []
    if proof:
        if proof.get('medoids'):
            refs.append(witness('Compared representatives','face',*proof['medoids']))
        groups = proof.get('support_ids', ())
        if len(groups) == 2 and all(groups):
            addresses = list(dict.fromkeys([*groups[0], *groups[1]]))
            bank = {r['id']:vector(r) for r in db.execute('SELECT id,embedding,dimensions FROM observations '
                    'WHERE id IN (SELECT value FROM json_each(?))',(encoded(addresses),)) if r['embedding']}
            def comparisons():
                for a in groups[0]:
                    check_cancelled()
                    if a in bank:
                        for b in groups[1]:
                            if b in bank:
                                yield float(np.dot(bank[a],bank[b])), a, b
            weakest = min(comparisons(), default=None)
            if weakest and abs(weakest[0]-proof['minimum_cross_similarity']) < 1e-6:
                refs.insert(0,witness('Weakest cross comparison','face',weakest[1],weakest[2]))
        if proof.get('cooccurring'):
            sql = ('SELECT x.id,x.geometry,x.embedding,y.id,y.geometry,y.embedding FROM members a JOIN observations x '
                   'ON x.id=a.observation JOIN observations y ON y.scan=x.scan AND y.cue=x.cue JOIN members b '
                   'ON b.observation=y.id WHERE a.person=? AND b.person=? AND x.id<>y.id '
                   'AND x.rejected IS NULL AND y.rejected IS NULL')
            for a, ga, ea, b, gb, eb in db.execute(sql,ids):
                if not duplicate(dict(geometry=ga,embedding=ea),dict(geometry=gb,embedding=eb),
                                 config.data['clustering'].get('deduplicate_detections',False)):
                    refs.append(witness('Co-presence in one cue','face',a,b))
                    break
    rival = (component or {}).get('nearest_rival')
    if isinstance(rival,dict) and rival.get('medoids'):
        refs.append(witness('Closest rival representatives','face',*rival['medoids']))
    return refs
