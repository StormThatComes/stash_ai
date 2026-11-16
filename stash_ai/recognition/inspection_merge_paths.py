"""Pair-specific desktop scenarios based on one consistent recognition snapshot."""
from contextlib import closing, contextmanager

from . import merge_report
from .inspection_pair_evidence import find_identity, gate, threshold, frozen_steps, face_witnesses, witness
from .body_identity import available, enabled, policy as body_policy
from .body_observations import contract
from .clustering import matching_policy
from .inspection_advice import _name, _path
from .inspection_anchor_paths import alternatives
from .inspection_runtime import Cancelled, check_cancelled, local, open_readonly, stamp
from .merge_report_policy import parameter
from .resource_errors import ResourceLimit
from .evidence_pixels import alias_scope
from .run_cache import scope as cache_scope
from .scratch_space import scope as scratch_scope
from .threshold_boundaries import at_or_above, at_or_below


ROUTES = {
    'strict_core': ('Merge strict face anchors', 'Compare the two strict face banks, then validate the component against competing identities.'),
    'confirmed_views': ('Merge through confirmed face views', 'Build supported views, compare their witnesses, then check the whole component and its rivals.'),
    'sapiensid_hybrid': ('Merge through face and body evidence', 'Check the face floor, compatible body appearances, independent support and competitors.'),
    'independent_person': ('Merge independent body identities', 'Compare the body cores, then check support, co-presence, face contradictions and outside rivals.'),
}


def _pair(proofs, ids):
    return next((p for p in proofs if set(p.get('people', (p.get('left'), p.get('right')))) == set(ids)), None)


def _measurements(proof):
    keys = ('match_score', 'mean_cross_similarity', 'minimum_cross_similarity', 'threshold',
            'score', 'minimum', 'support', 'face_minimum', 'face', 'person', 'cooccurring', 'medoids')
    for key in keys:
        value = proof.get(key)
        if isinstance(value, dict):
            value = ', '.join(f'{k} {v:.4f}' for k, v in value.items() if k in ('minimum', 'mean') and isinstance(v, (int,float)))
        if value is not None and value != '':
            yield f'{key.replace("_", " ")}: {value:.4f}' if isinstance(value, float) else f'{key.replace("_", " ")}: {value}'


def _catalogue_routes(db, config, report):
    ids = report['identity_ids']
    paths = {path['route']: path for path in report['paths']}
    for name, (title, sequence) in ROUTES.items():
        path = paths.get(name)
        if path is None:
            keys = ['sapiensid.enabled'] if name == 'sapiensid_hybrid' else [
                'sapiensid.enabled', 'sapiensid.collection.enabled', 'sapiensid.identities.enabled', 'sapiensid.global_matching.enabled']
            disabled = [f'{_name(config,key)} = false' for key in keys if not parameter(config.data,key)]
            yield dict(title=title, status='Disabled', notes=[sequence, *disabled], patches=[], steps=[gate('Enable route',False,', '.join(disabled)), gate('Score pair',None,reached=False)])
            continue
        current = path.get('current', {})
        notes = [sequence]
        blockers = list(current.get('blockers', ())) + list(path.get('unresolved_reasons', ()))
        pair_evidence = current.get('pair_evidence', ())
        if name in ('strict_core','confirmed_views'):
            proof = next(iter(pair_evidence), None)
        elif name == 'sapiensid_hybrid':
            proof = _pair(pair_evidence, ids) or _pair(report['hybrid'].get('pairs', ()), ids)
        else:
            proof = _pair(report['independent_person'].get('pairs', ()), ids) or _pair(current.get('pairs', ()), ids)
        if proof:
            notes.extend(_measurements(proof))
            blockers.extend(proof.get('blockers', ()))
        for item in current.get('effective_thresholds', {}).items():
            notes.append(f'Effective {item[0]}: {item[1]:g}')
        component = current.get('shared_component')
        patches = []
        for rec in path['recommendations']:
            shared = rec['verification']['shared_component']
            extra = sorted(set(shared['people']) - set(ids))
            evidence = ['The frozen planner accepts this component.']
            if extra:
                evidence.append('Also merges: ' + ', '.join(extra))
            patches.append(dict(rec, title=rec.get('strategy', 'Current YAML').replace('_',' '), evidence=evidence))
        status = ('Eligible on frozen references' if current.get('can_merge_in_frozen_catalogue') else
                  'Frozen planner verified' if patches else 'Blocked / insufficient evidence')
        if name == 'independent_person':
            full = report['independent_person']
            if not full['enabled']:
                status = 'Disabled'
                notes.append(f"{_name(config,'sapiensid.global_matching.enabled')} = false")
            relevant = [d for d in full.get('decisions', ()) if set(ids) <= set(d['people'])]
            component = next((d for d in relevant if d['outcome']=='merge'), next(iter(relevant),None))
            for decision in relevant:
                blockers.extend(decision['blockers'])
                if decision['outcome'] == 'merge':
                    status = 'Eligible on frozen references'
                    extra = sorted(set(decision['people']) - set(ids))
                    if extra:
                        notes.append('Also merges: ' + ', '.join(extra))
            vetoes = {'person_copresence','face_contradiction','missing_original_body_witness','historical_body_policy_requires_rebuild'}
            if full['enabled'] and not vetoes.intersection(blockers):
                for rec in path.get('proposals', ()):
                    if rec['changes'] and all(p['yaml'] != rec['yaml'] for p in patches):
                        patches.append(dict(rec, title='Measured body proposal', evidence=['Requires a rebuild to verify ownership.']))
                if patches and status != 'Eligible on frozen references':
                    status = 'Rebuild hypothesis'
        if not proof and not patches and status != 'Disabled':
            notes.append('No comparable pair of references was retained for this route.')
        if blockers:
            notes.append('Current blockers: ' + ', '.join(dict.fromkeys(blockers)))
        steps = frozen_steps(config,name,proof,current,blockers,component)
        if status == 'Disabled':
            steps = [gate('Enable route',False),gate('Compare references',None,reached=False)]
        refs = []
        if name in ('strict_core','confirmed_views'):
            refs = face_witnesses(db,config,ids,proof,component)
        elif name == 'sapiensid_hybrid' and proof:
            banks = proof.get('observations', ())
            if len(banks)==2 and all(banks):
                refs.append(witness('Supporting faces','face',banks[0][0],banks[1][0]))
        elif component and all(component.get('cores',{}).get(i) for i in ids):
            refs.append(witness('Supporting body cores','body',*(component['cores'][i][0] for i in ids)))
        yield dict(title=title, status=status, notes=notes, patches=patches, steps=steps, witnesses=refs)


def _local_face(db, config, scan, ids, report):
    from .fragmentation_report import _local_pair, _strict_scene_counts, _recorded_support
    from .clustering import vector
    proof = next(iter(report['paths'][0]['current']['pair_evidence']), None)
    if not proof:
        return []
    references = {p['identity_id']: [(s['observation_id'], vector(db.execute(
        'SELECT dimensions,embedding FROM observations WHERE id=?', (s['observation_id'],)).fetchone()))
        for s in p['reference_samples']] for p in report['identities']}
    pair = dict(cross_mean=proof['mean_cross_similarity'], cross_minimum=proof['minimum_cross_similarity'],
                co_present_observed=proof['cooccurring'])
    local = _local_pair(pair, *ids, config, _strict_scene_counts(db, scan['id'], matching_policy(config), config.data['clustering']))
    settings, grouping = config.data['recognition'], config.data['clustering']
    routes = []
    for same_scene in (False, True):
        support = _recorded_support(db, scan, *ids, references, config, local=same_scene)
        count = support['validated_nominations']
        notes = [f"Stored nominations: {support['stored_nominations']}. Distinct validated timestamps: {count}.",
                 f"Cross mean {pair['cross_mean']:.4f}, weakest comparison {pair['cross_minimum']:.4f}.",
                 'A rebuild can generate different nominations and independent votes.']
        changes = {}
        if same_scene:
            counts = local['strict_accepted_scene_cues']
            notes.append(f"Strict cues in this scene: A {counts['left']}, B {counts['right']}.")
            notes.extend(local['numeric_and_count_blockers'])
            key = 'clustering.same_scene_match_bonus'
            bonus = at_or_above(max(0., settings['match_threshold'] - pair['cross_mean'],
                                       grouping['cohesion_threshold'] - pair['cross_minimum']))
            if min(counts.values()) and 0 < bonus <= .1 and bonus > grouping['same_scene_match_bonus']:
                changes[key] = bonus
            if min(counts.values()) and max(counts.values()) < grouping['same_scene_min_support']:
                changes['clustering.same_scene_min_support'] = max(counts.values())
        elif min(pair['cross_mean'], pair['cross_minimum']) >= 0:
            changes['recognition.match_threshold'] = min(settings['match_threshold'], at_or_below(pair['cross_mean']))
            changes['clustering.cohesion_threshold'] = min(grouping['cohesion_threshold'],
                changes['recognition.match_threshold'], at_or_below(pair['cross_minimum']))
        if 0 < count < settings['min_merge_support']:
            changes['recognition.min_merge_support'] = count
        patches = []
        if changes and not (proof['cooccurring'] and pair['cross_minimum'] < settings['cooccurrence_merge_threshold']):
            patch = _path(config, 'Test the measured nomination gates', changes)
            if patch['changes']:
                patches.append(patch)
        notes.append(f"Review {_name(config, 'recognition.min_merge_support')} = {settings['min_merge_support']}.")
        routes.append(dict(title='Same-scene face core and nominations' if same_scene else 'Merge by independent face nominations',
                           status='Rebuild hypothesis' if patches else 'Measured only', notes=notes, patches=patches,
                           steps=[threshold('Cross mean',pair['cross_mean'],local['effective_match_threshold'] if same_scene else settings['match_threshold']),
                                  threshold('Cross cohesion',pair['cross_minimum'],local['effective_cohesion_threshold'] if same_scene else grouping['cohesion_threshold']),
                                  threshold('Validated nomination timestamps',count,settings['min_merge_support']),
                                  gate('Rebuild ownership',None,reached=False)],
                           witnesses=face_witnesses(db,config,ids,proof,None)))
    return routes


def _local_body_core(db, config, scan, ids):
    """Check both current cores before the global body planner would run."""
    from .body_identity import _load_scene, Scores, copresent
    from .body_guards import FaceGuard
    owners = dict(db.execute(
        "SELECT b.id,m.person FROM body_observations b JOIN body_members m ON m.observation=b.id "
        "WHERE b.scan=? AND b.contract=? AND m.policy=? AND m.status='accepted' AND m.role='anchor' "
        "AND b.rejected IS NULL AND b.embedding IS NOT NULL AND m.person IN (?,?) "
        "AND json_extract(m.proof,'$.reason')='body_scene_core'", (scan['id'], contract(config), body_policy(config), *ids)))
    if len(set(owners.values())) != 2:
        return []
    route = dict(title='Form one local body core', status='Measured only', notes=[], patches=[])
    notes = route['notes']
    try:
        _, rows, _, _, _ = _load_scene(db, scan['id'], config)
        banks = [[r for r in rows if owners.get(r['id']) == person] for person in ids]
        scores = Scores(config)
        guard = FaceGuard(db, config, scores=scores)
        if copresent(*banks):
            route['status'] = 'Blocked'
            right = {r['cue']:r['id'] for r in banks[1]}
            a = next(r for r in banks[0] if r['cue'] in right)
            route['witnesses'] = [witness('Co-present body anchors','body',a['id'],right[a['cue']])]
            notes.append('Anchors from both cores occur in the same cue. Local body grouping excludes co-present observations.')
        elif any(not guard.allows(row, banks[1-i]) for i, bank in enumerate(banks) for row in bank):
            route['status'] = 'Blocked'
            notes.append('The production face guard found incompatible face evidence between these cores.')
        else:
            match = cohesion = float('inf')
            for index, bank in enumerate(banks):
                for row in bank:
                    mean, low = scores.compare([row], banks[1-index])
                    match = min(match, mean)
                    if low < cohesion:
                        cohesion, weakest, opposite = low, row, 1-index
            singleton_banks = ([other] for other in banks[opposite])
            peer = next(other[0] for other in singleton_banks if scores.compare([weakest], other)[1] == cohesion)
            pair = (weakest['id'], peer['id']) if opposite == 1 else (peer['id'], weakest['id'])
            local = config.data['sapiensid']['identities']
            route['steps'] = [gate('Co-presence and face guards',True),
                              threshold('Lowest body query mean',match,local['match_threshold']),
                              threshold('Weakest cross-core similarity',cohesion,local['cohesion_threshold']),
                              gate('Rebuild ownership',None,reached=False)]
            route['witnesses'] = [witness('Weakest cross-core similarity','body',*pair)]
            notes.extend([f"Scene {scan['scene']}: {len(banks[0])} and {len(banks[1])} current body-core anchors.",
                          f"Lowest query mean {match:.4f}, weakest cross-core similarity {cohesion:.4f}.",
                          'Let scene anchors join the same group before publishing two identities. '
                          'Ordering, competing groups and independent support are checked again during rebuild.'])
            if min(match,cohesion) >= 0:
                prefix = 'sapiensid.identities.'
                patch = _path(config, 'Test the local body matching gates',
                    {prefix+k: min(parameter(config.data,prefix+k), at_or_below(v))
                     for k, v in (('match_threshold',match), ('cohesion_threshold',cohesion))})
                if patch['changes']:
                    route.update(status='Rebuild hypothesis', patches=[patch])
                else:
                    notes.append('The measured pair already clears the current numerical gates.')
    except ValueError as error:
        route.update(status='Incomplete', notes=[str(error)])
    return [route]


def _local_body_links(db, config, scan, ids):
    sql = ("SELECT b.cue,b.id,b.face,m.person,f.person AS face_owner,"
           "json_extract(m.proof,'$.track_blocker.reason') AS reason,"
           "json_extract(m.proof,'$.track_blocker.measured') AS measured,"
           "COALESCE(json_extract(m.proof,'$.track_blocker.partner_score'),-1) AS partner_score,peer.id AS peer,pm.person AS peer_owner "
           "FROM body_observations b JOIN body_members m ON m.observation=b.id "
           "LEFT JOIN members f ON f.observation=b.face "
           "LEFT JOIN body_observations peer ON peer.id=json_extract(m.proof,'$.track_blocker.peer') "
           "AND peer.scan=b.scan AND peer.contract=b.contract AND peer.rejected IS NULL "
           "LEFT JOIN body_members pm ON pm.observation=peer.id AND pm.policy=m.policy AND pm.status='accepted' "
           "WHERE b.scan=? AND b.contract=? AND b.rejected IS NULL AND m.policy=? AND m.status='accepted' "
           "AND m.person IN (?,?) ORDER BY b.cue,b.id")
    evidence = {}
    for row in db.execute(sql, (scan['id'], contract(config), body_policy(config), *ids)):
        opposite = ids[1] if row['person'] == ids[0] else ids[0]
        if row['face_owner'] == opposite:
            evidence.setdefault('Face-to-body association', (row, 'Inspect the linked face and pose, then rebuild the face-rooted body assignments.'))
        if row['peer_owner'] == opposite:
            evidence.setdefault('Body track · '+str(row['reason']), (row, 'Inspect the transition and its rival before changing the body track settings.'))
    routes = []
    for title, (row, step) in evidence.items():
        notes = [step, f"Scene {scan['scene']}, cue {row['cue']}, body {row['id']}."]
        patches = []
        if title.startswith('Body track'):
            notes.append(f"Peer {row['peer']}. Recorded blocker: {row['reason']}.")
            key = 'sapiensid.identities.track_similarity_threshold'
            notes.append(f'Review {_name(config,key)} = {parameter(config.data,key)}.')
            if (row['reason'] == 'body_similarity' and isinstance(row['measured'], (int,float))
                    and 0 < row['measured'] < parameter(config.data,key)
                    and row['measured'] - row['partner_score'] >= config.data['sapiensid']['identities']['ambiguity_margin']):
                patch = _path(config, 'Test the observed transition', {key: at_or_below(row['measured'])})
                if patch['changes']:
                    patches.append(patch)
        else:
            notes.append(f"Linked face {row['face']} belongs to the other selected identity.")
        routes.append(dict(title=title, status='Recorded link', notes=notes, patches=patches,
                           witnesses=[witness(title,'body',row['id'],row['peer'] if title.startswith('Body track') else row['face'],
                                              'body' if title.startswith('Body track') else 'face')]))
    return routes


@contextmanager
def _cancellable():
    from stash_ai.runtime.cancellation import Cancellation, Cancelled as ProductionCancelled, use
    token = Cancellation()
    token.event = getattr(local, 'cancel', None) or token.event
    try:
        with use(token):
            yield
    except ProductionCancelled as error:
        raise Cancelled() from error


def analyse(reader, scan, ids):
    if len(set(ids)) != 2 or len(ids) != 2:
        raise ValueError('Choose two different identities.')
    config = reader.config
    revision = stamp(config.database)
    catalogue = reader.catalogue()
    groups = {g['identity']: g for g in catalogue['groups']}
    if catalogue['scan']['id'] != scan['id'] or ids[0] not in groups:
        raise ValueError('The selected scene or identity memberships changed. Refresh the scene.')
    if ids[1] not in groups:
        groups[ids[1]] = find_identity(reader,scan['scope'],ids[1])
        if groups[ids[1]]['identity'] != ids[1]:
            raise ValueError('The selected identity changed. Refresh the scene.')
    labels = {i: f"Performer {groups[i]['performer']}" if groups[i]['performer'] else i for i in ids}
    with (_cancellable(), closing(open_readonly(config.database)) as db, cache_scope(config),
          scratch_scope(config), alias_scope(db, scan['scope'], reader.namespace)):
        if stamp(config.database) != revision:
            raise ValueError('Recognition results changed. Refresh the scene and compare again.')
        check_cancelled()
        try:
            report = merge_report.analyse_pair(db, config, *ids, scope=scan['scope'])
        except ResourceLimit as error:
            coverage = {'catalogue': dict(complete=False, detail=str(error))}
            routes = [dict(title='Catalogue comparison', status='Incomplete', notes=[str(error)], patches=[])]
        else:
            if report['status'] == 'already_merged':
                return dict(routes=[], summary=report['message'])
            if list(ids) != report['identity_ids']:
                raise ValueError('The selected identities have been redirected. Refresh the scene.')
            coverage = report['measurement_coverage']
            routes = list(_catalogue_routes(db, config, report))
            routes.extend(_local_face(db, config, scan, ids, report))
        check_cancelled()
        if enabled(config) and available(db):
            routes.extend(_local_body_core(db, config, scan, ids))
            routes.extend(_local_body_links(db, config, scan, ids))
        routes.extend(alternatives(db, config, scan, ids, labels))
    summary = ('Catalogue routes compare retained references across the scope. Local routes use scene '+scan['scene']+
               '. YAML changes apply globally. Rebuild both identities to check final ownership and unassigned observations.')
    for name, detail in coverage.items():
        if not detail['complete']:
            summary += f" Incomplete {name}: {detail['detail']}"
    if stamp(config.database) != revision:
        summary += ' The database changed during analysis. Refresh to compare the latest results.'
    for route in routes:
        if 'steps' not in route:
            status = route['status']
            route['steps'] = [gate('Retained scene evidence', None if status=='Incomplete' else True),
                              gate('Current route',False if status in ('Blocked','No separating boundary') else None,
                                   ' '.join(route['notes']) if status=='Blocked' else status),
                              gate('Verify final assignment',None,reached=False)]
        route['first_blocker'] = next((s['label'] for s in route['steps'] if s['status']=='Blocked'),None)
    routes.sort(key=lambda r: (r['status'] not in ('Eligible on frozen references', 'Frozen planner verified'),
                               not bool(r['patches']), r['status'] == 'Disabled'))
    return dict(routes=routes, summary=summary, revision=revision)


def verify(config, scope, ids, patch, revision):
    from .inspection_pair_replay import describe, experiment
    result = experiment(config, scope, ids, {'option': patch}, revision)
    return describe(result, 'option')
