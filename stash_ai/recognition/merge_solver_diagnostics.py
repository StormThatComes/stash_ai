"""Configuration-scoped diagnostic hints; never a replacement for production decisions."""
from collections import Counter, defaultdict, deque
from .config import digest, plain
from .database import encoded
from .merge_solver_search import PREFIX

GATES = {'face_floor': (PREFIX+'face_floor',), 'person_threshold': (PREFIX+'person_threshold',),
    'reference_support': (PREFIX+'min_support',), 'query_support': (PREFIX+'min_support',),
    'ambiguity_margin': (PREFIX+'ambiguity_margin',), 'face_contradiction': (PREFIX+'face_floor',),
    'person_below_threshold': (PREFIX+'person_threshold',),
    'insufficient_independent_support': (PREFIX+'min_support',),
    'copresence_conflict': ('recognition.cooccurrence_merge_threshold',),
    'auxiliary_insufficient_support': ('clustering.auxiliary_min_support',),
    'auxiliary_ambiguous': ('recognition.ambiguity_margin',),
    'view_insufficient_support': ('clustering.view_min_support','clustering.view_attach_min_support'),
    'insufficient_match_or_cohesion': ('recognition.match_threshold','clustering.cohesion_threshold')}
GATES.update({
    'body_scene_core':('sapiensid.identities.min_support','sapiensid.identities.min_appearance_distance'),
    'body_anchor_quality':('sapiensid.identities.anchor_min_keypoints','sapiensid.identities.anchor_min_sharpness'),
    'body_anchor_pixels':('sapiensid.identities.anchor_min_person_pixels',),
    'body_anchor_pixels_missing':(),
    'body_group_ambiguous':('sapiensid.identities.cohesion_threshold','sapiensid.identities.ambiguity_margin'),
    'ambiguous_or_insufficient_face_body_bank':tuple('sapiensid.identities.'+k for k in
        ('match_threshold','cohesion_threshold','ambiguity_margin','attach_min_support','min_appearance_distance')),
    'track_body_similarity':('sapiensid.identities.track_similarity_threshold',),
    'body_similarity':('sapiensid.global_matching.match_threshold',),
    'body_cohesion':('sapiensid.global_matching.match_threshold','sapiensid.global_matching.cohesion_threshold'),
    'independent_support':('sapiensid.global_matching.min_support','sapiensid.identities.min_appearance_distance'),
    'body_ambiguity':('sapiensid.global_matching.ambiguity_margin',),
})
HARD = {'body_anchor_keypoint_rules_missing','missing_person_evidence','missing_original_core_evidence','invalid_alignment',
        'track_elapsed_seconds','track_displacement','track_box_scale','track_empty_box','track_uncertain_shot','track_shot_change','track_cut_colour','track_cut_structure','track_cut','track_cue_gap','track_no_face_root','track_different_face_root',
        'unknown_competitor','no_within_set_query_reference','person_copresence','missing_original_body_witness','body_anchor_pixels_missing'}


PHYSICAL_VETOES = {'body_copresence_conflict','person_copresence','different_face_root',
                   'unresolved_partner','weak_face_track_conflict','unconfirmed_contact_face'}
HARD.update(PHYSICAL_VETOES)


def physical_veto(reason):
    return reason in PHYSICAL_VETOES


def round_robin(groups):
    pending = deque(iter(v) for _,v in sorted(groups.items()))
    while pending:
        stream = pending.popleft()
        try:
            yield next(stream)
            pending.append(stream)
        except StopIteration:
            pass


def fair_queries(db, config, goals, selected, births, limit, *, visits=None, priorities=None):
    """Reserve diagnostic queries across sets, performers, then scenes."""
    from .clustering import matching_policy
    from .reference_roles import observation_sql
    from .pose_policy import query_sql
    membership = {o['id']: (g['index'],m['identity_id']) for g in goals['groups']
                  for m in g['members'] for o in m['observations']}
    rows = db.execute('SELECT DISTINCT o.id,o.scan FROM candidates c JOIN observations o ON o.id='+observation_sql('c')+
        " WHERE c.policy=? AND (c.status='withheld' OR c.person IN (SELECT value FROM json_each(?))) "
        'AND o.scan IN (SELECT value FROM json_each(?)) AND o.id IN (SELECT value FROM json_each(?)) '
        'AND o.rejected IS NULL AND '+query_sql('o',config.data['clustering'])+' ORDER BY o.scan,o.cue,o.id',
        (matching_policy(config), encoded(list(births)), encoded([s['scan'] for s in selected]), encoded(sorted(membership))))
    buckets = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    eligible = 0
    for oid,scan in rows:
        group,person = membership[oid]
        buckets[group][person][scan].append(oid)
        eligible += 1
    visits, priorities = visits or {}, priorities or {}
    def order(ids):
        return min((visits.get(oid, 0), -priorities.get(oid, 0), oid) for oid in ids)
    # Rotate at every level, including sets when the quota is smaller than the
    # number of sets. Least-visited evidence wins before priority to avoid starvation.
    def weave(items):
        return round_robin({i: values for i, (_, values) in enumerate(sorted(items, key=lambda x: x[0]))})
    group_streams = []
    for group, members in buckets.items():
        member_streams, group_ids = [], []
        for person, scenes in members.items():
            scene_streams, member_ids = [], []
            for scene, ids in scenes.items():
                ids.sort(key=lambda oid: (visits.get(oid, 0), -priorities.get(oid, 0), oid))
                scene_streams.append((order(ids), ids))
                member_ids.extend(ids)
            member_streams.append((order(member_ids), weave(scene_streams)))
            group_ids.extend(member_ids)
        group_streams.append((order(group_ids), weave(member_streams)))
    ordered = weave(group_streams)
    from itertools import islice
    return set(islice(ordered,max(0,limit))), eligible


def latest_decisions(db, observation_ids):
    """Use the production observation index and explicit rowid ordering.

    Latest means across all policies, just like reference authority. A stale
    policy or a candidate's claimed scan must not hide its terminal decision.
    """
    import json
    from .reference_roles import observation_sql, index
    from .decision_contexts import unpack
    index(db)  # The solver calls this only on its disposable replay database.
    rows = db.execute('SELECT o.id,c.scan,c.person,c.status,c.payload FROM observations o '
        'JOIN candidates c ON c.rowid=(SELECT c2.rowid FROM candidates c2 WHERE '+
        observation_sql('c2')+'=o.id ORDER BY c2.rowid DESC LIMIT 1) '
        'WHERE o.id IN (SELECT value FROM json_each(?)) ORDER BY o.id', (encoded(sorted(observation_ids)),))
    result = []
    for row in rows:
        payload = unpack(db, row['payload'])
        merge_proofs = [payload[k] for k in ('strict_partition', 'global_view_merge', 'final_subcomponent_merge')
                        if isinstance(payload.get(k), dict)]
        evidence = payload.get('hybrid_query') or payload.get('view_prototype') or payload.get('strict_confirmation') or {}
        samples = evidence.get('core_observations') or evidence.get('support_examples') or evidence.get('support_ids') or ()
        scores = payload.get('scores') or ()
        measurements = {k:v for k,v in payload.items() if type(v) in (int,float,bool)
                        and any(t in k for t in ('score','similarity','threshold','support','margin'))}
        for key,value in [('score',evidence.get('score',evidence.get('match_score',scores[0] if scores else None))),
                          ('runner',evidence.get('rival_score',scores[1] if len(scores)>1 else None))]:
            if type(value) in (int,float):
                measurements[key] = value
        result.append(dict(observation=row['id'], scan=row['scan'], status=row['status'],
            owner=row['person'], witnesses=list(samples)[:8],
            reason=payload.get('reason'), hybrid_merge=payload.get('hybrid_merge'),
            face_view=any(p.get('authority') == 'strict_view' for p in merge_proofs),
            cooccurring_merge=any(p.get('cooccurring') for p in merge_proofs),
            hybrid_query=payload.get('hybrid_query'), measurements=measurements))
    from .body_identity import available
    if available(db):
        for row in db.execute('SELECT b.id,b.scan,m.status,m.person,m.proof FROM body_members m '
            'JOIN body_observations b ON b.id=m.observation WHERE b.id IN (SELECT value FROM json_each(?)) ORDER BY b.id',
            (encoded(sorted(observation_ids)),)):
            proof=json.loads(row['proof'])
            tracks = (proof.get('source_tracks') or [])[:8]
            witnesses = list(dict.fromkeys(oid for track in tracks for oid in track)) if tracks else (proof.get('witness_ids') or [])[:8]
            result.append(dict(observation=row['id'],scan=row['scan'],status=row['status'],
                reason=proof.get('reason'),evidence_kind='person',owner=row['person'],
                witnesses=witnesses, **({'source_tracks':tracks} if tracks else {}), rival=proof.get('rival'),
                measurements={k:v for k,v in proof.items()
                    if type(v) in (int,float,bool)},
                turn_evidence={k:proof[k] for k in ('weakest_edge','rival_observation','view_rival','view_rival_observation','view_witnesses','view_roots') if k in proof},
                scene_probe=proof.get('scene_probe'), body_global=bool(proof.get('global_merge')), endpoint_body_gate=proof.get('endpoint_body_gate',False),
                face_endpoints='fragment_face_endpoints' in (proof.get('route'),proof.get('reason')),
                used_scene_bonus=bool(proof.get('roster_context') or proof.get('reason')=='external_rear_view' and proof.get('shot_context')),
                **({'track_blocker':proof['track_blocker']} if proof.get('track_blocker') else {}),
                **({'keypoint_rules':proof['anchor_keypoint_rules']} if 'anchor_keypoint_rules' in proof else {})))
    return result


def diagnostic_record(settings, decisions, queries, hybrid, quality, *, wrong=None):
    trace, active = [], Counter()
    def add(stage, reason, observation=None, **detail):
        parameters = GATES.get(reason, ())
        active.update(parameters)
        trace.append(dict(stage=stage, reason=reason, observation=observation,
            parameters=list(parameters), requires_evidence=reason in HARD, **detail))
    # Candidate reasons are terminal production decisions, not inferred failures.
    terminal = {r['observation']:r for r in decisions}
    for row in terminal.values():
        conflict = (wrong or {}).get(row['observation'])
        if conflict and row['status'] == 'accepted':
            measured = row.get('measurements', {})
            add('assignment', 'accepted_wrong_owner', row['observation'],
                route=row['reason'], witnesses=row.get('witnesses', []),
                **({'source_tracks':row['source_tracks']} if row.get('source_tracks') else {}),
                owner=row.get('owner'), rival=row.get('rival'), **conflict,
                score=measured.get('score',measured.get('track_floor')), runner=measured.get('runner'),
                threshold=measured.get('threshold'), required_margin=measured.get('required_margin'),
                margin=(measured['score'] - measured['runner']
                        if 'score' in measured and 'runner' in measured else None))
        if row['status'] != 'accepted':
            blocker = row.get('track_blocker')
            if blocker:
                details = {k:blocker[k] for k in ('scene','cue','observation_cue','rival','peer','measured','limit','partner_score','partner_observation') if k in blocker}
                add('body_track','track_' + blocker['reason'],row['observation'],**details)
            add('assignment', row['reason'], row['observation'],
                **({'keypoint_rules':row['keypoint_rules']} if 'keypoint_rules' in row else {}))
    for row in queries.get('decisions', ()):
        add('hybrid_query.'+row['stage'], row['reason'], row['observation'], details=row.get('details', {}))
    for q in queries.get('queries', ()):
        for reason in q['blockers']:
            add('hybrid_query.measurement',reason,q['observation'], target=q.get('identity'))
    for p in hybrid.get('pairs', ()):
        if p['relation'] == 'within_set':
            for reason in p['blockers']:
                add('hybrid_merge',reason,left=p['left'],right=p['right'])
    for p in hybrid.get('independent_person',{}).get('pairs',()):
        if p.get('relation')=='within_set':
            for reason in p['blockers']:
                add('body_global',reason,left=p['people'][0],right=p['people'][1])
    for q in quality:
        active.update(q.get('patch', {}).keys())
        for reason in q.get('blockers', ()):
            add('quality',reason,q['observation'], **({'keypoint_rules':q['keypoint_rules']}
                if reason.startswith('body_anchor_keypoint_rules') and 'keypoint_rules' in q else {}))
        if q.get('missing_measurements'):
            add('quality','missing_measurements',q['observation'], fields=q['missing_measurements'])
    return dict(configuration_sha256=digest(plain(settings.data)),
        evidence_sha256=digest(dict(decisions=decisions,queries=queries,hybrid=hybrid,quality=quality)),
        active_parameters=dict(sorted(active.items())), blocker_trace=trace,
        interpretation='Production decisions and frozen measurements; unknown causes remain explicit.')
