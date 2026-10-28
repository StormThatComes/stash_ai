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
    'body_group_ambiguous':('sapiensid.identities.cohesion_threshold','sapiensid.identities.ambiguity_margin'),
    'ambiguous_or_insufficient_face_body_bank':tuple('sapiensid.identities.'+k for k in
        ('match_threshold','cohesion_threshold','ambiguity_margin','attach_min_support','min_appearance_distance')),
    'body_similarity':('sapiensid.global_matching.match_threshold',),
    'body_cohesion':('sapiensid.global_matching.match_threshold','sapiensid.global_matching.cohesion_threshold'),
    'independent_support':('sapiensid.global_matching.min_support','sapiensid.identities.min_appearance_distance'),
    'body_ambiguity':('sapiensid.global_matching.ambiguity_margin',),
})
HARD = {'missing_person_evidence','missing_original_core_evidence','invalid_alignment',
        'unknown_competitor','no_within_set_query_reference','person_copresence','missing_original_body_witness'}


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
    index(db)  # The solver calls this only on its disposable replay database.
    rows = db.execute('SELECT o.id,c.scan,c.status,c.payload FROM observations o '
        'JOIN candidates c ON c.rowid=(SELECT c2.rowid FROM candidates c2 WHERE '+
        observation_sql('c2')+'=o.id ORDER BY c2.rowid DESC LIMIT 1) '
        'WHERE o.id IN (SELECT value FROM json_each(?)) ORDER BY o.id', (encoded(sorted(observation_ids)),))
    result = []
    for row in rows:
        payload = json.loads(row['payload'])
        result.append(dict(observation=row['id'], scan=row['scan'], status=row['status'],
            reason=payload.get('reason'), hybrid_merge=payload.get('hybrid_merge'),
            hybrid_query=payload.get('hybrid_query'), measurements={k: v for k, v in payload.items()
                if type(v) in (int, float, bool) and any(t in k for t in ('score','similarity','threshold','support','margin'))}))
    from .body_identity import available
    if available(db):
        for row in db.execute('SELECT b.id,b.scan,m.status,m.proof FROM body_members m '
            'JOIN body_observations b ON b.id=m.observation WHERE b.id IN (SELECT value FROM json_each(?)) ORDER BY b.id',
            (encoded(sorted(observation_ids)),)):
            proof=json.loads(row['proof'])
            result.append(dict(observation=row['id'],scan=row['scan'],status=row['status'],
                reason=proof.get('reason'),evidence_kind='person',measurements={k:v for k,v in proof.items()
                    if type(v) in (int,float,bool)}))
    return result


def diagnostic_record(settings, decisions, queries, hybrid, quality):
    trace, active = [], Counter()
    def add(stage, reason, observation=None, **detail):
        parameters = GATES.get(reason, ())
        active.update(parameters)
        trace.append(dict(stage=stage, reason=reason, observation=observation,
            parameters=list(parameters), requires_evidence=reason in HARD, **detail))
    # Candidate reasons are terminal production decisions, not inferred failures.
    terminal = {r['observation']:r for r in decisions}
    for row in terminal.values():
        if row['status'] != 'accepted':
            add('assignment', row['reason'], row['observation'])
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
            add('quality',reason,q['observation'])
        if q.get('missing_measurements'):
            add('quality','missing_measurements',q['observation'], fields=q['missing_measurements'])
    return dict(configuration_sha256=digest(plain(settings.data)),
        evidence_sha256=digest(dict(decisions=decisions,queries=queries,hybrid=hybrid,quality=quality)),
        active_parameters=dict(sorted(active.items())), blocker_trace=trace,
        interpretation='Production decisions and frozen measurements; unknown causes remain explicit.')
