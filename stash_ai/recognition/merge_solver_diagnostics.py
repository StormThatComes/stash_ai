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
HARD = {'missing_person_evidence','missing_original_core_evidence','invalid_alignment',
        'unknown_competitor','no_within_set_query_reference'}


def round_robin(groups):
    pending = deque(iter(v) for _,v in sorted(groups.items()))
    while pending:
        stream = pending.popleft()
        try:
            yield next(stream)
            pending.append(stream)
        except StopIteration:
            pass


def fair_queries(db, config, goals, selected, births, limit):
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
    ordered = round_robin({g:round_robin({p:round_robin(scenes) for p,scenes in members.items()})
                          for g,members in buckets.items()})
    from itertools import islice
    return set(islice(ordered,max(0,limit))), eligible


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
