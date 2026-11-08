"""Revalidate nominations from split face IDs through the frozen face merge engine."""
import json
from .scan_heads import predicate
from .fragment_endpoints import enabled
from .body_identity import policy


def verified(db, config, scope, namespace, bank, result):
    if not enabled(config.data['sapiensid']['identities']):
        return
    from .database import encoded
    from .reference_roles import strict_sql
    from .pose_policy import anchor_sql
    from .global_views import consolidation_plan
    from .strict_partition import summary_views
    from .clustering import _groups
    from .reference_usage import ReferenceUsage
    from .body_global_search import Search, exact_pair
    from .resource_errors import ResourceLimit
    proposals = {}
    occupied = {p for d in result['decisions'] if d['outcome']=='merge' for p in d['people']}
    rows = db.execute("""SELECT b.scan,j.value FROM body_members m JOIN body_observations b ON b.id=m.observation
        JOIN scans s ON s.id=b.scan JOIN json_each(m.proof,'$.endpoint_pairs') j
        WHERE s.scope=? AND s.namespace=? AND m.policy=? AND """+predicate(),(scope,namespace,policy(config)))
    for scan, raw in rows:
        proof = json.loads(raw)
        a,b = proof['people']
        if a in occupied or b in occupied or a not in bank or b not in bank or (a,b) in proposals:
            continue
        current = dict(db.execute('SELECT rb.id,m.person FROM body_observations rb JOIN observations o ON o.id=rb.face '
            'JOIN members m ON m.observation=o.id WHERE rb.scan=? AND rb.id IN (SELECT value FROM json_each(?)) AND '
            +strict_sql()+' AND '+anchor_sql('o',config.data['clustering']),(scan,encoded(proof['roots']))))
        if [current.get(k) for k in proof['roots']] == [a,a,b,b]:
            proposals[a,b] = proof
            if len(proposals)>config.data['sapiensid']['execution']['max_candidate_pairs']:
                raise ResourceLimit('Too many distinct cross-perspective endpoint nominations.')
    if not proposals:
        return
    usage = ReferenceUsage(config.data['clustering']['max_reference_bytes'])
    groups = _groups(db,scope,namespace,config.data['clustering'],usage=usage)
    focus = {p for pair in proposals for p in pair}
    views = summary_views(groups,usage)
    face_plan = consolidation_plan(db,views,config,focus=focus)
    with Search(db,config,bank) as search:
        for decision in face_plan:
            pair = tuple(sorted(decision['people']))
            if decision['outcome']!='merge' or pair not in proposals:
                continue
            a,b = pair
            evidence = exact_pair(a,b,bank,config,search.evidence)
            if set(evidence['blockers']) - {'body_similarity','body_cohesion'} or evidence['face_minimum'] is None:
                continue
            result['decisions'].append(dict(people=list(pair),outcome='merge',blockers=[],endpoint_pair=proposals[pair],
                cores={p:[[r['id'] for r in core] for core in bank[p]['cores']] for p in pair},
                face_witnesses={p:list(bank[p]['face_ids']) for p in pair}))
