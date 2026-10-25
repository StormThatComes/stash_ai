"""Small frozen strict assignment problems, with explicit alternative identities.

No greedy winner: only a complete solution separated from every other feasible
solution by the ambiguity margin can resolve a cue. Larger/dense problems keep
the normal conservative path. Recovered rows remain attachment-only evidence.
"""
from itertools import groupby, product
from .clustering import vector, _similarities
from .physical_faces import duplicate
from .database import encoded


def solve(domains, compatible, margin, *, maximum=4096):
    if not domains or any(not domain for domain in domains):return None
    best=second=None;visited=0;exhausted=False
    order=sorted(range(len(domains)),key=lambda i:(len(domains[i]),i))
    assigned={}
    def visit(position,score):
        nonlocal best,second,visited,exhausted
        visited+=1
        if visited>maximum:exhausted=True;return
        if position==len(order):
            assignment=tuple(assigned[i] for i in range(len(domains)))
            if best is None or score>best[0]:second,best=best,(score,assignment)
            elif second is None or score>second[0]:second=(score,assignment)
            return
        i=order[position]
        for choice in domains[i]:
            if all(compatible(i,choice,j,other) if j<i else compatible(j,other,i,choice)
                   for j,other in assigned.items()):
                assigned[i]=choice;visit(position+1,score+choice[1]);del assigned[i]
                if exhausted:return
    visit(0,0.)
    if exhausted or best is None or second is not None and best[0]-second[0]<margin:return None
    return best[1]


def recover(db, policy, history, config):
    import numpy as np
    settings, grouping = config.data['recognition'], config.data['clustering']
    threshold = max(settings['match_threshold'], settings['cooccurrence_merge_threshold'])
    from .recovery_worklist import condition
    rows = db.execute('SELECT o.*,c.id AS candidate FROM candidates c JOIN observations o '
        "ON o.id=json_extract(c.payload,'$.observation') WHERE c.policy=? "
        "AND c.status IN ('pending','withheld') AND json_extract(c.payload,'$.reason') IN "
        "('candidate_group','insufficient_match_or_cohesion','late_strict_identity','ambiguous_identity','co_present_identity_conflict') "
        'AND o.rejected IS NULL AND '+condition(db)+' ORDER BY o.scan,o.cue,o.id', (policy,))
    body_catalogue=None
    for _, cue in groupby(rows, lambda row:(row['scan'],row['cue'])):
        cue = list(cue)
        if not 2 <= len(cue) <= 64:
            continue
        samples = [vector(row) for row in cue]
        fixed = list(db.execute('SELECT o.*,c.person FROM candidates c JOIN observations o '
            "ON o.id=json_extract(c.payload,'$.observation') WHERE c.policy=? "
            "AND c.status='accepted' AND o.scan=? AND o.cue=?", (policy,cue[0]['scan'],cue[0]['cue'])))
        domains = []
        for row, query in zip(cue,samples):
            options = []
            for group in history.groups:
                score, low = _similarities(group, query)
                if score < settings['match_threshold'] or low < grouping['cohesion_threshold']:
                    continue
                if any(other['person'] == group.identifier and
                       not duplicate(row,other,grouping.get('deduplicate_detections',False)) and
                       float(np.dot(query,vector(other))) < threshold for other in fixed):
                    continue
                options.append((group.identifier,score,low))
            domains.append(sorted(options))
        # Never truncate alternatives: an omitted rival could create certainty.
        def compatible(i, a, j, b):
            same = duplicate(cue[i],cue[j],grouping.get('deduplicate_detections',False))
            if same:
                return a[0] == b[0]
            return a[0] != b[0] or min(a[2],b[2],float(np.dot(samples[i],samples[j]))) >= threshold
        solution = solve(domains, compatible, settings['ambiguity_margin'])
        signal='face'
        if solution is None and config.data['sapiensid']['enabled']:
            from .person_signal import contract
            from .hybrid_evidence import catalogue,scores
            if body_catalogue is None:
                scope,namespace=db.execute('SELECT scope,namespace FROM scans WHERE id=?',(cue[0]['scan'],)).fetchone()
                body_catalogue={i.person:i for i in catalogue(db,config,scope,namespace)}
            body_domains=[]
            complete=True
            for row,options in zip(cue,domains):
                stored=db.execute("SELECT embedding FROM person_evidence WHERE observation=? AND contract=? AND status='ready'",
                                  (row['id'],contract(config))).fetchone()
                if stored is None:complete=False;break
                query=np.frombuffer(stored[0],dtype='<f4');choices=[]
                for person,_,low in options:
                    identity=body_catalogue.get(person)
                    if identity is None or not len(identity.persons):complete=False;break
                    from .hybrid_modes import query_score
                    measure=query_score(query[None,:],identity.persons,config.data['sapiensid']['matching'],identity.body_modes)
                    choices.append((person,measure['mean'],low,measure['minimum']))
                if not complete:break
                body_domains.append(choices)
            if complete:
                threshold_body=config.data['sapiensid']['matching']['person_threshold']
                def body_compatible(i,a,j,b):return compatible(i,a,j,b)
                candidate=solve(body_domains,body_compatible,config.data['sapiensid']['matching']['ambiguity_margin'])
                if candidate is not None and all(c[3]>=threshold_body for c in candidate):
                    # Preserve face scores for all existing final face/rival gates.
                    solution=tuple(next(o for o in options if o[0]==chosen[0]) for options,chosen in zip(domains,candidate))
                    signal='face_and_body'
        if solution is None:
            continue
        # Raw rivals are exempt only when another distinct query independently
        # explains that rival in this unique joint assignment.
        from .anchor_views import _competing_anchor
        valid = True
        for i,(person,score,_) in enumerate(solution):
            occupied = {other[0] for j,other in enumerate(solution) if j != i and
                        not duplicate(cue[i],cue[j],grouping.get('deduplicate_detections',False))}
            rivals = [g for g in history.groups if g.identifier not in occupied]
            if _competing_anchor(rivals,person,samples[i],score,settings['ambiguity_margin']):
                valid = False
                break
        if valid:
            from .clustering import co_presence_allows
            for i,(person,_,_) in enumerate(solution):
                group = next(g for g in history.groups if g.identifier == person)
                explained = {cue[j]['id'] for j,other in enumerate(solution) if other[0] != person}
                if not co_presence_allows(db,cue[i],group,history.groups,samples[i],config,
                                          resolved_peers=explained):
                    valid = False
                    break
        if not valid:
            continue
        assignment = [dict(observation=r['id'],identity=s[0],score=s[1]) for r,s in zip(cue,solution)]
        for row,(person,score,low) in zip(cue,solution):
            proof = dict(format=1,frozen=True,context=history.context,identity=person,mode='strict',
                medoids=[],view_evidence=[],mean_score=score,minimum_score=low,
                ambiguity_margin=settings['ambiguity_margin'],joint_assignment=assignment,joint_signal=signal)
            db.execute("UPDATE candidates SET person=?,status='pending',payload=json_set(payload,"
                "'$.reason','late_strict_identity','$.shared_reference',json(?)) WHERE id=?",
                (person,encoded(proof),row['candidate']))
