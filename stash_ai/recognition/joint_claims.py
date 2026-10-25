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
    size = 1
    for domain in domains:
        size *= len(domain)
    if not domains or not size or size > maximum:
        return None
    best = second = None
    for assignment in product(*domains):
        if any(not compatible(i, a, j, b) for i,a in enumerate(assignment)
               for j,b in enumerate(assignment[:i])):
            continue
        score = sum(item[1] for item in assignment)
        if best is None or score > best[0]:
            second, best = best, (score, assignment)
        elif second is None or score > second[0]:
            second = score, assignment
    if best is None or second is not None and best[0] - second[0] < margin:
        return None
    return best[1]


def recover(db, policy, history, config):
    import numpy as np
    settings, grouping = config.data['recognition'], config.data['clustering']
    threshold = max(settings['match_threshold'], settings['cooccurrence_merge_threshold'])
    rows = db.execute('SELECT o.*,c.id AS candidate FROM candidates c JOIN observations o '
        "ON o.id=json_extract(c.payload,'$.observation') WHERE c.policy=? "
        "AND c.status IN ('pending','withheld') AND json_extract(c.payload,'$.reason') IN "
        "('candidate_group','insufficient_match_or_cohesion','late_strict_identity') "
        'AND o.rejected IS NULL ORDER BY o.scan,o.cue,o.id', (policy,))
    for _, cue in groupby(rows, lambda row:(row['scan'],row['cue'])):
        cue = list(cue)
        if not 2 <= len(cue) <= 6:
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
        if any(len(domain)>3 for domain in domains):
            continue
        def compatible(i, a, j, b):
            same = duplicate(cue[i],cue[j],grouping.get('deduplicate_detections',False))
            if same:
                return a[0] == b[0]
            return a[0] != b[0] or float(np.dot(samples[i],samples[j])) >= threshold
        solution = solve(domains, compatible, settings['ambiguity_margin'])
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
        if not valid:
            continue
        assignment = [dict(observation=r['id'],identity=s[0],score=s[1]) for r,s in zip(cue,solution)]
        for row,(person,score,low) in zip(cue,solution):
            proof = dict(format=1,frozen=True,context=history.context,identity=person,mode='strict',
                medoids=[],view_evidence=[],mean_score=score,minimum_score=low,
                ambiguity_margin=settings['ambiguity_margin'],joint_assignment=assignment)
            db.execute("UPDATE candidates SET person=?,status='pending',payload=json_set(payload,"
                "'$.reason','late_strict_identity','$.shared_reference',json(?)) WHERE id=?",
                (person,encoded(proof),row['candidate']))
