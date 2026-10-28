"""Pre-optimization exhaustive planner, retained only as a parity oracle."""
import numpy as np
from stash_ai.recognition.body_identity import Scores
from stash_ai.recognition.body_observations import contract
from stash_ai.recognition.database import encoded
from stash_ai.recognition.progress import tracking


def exhaustive(bank,config):
    result=dict(enabled=config.data['sapiensid']['global_matching']['enabled'],contract=contract(config),pairs=[],decisions=[])
    settings=config.data['sapiensid']['global_matching']
    scores=Scores(config);identities=sorted(bank);pairs={};edges={p:set() for p in identities};plan_bytes=0
    with tracking(identities,'Compare global person identities',unit='identity') as people:
        for i,a in enumerate(people):
            for b in identities[i+1:]:
                first,second=bank[a],bank[b]
                mean,minimum=scores.compare(first['refs'],second['refs']);blockers=[]
                if mean<settings['match_threshold']:blockers.append('body_similarity')
                for left in first['cores']:
                    for right in second['cores']:
                        m,n=scores.compare(left,right);minimum=min(minimum,n)
                        if m<settings['match_threshold'] or n<settings['cohesion_threshold']:
                            if 'body_cohesion' not in blockers:blockers.append('body_cohesion')
                if min(first['support'],second['support'])<settings['min_support']:blockers.append('independent_support')
                if first['cues']&second['cues']:blockers.append('person_copresence')
                if first['missing_original'] or second['missing_original']:blockers.append('missing_original_body_witness')
                face_min=None
                for left in first['faces']:
                    for right in second['faces']:
                        if len(left)!=len(right):raise ValueError('Incompatible face namespaces in body consolidation.')
                        scores.count+=1
                        if scores.count>scores.limit:raise ValueError('Face contradiction checks exceed max_candidate_pairs.')
                        score=float(np.dot(left,right));face_min=score if face_min is None else min(face_min,score)
                if face_min is not None and face_min<config.data['sapiensid']['matching']['face_floor']:blockers.append('face_contradiction')
                pair=dict(people=[a,b],score=mean,minimum=minimum,face_minimum=face_min,
                    support=[first['support'],second['support']],
                    face_evidence='compared' if face_min is not None else 'unavailable',blockers=blockers)
                pairs[a,b]=pair;result['pairs'].append(pair)
                if not blockers:edges[a].add(b);edges[b].add(a)
                plan_bytes+=len(encoded(pair))+256
                if plan_bytes>config.data['sapiensid']['execution']['plan_bytes']:
                    raise ValueError('Global body plan exceeds sapiensid.execution.plan_bytes.')
    seen=set()
    for start in identities:
        if start in seen or not edges[start]:continue
        pending=[start];component=set()
        while pending:
            current=pending.pop()
            if current in component:continue
            component.add(current);pending.extend(edges[current]-component)
        seen.update(component);blockers=[];nearest=-1.
        if any(b not in edges[a] for a in component for b in component if a!=b):blockers.append('incomplete_body_component')
        for a in component:
            outsiders=[pairs[tuple(sorted((a,b)))]['score'] for b in identities if b not in component]
            runner=max(outsiders,default=-1.);nearest=max(nearest,runner)
            inside=min(pairs[tuple(sorted((a,b)))]['score'] for b in component if b!=a)
            if inside-runner<settings['ambiguity_margin'] and 'body_ambiguity' not in blockers:blockers.append('body_ambiguity')
        result['decisions'].append(dict(people=sorted(component),outcome='blocked' if blockers else 'merge',
            blockers=blockers,nearest_rival=nearest,
            cores={p:[r['id'] for core in bank[p]['cores'] for r in core] for p in sorted(component)},
            face_witnesses={p:bank[p]['face_ids'] for p in sorted(component)}))
    if len(encoded(result))>config.data['sapiensid']['execution']['plan_bytes']:
        raise ValueError('Global body proof exceeds sapiensid.execution.plan_bytes.')
    return result
