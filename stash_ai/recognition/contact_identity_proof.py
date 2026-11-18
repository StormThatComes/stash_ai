"""Correct a contact body link only between independently measured clean roots."""
from .body_numeric import body_similarity
import json
from .contact_anchors import crowded_link


def correct(rows,graph,settings,scores):
    if not settings.get('correct_contact_face_links',False) or not graph:return rows
    from .independent_evidence import count
    clean=[r for r in rows if r.get('face_anchor') and not crowded_link(json.loads(r['quality']).get('quality',{}))]
    clean_ids={r['id'] for r in clean}
    repairs={}
    for chain,_ in graph:
        endpoints=[(i,r) for i,r in enumerate(chain) if r['id'] in clean_ids]
        for (a,left),(b,right) in zip(endpoints,endpoints[1:]):
            owner=left.get('face_person')
            if not owner or owner!=right.get('face_person') or right['cue']-left['cue']>settings.get('contact_max_cues',12):continue
            refs=[dict(scan=r['scan'],seconds=r['seconds'],quality=r.get('face_quality') or '{}') for r in (left,right)]
            if count(refs)<2:continue
            for row in chain[a+1:b]:
                if not row.get('face_person') or row['face_person']==owner:continue
                if not crowded_link(json.loads(row['quality']).get('quality',{})):continue
                support=min(body_similarity(scores,row,r) for r in (left,right))
                rival=max((body_similarity(scores,row,r) for r in clean if r['face_person']!=owner),default=-1.)
                if support<max(settings['match_threshold'],settings['cohesion_threshold']) or support-rival<settings['ambiguity_margin']:continue
                repairs[row['id']]=dict(row,face_person=owner,face_anchor=False,_face_vector=None,
                    _contact_correction=dict(original_owner=row['face_person'],owner=owner,
                        witness_ids=[left['id'],right['id']],score=support,rival=rival,
                        original_anchor=bool(row.get('face_anchor')),reference_scope='scene'))
    return [repairs.get(r['id'],r) for r in rows]


def birth_ready(group,settings,independent):
    if not settings.get('defer_contact_births',False):return True
    quality=lambda r:json.loads(r['quality']).get('quality',{})
    if not any(crowded_link(quality(r)) for r in group):return True
    clean=[r for r in group if not crowded_link(quality(r)) and not quality(r).get('query_only')]
    return len(independent(clean,settings))>=settings.get('contact_birth_support',2)


def rebind(graph, rows):
    """All graph consumers must see corrected ownership, without changing edges."""
    if graph is None:
        return None
    current = {row['id']:row for row in rows}
    return [([current[row['id']] for row in chain], strengths) for chain,strengths in graph]


def pending_proof(group, settings, independent):
    from .config import digest
    clean = [r for r in group if not crowded_link(json.loads(r['quality']).get('quality',{}))
             and not json.loads(r['quality']).get('quality',{}).get('query_only')]
    witnesses = sorted(r['id'] for r in independent(clean,settings))
    return dict(required_clean_support=settings.get('contact_birth_support',2),
                clean_witness_ids=witnesses, evidence_generation=digest([
                    (r['id'],r['quality'],r.get('face_person')) for r in sorted(group,key=lambda r:r['id'])]))
