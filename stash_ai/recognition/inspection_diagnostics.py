"""Explain retained decisions and prioritize review without changing assignments."""
from collections import Counter
from contextlib import closing
import heapq
import json
import math

from .inspection_runtime import open_readonly, check_cancelled, stamp
from .admission_panel import SOURCES


def number(value):
    return type(value) in (int,float) and math.isfinite(value)


def stages(detail):
    """Provenance belongs to every stage: current gates are not historic decisions."""
    result=[]
    for group in detail.get('admission',{}).get('groups',[]):
        checks=[dict(label=c['parameter'],status=c['status'],value=f"{c['measured'] if c['measured'] is not None else 'unknown'} {c['operator']} {c['threshold']}",
                     note=c.get('note','')) for c in group['checks']]
        result.append(dict(label=group['stage'],status=group['status'],source=SOURCES[group['configuration_source']],
                           note=group.get('note',''),children=checks))
    trace=detail['trace']
    for decision in trace['decisions']:
        children=[dict(label=k,status='recorded',value=str(v),note='') for k,v in decision['metrics'].items()]
        children.extend(dict(label=r['label'],status='evidence',value=r['observation'],note='',reference=r)
                        for r in decision['references'])
        for merge in decision['merges']:
            children.append(dict(label='Identity merge',status='recorded',value=f"{merge.get('source') or 'unknown'} → {merge.get('target') or 'survivor'}",
                                 note=merge.get('route') or 'Route unavailable'))
        relation={'recorded_current_policy':'current','recorded_other_policy':'historical'}.get(trace['evidence_status'])
        if relation is None:
            record=next((r for r in detail.get('recorded_decisions',[]) if r['policy']==decision['policy']),None)
            relation=('current' if record['current'] else 'historical') if record else 'relation unknown'
        result.append(dict(label=decision['reason'],status='recorded',source=f'Retained matching policy ({relation}): '+decision['policy'],
                           note=decision['summary'],children=children))
    result.append(dict(label='Membership',status=trace['status'],source=trace['evidence_status'],note=trace['role'],children=[]))
    return result


def contribution(detail):
    trace=detail['trace']
    lines=['Recorded membership: '+trace['role']+'. Membership alone does not prove reference-bank use.']
    for d in trace['decisions']:
        selected=d.get('reference_selected')
        value='yes' if selected is True else 'no' if selected is False else 'not retained'
        lines.append(f"{d['reason']}: recorded reference role {d.get('reference_role') or 'not retained'}; bank selection {value}. Policy: {d['policy']}.")
        lines.append(f"{len(d['references'])} retained evidence links for this decision; {len(d['merges'])} recorded merge edges.")
    if not trace['decisions']:lines.append('No retained decision establishes reference contribution.')
    lines.append('Bank selection is not a count of later comparisons. Quality gates above describe eligibility, not actual use.')
    return '\n'.join(lines)


def retained_rivals(detail):
    result=[]
    for d in detail['trace']['decisions']:
        metrics=d['metrics']
        prefixes={k.rsplit('.',1)[0]+'.' if '.' in k else '' for k in metrics}
        for prefix in sorted(prefixes):
            score=metrics.get(prefix+'runner',metrics.get(prefix+'rival_score'))
            rival=metrics.get(prefix+'rival_cluster',metrics.get(prefix+'rival'))
            # A rival identifier is never inferred from a runner-up score.
            if rival is not None or number(score) and score!=-1:
                result.append(dict(route=prefix.rstrip('.') or d['reason'],identity=rival if isinstance(rival,str) else None,
                                   score=score if number(score) else None,policy=d['policy']))
        scores=metrics.get('recorded_scores')
        if isinstance(scores,list) and len(scores)>1 and number(scores[1]):
            result.append(dict(route=d['reason'],identity=None,score=scores[1],policy=d['policy']))
    return result


def review_reasons(row, config):
    """Integer review priority, deliberately not a calibrated error probability."""
    reasons=[]
    if row['status']!='accepted':return reasons
    margin=row.get('margin')
    if number(margin) and margin<=.03:reasons.append((4,f'Recorded top-two lead {margin:.4f} ≤ 0.03'))
    if row.get('rejected'):reasons.append((5,'Accepted membership retains rejection: '+row['rejected']))
    if row.get('policy_state')=='historical':reasons.append((2,'Decision uses a different matching policy'))
    elif row.get('policy_state')=='missing':reasons.append((2,'No retained matching decision'))
    if row['state']=='unknown':reasons.append((1,'Reference role cannot be established'))
    if row['kind']=='face':
        settings=config.data['clustering'];floor=settings['min_auxiliary_sharpness'] or 0
        if row['state']=='anchor':floor=max(floor,settings['min_anchor_sharpness'] or 0)
        sharp=row.get('sharpness')
        if number(sharp) and sharp<floor:reasons.append((3,f'Retained sharpness {sharp:g} below current role floor {floor:g}'))
    return reasons


def review_queue(scene_reader, *, limit=500):
    from .inspection_search import SearchReader
    revision=stamp(scene_reader.config.database)
    source=SearchReader(scene_reader)
    top=[];seen=flagged=0
    for row in source.iter_rows():
        if row['status']!='accepted':continue
        seen+=1;reasons=review_reasons(row,scene_reader.config)
        if not reasons:continue
        flagged+=1
        score=sum(p for p,_ in reasons)
        record=dict(row,priority=score,reasons=[text for _,text in reasons])
        rank=(score,row['kind'],row['id'])
        if len(top)<limit:heapq.heappush(top,(rank,record))
        elif rank>top[0][0]:heapq.heapreplace(top,(rank,record))
    rows=[dict(row,reader=source.owner(row)) for _,row in sorted(top,reverse=True)]
    return dict(rows=rows,seen=seen,flagged=flagged,changed=revision!=stamp(scene_reader.config.database))


def unassigned_summary(scene_reader):
    """Stream metadata once; bounded aggregate rows, no images or embeddings."""
    from .scene_inspection import TABLES, _unassigned_query
    from .body_observations import available
    from .admission_report import Policies, face, person
    config=scene_reader.config;revision=stamp(config.database)
    scan=scene_reader.catalogue()['scan'];counts=Counter();gates={};total=0
    with closing(open_readonly(config.database)) as db:
        policies=Policies(db,config,scan['scope'],scan['namespace'])
        for kind,(observations,_) in TABLES.items():
            if kind=='body' and not available(db):continue
            query='SELECT r.*,o.quality,o.geometry'+(',o.contract' if kind=='body' else '')+' FROM ('+_unassigned_query(kind)+') r JOIN '+observations+' o ON o.id=r.id'
            for row in db.execute(query,(scan['scene'],scan['id'])):
                check_cancelled();total+=1;counts[kind,row['state']]+=1
                quality=json.loads(row['quality'])
                admission=(face(quality,json.loads(row['geometry']),config,policies,scan['policy_sha']) if kind=='face'
                           else person(quality,config,policies,row['contract'],independent=True))
                for group in admission['groups']:
                    for check in group['checks']:
                        key=kind,group['stage'],check['parameter'],group['configuration_source'],str(check['threshold']),check['operator']
                        item=gates.setdefault(key,dict(kind=kind,stage=group['stage'],parameter=check['parameter'],source=SOURCES[group['configuration_source']],
                            threshold=check['threshold'],operator=check['operator'],counts=Counter(),minimum=None,maximum=None,closest_failure=None))
                        item['counts'][check['status']]+=1
                        value=check['measured']
                        if number(value):
                            item['minimum']=value if item['minimum'] is None else min(value,item['minimum'])
                            item['maximum']=value if item['maximum'] is None else max(value,item['maximum'])
                        if check['status']=='fail' and number(check['margin']):
                            distance=abs(check['margin'])
                            item['closest_failure']=distance if item['closest_failure'] is None else min(distance,item['closest_failure'])
    states=[dict(kind=k,state=s,count=n) for (k,s),n in sorted(counts.items(),key=lambda p:(-p[1],p[0]))]
    return dict(total=total,states=states,gates=sorted(gates.values(),key=lambda g:(-g['counts']['fail'],g['kind'],g['stage'],g['parameter'])),
                changed=revision!=stamp(config.database))
