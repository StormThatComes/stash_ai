"""Measured independent-person hints; production replay validates every trial."""
from .body_identity import enabled,available
from .body_global import report
from .merge_report_policy import _floor
from .threshold_boundaries import sides


def analyse(db,config,goals):
    output=dict(options={},proposals=[],evidence=dict(enabled=False,pairs=[]))
    if not enabled(config) or not available(db):return output
    owners={m['identity_id']:g.get('index',i) for i,g in enumerate(goals['groups']) for m in g['members']}
    evidence=report(db,config,goals['scope'],goals['namespace'])
    output['evidence']=evidence
    for section in ('identities','global_matching'):
        if section=='global_matching' and not config.data['sapiensid'][section]['enabled']:continue
        for key in ('match_threshold','cohesion_threshold','ambiguity_margin','min_support'):
            value=config.data['sapiensid'][section][key]
            output['options']['sapiensid.'+section+'.'+key]=([max(1,value-1),value+1] if key=='min_support'
                else [max(0,value-.02),max(0,value-.05),min(1,value+.02)])
    proposals=[]
    for pair in evidence['pairs']:
        a,b=pair['people']
        if a not in owners and b not in owners:continue
        pair['relation']='within_set' if a in owners and b in owners and owners[a]==owners[b] else 'competitor'
        prefix='sapiensid.global_matching.'
        for key,field in (('match_threshold','score'),('cohesion_threshold','minimum')):
            output['options'].setdefault(prefix+key,[]).extend(sides(pair[field]))
        if pair['relation']!='within_set' or min(pair['score'],pair['minimum'])<0:continue
        if set(pair['blockers']) & {'person_copresence','face_contradiction','missing_original_body_witness'}:continue
        settings=config.data['sapiensid']['global_matching']
        patch={prefix+'match_threshold':min(settings['match_threshold'],_floor(pair['score'])),
               prefix+'cohesion_threshold':min(settings['cohesion_threshold'],_floor(pair['minimum'])),
               prefix+'min_support':min(settings['min_support'],min(pair['support']))}
        rivals=[p['score'] for p in evidence['pairs'] if set(p['people']) & {a,b} and
                any(x not in owners or owners.get(x)!=owners[a] for x in p['people'])]
        if rivals and pair['score']>=max(rivals):
            patch[prefix+'ambiguity_margin']=min(settings['ambiguity_margin'],_floor(pair['score']-max(rivals)))
        proposals.append(patch)
    if proposals:
        joint={}
        for patch in proposals:
            for key,value in patch.items():joint[key]=min(joint.get(key,value),value)
        output['proposals']=[joint,*proposals]
    return output
